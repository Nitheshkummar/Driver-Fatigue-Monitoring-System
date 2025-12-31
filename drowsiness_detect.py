'''Enhanced Driver Fatigue & Attention Monitoring System
Features:
- Eye Aspect Ratio (EAR) for drowsiness detection
- Head Pose Estimation (left/right/up/down)
- Eye Gaze Direction (left/right/center)
- Yawn Detection using Mouth Aspect Ratio (MAR)
- Blink Rate Monitoring (blinks per minute)
- Fatigue Score (0-100%) combining all signals
- Real-time FPS display
'''

# Import necessary libraries
from scipy.spatial import distance
from imutils import face_utils
import numpy as np
import pygame
import time
import dlib
import cv2
import math

# ===================== CONFIGURATION =====================
# FPS Control - Set your desired FPS here
TARGET_FPS = 30  # Change this to control processing speed

# Detection Thresholds
EYE_ASPECT_RATIO_THRESHOLD = 0.25  # Below this = eyes closed
MOUTH_ASPECT_RATIO_THRESHOLD = 0.6  # Above this = yawning
HEAD_TURN_THRESHOLD = 15  # Degrees - head turn alert threshold
GAZE_THRESHOLD = 0.35  # Eye gaze off-center threshold

# Alert Thresholds
DROWSY_FRAMES_THRESHOLD = 50  # Frames with closed eyes to trigger drowsy alert
ATTENTION_FRAMES_THRESHOLD = 45  # Frames looking away to trigger attention alert
YAWN_FRAMES_THRESHOLD = 15  # Consecutive frames to count as one yawn

# Fatigue Score Weights (must sum to 100)
WEIGHT_DROWSINESS = 40
WEIGHT_YAWN = 25
WEIGHT_BLINK = 20
WEIGHT_ATTENTION = 15

# ===================== INITIALIZATION =====================
# Initialize Pygame for sound
pygame.mixer.init()
alert_sound = pygame.mixer.Sound('audio/alert.wav')
alert_sound.set_volume(1.0)

# State variables
DROWSY_COUNTER = 0
ATTENTION_COUNTER = 0
YAWN_COUNTER = 0
BLINK_COUNTER = 0
TOTAL_BLINKS = 0
TOTAL_YAWNS = 0
EYE_CLOSED_PREV = False
YAWNING_PREV = False
ALARM_ON = False

# Fatigue tracking
fatigue_score = 0
start_time = time.time()

# FPS calculation
fps_start_time = time.time()
fps_frame_count = 0
current_fps = 0

# 3D model points for head pose estimation
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip (30)
    (0.0, -330.0, -65.0),        # Chin (8)
    (-225.0, 170.0, -135.0),     # Left eye left corner (36)
    (225.0, 170.0, -135.0),      # Right eye right corner (45)
    (-150.0, -150.0, -125.0),    # Left mouth corner (48)
    (150.0, -150.0, -125.0)      # Right mouth corner (54)
], dtype=np.float64)

# Load face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Get landmark indices
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS['mouth']

# ===================== HELPER FUNCTIONS =====================

def eye_aspect_ratio(eye):
    """Calculate Eye Aspect Ratio (EAR)"""
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    """Calculate Mouth Aspect Ratio (MAR) for yawn detection"""
    # Vertical distances
    A = distance.euclidean(mouth[2], mouth[10])  # 51, 59
    B = distance.euclidean(mouth[4], mouth[8])   # 53, 57
    # Horizontal distance
    C = distance.euclidean(mouth[0], mouth[6])   # 49, 55
    mar = (A + B) / (2.0 * C)
    return mar

def get_head_pose(shape, frame_size):
    """Estimate head pose (pitch, yaw, roll) from facial landmarks"""
    # 2D image points from landmarks
    image_points = np.array([
        shape[30],  # Nose tip
        shape[8],   # Chin
        shape[36],  # Left eye left corner
        shape[45],  # Right eye right corner
        shape[48],  # Left mouth corner
        shape[54]   # Right mouth corner
    ], dtype=np.float64)
    
    # Camera matrix (approximate)
    focal_length = frame_size[1]
    center = (frame_size[1] / 2, frame_size[0] / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    
    dist_coeffs = np.zeros((4, 1))
    
    # Solve PnP
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    # Convert rotation vector to rotation matrix and then to Euler angles
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    
    # Get Euler angles
    sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6
    
    if not singular:
        pitch = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        yaw = math.atan2(-rotation_matrix[2, 0], sy)
        roll = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        pitch = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        yaw = math.atan2(-rotation_matrix[2, 0], sy)
        roll = 0
    
    # Convert to degrees
    pitch = math.degrees(pitch)
    yaw = math.degrees(yaw)
    roll = math.degrees(roll)
    
    return pitch, yaw, roll

def get_gaze_direction(eye, shape, side):
    """Detect eye gaze direction (left/right/center)"""
    # Get eye region
    eye_center = np.mean(eye, axis=0).astype(int)
    
    # For left eye (36-41), for right eye (42-47)
    if side == 'left':
        left_corner = shape[36]
        right_corner = shape[39]
    else:
        left_corner = shape[42]
        right_corner = shape[45]
    
    # Calculate horizontal position of eye center relative to corners
    eye_width = distance.euclidean(left_corner, right_corner)
    
    # Get pupil position (approximate using eye center)
    pupil_x = eye_center[0]
    left_x = left_corner[0]
    right_x = right_corner[0]
    
    # Normalize position (0 = left, 0.5 = center, 1 = right)
    if eye_width > 0:
        gaze_ratio = (pupil_x - left_x) / eye_width
    else:
        gaze_ratio = 0.5
    
    return gaze_ratio

def get_head_direction(yaw, pitch):
    """Convert head pose angles to direction string"""
    direction = "FORWARD"
    
    if abs(yaw) > HEAD_TURN_THRESHOLD:
        if yaw > 0:
            direction = "LEFT"
        else:
            direction = "RIGHT"
    elif abs(pitch) > HEAD_TURN_THRESHOLD:
        if pitch > 0:
            direction = "DOWN"
        else:
            direction = "UP"
    
    return direction

def calculate_fatigue_score(drowsy_ratio, yawn_count, blink_rate, attention_ratio):
    """Calculate overall fatigue score (0-100%)"""
    global start_time
    
    elapsed_minutes = max((time.time() - start_time) / 60.0, 0.5)
    
    # Drowsiness component (0-40)
    drowsy_component = min(drowsy_ratio * 100, 100) * (WEIGHT_DROWSINESS / 100)
    
    # Yawn component (0-25) - 3+ yawns per minute is concerning
    yawns_per_min = yawn_count / elapsed_minutes
    yawn_component = min(yawns_per_min / 3.0 * 100, 100) * (WEIGHT_YAWN / 100)
    
    # Blink rate component (0-20) - Normal: 15-20/min, <10 or >30 is abnormal
    if blink_rate < 10:
        blink_component = (10 - blink_rate) / 10 * 100 * (WEIGHT_BLINK / 100)
    elif blink_rate > 30:
        blink_component = (blink_rate - 30) / 20 * 100 * (WEIGHT_BLINK / 100)
    else:
        blink_component = 0
    
    # Attention component (0-15)
    attention_component = min(attention_ratio * 100, 100) * (WEIGHT_ATTENTION / 100)
    
    total_score = drowsy_component + yawn_component + blink_component + attention_component
    return min(total_score, 100)

# ===================== MAIN LOOP =====================

# Start webcam
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FPS, TARGET_FPS)

# Wait for camera to initialize
time.sleep(2)

print("=" * 50)
print("ENHANCED DRIVER MONITORING SYSTEM STARTED")
print(f"Target FPS: {TARGET_FPS}")
print("Press 'q' to quit")
print("=" * 50)

frame_delay = 1.0 / TARGET_FPS  # Delay between frames

while True:
    loop_start = time.time()
    
    ret, frame = video_capture.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_size = frame.shape[:2]
    
    # FPS calculation
    fps_frame_count += 1
    if time.time() - fps_start_time >= 1.0:
        current_fps = fps_frame_count
        fps_frame_count = 0
        fps_start_time = time.time()
    
    # Detect faces
    faces = detector(gray, 0)
    
    # Default values
    head_direction = "NO FACE"
    gaze_direction = "N/A"
    ear = 0.0
    mar = 0.0
    is_drowsy = False
    is_yawning = False
    is_distracted = False
    
    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)
        
        # Draw face landmarks
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)
        
        # ===== EYE ASPECT RATIO =====
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        
        # Draw eye contours
        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)
        
        # Blink detection
        if ear < EYE_ASPECT_RATIO_THRESHOLD:
            if not EYE_CLOSED_PREV:
                EYE_CLOSED_PREV = True
        else:
            if EYE_CLOSED_PREV:
                TOTAL_BLINKS += 1
                EYE_CLOSED_PREV = False
        
        # Drowsiness detection
        if ear < EYE_ASPECT_RATIO_THRESHOLD:
            DROWSY_COUNTER += 1
            if DROWSY_COUNTER >= DROWSY_FRAMES_THRESHOLD:
                is_drowsy = True
        else:
            DROWSY_COUNTER = 0
        
        # ===== MOUTH ASPECT RATIO (YAWN) =====
        mouth = shape[mStart:mEnd]
        mar = mouth_aspect_ratio(mouth)
        
        # Draw mouth contour
        cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (255, 0, 255), 1)
        
        # Yawn detection
        if mar > MOUTH_ASPECT_RATIO_THRESHOLD:
            YAWN_COUNTER += 1
            if YAWN_COUNTER >= YAWN_FRAMES_THRESHOLD:
                is_yawning = True
                if not YAWNING_PREV:
                    TOTAL_YAWNS += 1
                    YAWNING_PREV = True
        else:
            YAWN_COUNTER = 0
            YAWNING_PREV = False
        
        # ===== HEAD POSE =====
        pitch, yaw, roll = get_head_pose(shape, frame_size)
        head_direction = get_head_direction(yaw, pitch)
        
        # Attention check
        if head_direction != "FORWARD":
            ATTENTION_COUNTER += 1
            if ATTENTION_COUNTER >= ATTENTION_FRAMES_THRESHOLD:
                is_distracted = True
        else:
            ATTENTION_COUNTER = 0
        
        # ===== EYE GAZE =====
        left_gaze = get_gaze_direction(leftEye, shape, 'left')
        right_gaze = get_gaze_direction(rightEye, shape, 'right')
        avg_gaze = (left_gaze + right_gaze) / 2
        
        if avg_gaze < (0.5 - GAZE_THRESHOLD):
            gaze_direction = "LEFT"
        elif avg_gaze > (0.5 + GAZE_THRESHOLD):
            gaze_direction = "RIGHT"
        else:
            gaze_direction = "CENTER"
    
    # ===== CALCULATE FATIGUE SCORE =====
    elapsed_time = max(time.time() - start_time, 1)
    elapsed_minutes = elapsed_time / 60.0
    
    drowsy_ratio = DROWSY_COUNTER / max(DROWSY_FRAMES_THRESHOLD, 1)
    attention_ratio = ATTENTION_COUNTER / max(ATTENTION_FRAMES_THRESHOLD, 1)
    blink_rate = TOTAL_BLINKS / max(elapsed_minutes, 0.1)
    
    fatigue_score = calculate_fatigue_score(
        drowsy_ratio, TOTAL_YAWNS, blink_rate, attention_ratio
    )
    
    # ===== ALERTS =====
    alert_triggered = False
    
    if is_drowsy:
        cv2.putText(frame, "!!! DROWSY - WAKE UP !!!", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        alert_triggered = True
    
    if is_distracted:
        cv2.putText(frame, "!!! LOOK AT ROAD !!!", (50, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
        alert_triggered = True
    
    if is_yawning:
        cv2.putText(frame, "YAWNING DETECTED", (50, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
    
    # Sound alarm
    if alert_triggered:
        if not ALARM_ON:
            alert_sound.play(-1)
            ALARM_ON = True
    else:
        if ALARM_ON:
            alert_sound.stop()
            ALARM_ON = False
    
    # ===== DISPLAY INFO =====
    # Fatigue score bar
    bar_width = int(fatigue_score * 2)
    bar_color = (0, 255, 0) if fatigue_score < 40 else (0, 165, 255) if fatigue_score < 70 else (0, 0, 255)
    cv2.rectangle(frame, (10, 10), (10 + bar_width, 30), bar_color, -1)
    cv2.rectangle(frame, (10, 10), (210, 30), (255, 255, 255), 2)
    cv2.putText(frame, f"Fatigue: {fatigue_score:.0f}%", (220, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Status panel (right side)
    panel_x = frame.shape[1] - 200
    cv2.putText(frame, f"FPS: {current_fps}", (panel_x, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"HEAD: {head_direction}", (panel_x, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f"GAZE: {gaze_direction}", (panel_x, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f"EAR: {ear:.2f}", (panel_x, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, f"MAR: {mar:.2f}", (panel_x, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, f"Blinks: {TOTAL_BLINKS}", (panel_x, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, f"Yawns: {TOTAL_YAWNS}", (panel_x, 195),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Show frame
    cv2.imshow('Driver Fatigue Monitor', frame)
    
    # FPS control - wait to maintain target FPS
    elapsed = time.time() - loop_start
    wait_time = max(1, int((frame_delay - elapsed) * 1000))
    
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break

# Cleanup
alert_sound.stop()
video_capture.release()
cv2.destroyAllWindows()
pygame.mixer.quit()

print("\n" + "=" * 50)
print("SESSION SUMMARY")
print(f"Duration: {(time.time() - start_time) / 60:.1f} minutes")
print(f"Total Blinks: {TOTAL_BLINKS}")
print(f"Total Yawns: {TOTAL_YAWNS}")
print(f"Final Fatigue Score: {fatigue_score:.0f}%")
print("=" * 50)

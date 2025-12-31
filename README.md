#  Driver Fatigue Monitoring System (AIoT)

**Team Name:** Hallies
**Team Leader:** Nitheshkummar C
**Track:** AIoT
**Institution:** Amrita Vishwa Vidyapeetham
**Forum:** IETE Student Forum

---

##  Project Overview

Driver fatigue and stress are among the leading causes of road accidents, especially in **public transportation and fleet operations** such as buses and taxis. Existing solutions often rely on **camera-only systems, wearable devices, or cloud-based processing**, which raise concerns about **privacy, cost, latency, and adoption**.

This project presents a **privacy-preserving, edge-based Driver Fatigue Monitoring System** that uses **multi-modal sensing** and **TinyML** to detect driver alertness levels in real time and issue timely alerts—**without storing video or biometric data**.

---

##  Problem Statement

* Driver fatigue and stress significantly increase accident risk
* Camera-only systems fail in low light or occlusion scenarios
* Wearables are uncomfortable and poorly adopted
* Cloud-based solutions introduce latency and privacy risks

**There is a need for a deployable, ethical, and reliable fatigue monitoring system that works in real time at the edge.**

---

##  Proposed Solution

A **multi-modal AIoT system** that:

* Combines **visual**, **physiological**, and **behavioral** cues
* Performs **on-device (edge) inference**
* Classifies driver state as:

  * 🟢 Alert
  * 🟡 Stressed
  * 🔴 Fatigued
* Triggers **progressive visual and audio alerts**

---

##  Key Features

* Real-time fatigue and stress detection
* Camera-based eye closure & head movement analysis
* Steering-wheel–embedded sensors (HRV & grip pressure)
* Multi-modal sensor fusion for reduced false alarms
* Edge-based processing (no cloud dependency)
* Privacy-preserving design (no data storage)
* Visual (LED) and audio (buzzer) alerts

---

##  AI Technologies Used

* **Computer Vision (OpenCV)**

  * Eye closure detection
  * Head movement analysis

* **TinyML**

  * Lightweight fatigue & stress classification
  * Optimized for microcontroller deployment

* **Hybrid Decision Logic**

  * Rule-based + ML fusion for reliability

---

##  Hardware Components

| Component             | Purpose                             |
| --------------------- | ----------------------------------- |
| ESP32 Microcontroller | Sensor interfacing & decision logic |
| Camera Module         | Facial cue capture                  |
| HRV Sensor            | Heart rate variability monitoring   |
| Grip Pressure Sensor  | Behavioral fatigue detection        |
| LEDs                  | Visual alerts                       |
| Buzzer                | Audio alerts                        |

---

##  System Architecture

**Input Sensors → ESP32 Edge Processing → Driver State Classification → Alerts**

**Sensors:**

* Camera (Eye closure, head movement)
* HRV Sensor
* Grip Pressure Sensor

**Processing:**

* TinyML inference
* Sensor fusion
* Decision logic

**Output:**

* LED indicators
* Buzzer alerts

---

##  Process Flow

1. System initialization
2. Continuous camera data capture
3. Steering-wheel sensor data acquisition
4. Sensor data preprocessing
5. Multi-modal sensor fusion
6. Driver state classification
7. Alert triggering
8. Continuous monitoring loop

---

##  Simulation & Mock Frames

* Driver fatigue detection visualization
* ESP32-based system simulation
* Alert triggering based on fatigue severity

(Mock frames are included in the PPT and simulation visuals.)

---

##  What Makes This Different?

* ✅ Multi-modal sensing (not camera-only)
* ✅ No wearable devices
* ✅ Fully edge-based (no cloud)
* ✅ Works even if one sensor fails
* ✅ Privacy-first architecture

---

##  Applications

* Public transport buses
* Taxi & ride-hailing fleets
* Long-haul commercial vehicles
* Occupational driver safety monitoring


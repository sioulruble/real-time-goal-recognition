# Real-Time Goal Recognition

A small research prototype for **online human goal inference**
from RGB-D vision.

The system combines object detection, hand motion,
vision–language cues, and probabilistic temporal inference
to estimate user intent during manipulation.

---

## How it works

RGB-D → Objects & Hand → VLM Hypotheses → HMM → Goal Beliefs

- Object detection runs at a controlled temporal rate  
- Semantic cues suggest goals, never decide them  
- A Hidden Markov Model fuses noisy observations over time  

The main loop only orchestrates.

Two abstract interfaces structure the system:

- RGB-D sensor interface : A common API used to integrate different depth sensors.

- Hand tracking interface : A shared interface for MediaPipe, tracker-based methods, or teleoperation.

Concrete implementations are selected through simple factory functions,
so the system remains independent from any specific sensor or tracker.

---

## Run

```bash
python3 main.py
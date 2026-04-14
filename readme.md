# HMM-based intent state inference

<video src="https://github.com/user-attachments/assets/cbe5f18f-f11e-473c-8bf5-284c170bca3a" controls width="50%"></video>

A small research prototype for **online human goal inference**
from RGB-D vision.

The system combines object detection, hand motion,
vision–language cues, and probabilistic temporal inference
to estimate user intent during manipulation/shared autonomy.

---

## how it works

RGB-D → Objects & Hand → VLM Hypotheses → HMM → Goal Beliefs

- Object detection runs at a controlled temporal rate  
- Semantic cues suggest goals, never decide them  
- A Hidden Markov Model fuses noisy observations over time  

The main loop is minimalistic and only orchestrates data flow.

## structure

```bash
├── main.py
├── requirements.txt
│
├── data/                    # goal taxonomy, goal relationship graph, HMM transition probabilities
│
└── src/rtgr/
    ├── config/              # global settings (thresholds, labels, paths)
    ├── sensors/             # camera stream acquisition
    ├── perception/          # object & hand detection/tracking (YOLO)
    ├── inference/           # goal recognition engine (HMM, rules)
    └── visualization/       # real-time annotation rendering
```

Two abstract interfaces structure the system:

- **RGB-D sensor interface**  
  A common API used to integrate different depth sensors (stereo or RGB-D).

- **Hand tracking interface**  
  A shared interface for MediaPipe, tracker-based methods, or teleoperation inputs.

Start by implementing the abstract interfaces for your RGB-D sensor and hand tracking pipeline, then plug in your object detection and VLM models through the provided factory functions.

---

## run

### 1. Environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Launch 

```bash
python3 main.py
```

The system continuously acquires RGB-D data, extracts object and hand observations, generates semantic hypotheses, and estimates goal belief states through temporal probabilistic inference.

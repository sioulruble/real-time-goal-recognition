OBS_TYPE_TO_ID={
    "closest_object": 1000,
    "moving_closer": 2000,
    "looking_at": 3000,
    "aligned_with": 4000,
    "already_done": 5000,
}

ACTIONS = ["grab(object1)", "push(object1)", "place(object1)", "pull(object1)", "press(object1)"]

TIMING_STATS = {
    "YOLO": 0.0,
    "VLM": 0.0,
    "LLM": 0.0,
    "HMM": 0.0,
}

#hmm constants 
HMM_UPDATING_TIME = 0.1
HMM_MEMORY_LOSS_VALUE = 0.975
HMM_HEURISTIC_RATIO = 0.6
HMM_MIN_PERSISTENCE = 3
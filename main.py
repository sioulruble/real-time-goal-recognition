import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


import cv2
import time
import random
import torch
import numpy as np
from PIL import Image
from datetime import datetime, timedelta
from collections import defaultdict, deque
from multiprocessing import Process, Manager
import threading
import re
import json
import csv

from rtgr.inference.hmm import HMM
from rtgr.inference.VLM import VLMProcessor
from langchain_ollama import OllamaLLM as Ollama
from sentence_transformers import SentenceTransformer 
from rtgr.inference.similaritymodel import SentenceSimilarityModel
import cv2
import time
from collections import defaultdict
from ultralytics import YOLO
import os
import open3d as o3d
from rtgr.inference.hmm_grab import GrabHMM
from rtgr.inference.Hand_position import HandPositionReader
from rtgr.inference.receive_gaze import EyesTracking
from rtgr.perception.object_detection import *
from rtgr.utils import *
from rtgr.config.paths import  YOLO_MODEL, TRANSITION_MATRIX, GOALS_TYPE_CSV, TIME_SPENT_CSV, RELATED_GOALS_JSON
from rtgr.sensors.orbbec.orbbec_depth_sensor import *
from rtgr.perception.hand_tracking.hand_mediapipe import *
import time
from ultralytics import RTDETR

# === Parameters ===
video_path = "recorded_stream.svo"

VLM_model_name = "llava-phi3" 
OBJECT_TYPES = load_object_types(str(GOALS_TYPE_CSV))
model = YOLO(YOLO_MODEL)
classes = list(model.names.values()) if hasattr(model, "names") else []

#mediapipe hand detection
mp_hands = mp.solutions.hands 

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# Timers and thresholds
yolo_timer = datetime.now()
timer = datetime.now()
waiting_timer = datetime.now()
yolo_updating_time = 0.5
updating_time = 0.1
threshold_proba = 0.75
temperature = 0.4
heuristic_ratio = 0.6
memory_loss_value = 0.975
timing_stats = {
    "YOLO": 0.0,
    "VLM": 0.0,
    "LLM": 0.0,
    "HMM": 0.0,
}

# State variables
new_goal_achieved = False
list_of_actions = ["grab(object1)", "push(object1)", "place(object1)", "pull(object1)", "press(object1)"]
current_state = []
object_goals = {}
fps_times = []
last_possible = []
dict_of_objects_at_vlm_time = {}
obs_type_to_id={
    "closest_object": 1000,
    "moving_closer": 2000,
    "looking_at": 3000,
    "aligned_with": 4000,
    "already_done": 5000,
    #"holding": 4000,
    #"grasp_attempt": 5000,
    #"current_state": 6000,
}


tracker = EyesTracking()
eye_thread = threading.Thread(target=tracker.stream_data, daemon=True)
eye_thread.start()

# Initialize VLM processor
processorLL = VLMProcessor()
vlm_result_container = {}
round= 0  #use frequency of 5 to update the VLM
shared_caption = type('', (), {})()
shared_caption.value = "No description yet"
vlm_thread = None
last_vlm_time = datetime.min
vlm_result_container = {}
list_of_goals = []
object_to_id = {}
last_distance = {}
#Initialize similarity sentence
smodel = SentenceTransformer("all-MiniLM-L6-v2")
smModel = SentenceSimilarityModel(smodel)


camera: DepthSensor = OrbbecDepthSensor()
camera.start()
hand: HandTracker = MediaPipeHand()


#iter counter
iter = 0
iter_time = 0
if __name__ == "__main__":


    #Initialize HMM
    image, depth = camera.get_frames()
    while image is None:
        image, depth = camera.get_frames()
    detections = detect_objects(image, model)
    classes = list(model.names.values()) if hasattr(model, "names") else []
    object_to_id = {obj: idx for idx, obj in enumerate(classes)}
    goals_beliefs = {goal: 1 / len(list_of_goals) for goal in list_of_goals}
    loaded_matrix= load_transition_matrix(TRANSITION_MATRIX)
    transition_proba =create_transition_matrix(goals_beliefs, list_of_goals, loaded_matrix)
    current_goals_landmarks = build_current_goals_landmarks(goals_beliefs, obs_type_to_id, object_to_id)
    mapping_info = mapping_infos(obs_type_to_id, object_to_id)
    decreasing_actions = [] # actions that decrease the probability of the goal
    hmm = HMM(goals_beliefs, transition_proba, current_goals_landmarks, decreasing_actions)
    landmark_uniqueness = hmm.get_landmarks_uniqueness() #uniqueness computation (proba d'observer certaines actions selon chaque objectif)
    print(landmark_uniqueness)
    hmm.compute_likelihood_table(heuristic_ratio, landmark_uniqueness)
    hand._3Dpos = np.array([np.inf, np.inf, np.inf])
    dict_3d_positions = estimate_objects_3d_positions(detections, depth, classes)

    list_of_observations = []
    last_distance= save_last_distance(dict_3d_positions, hand._3Dpos)



    while True:
        current_time = datetime.now()
        yolo_elapsed_time = current_time - yolo_timer
        elapsed_time = current_time - timer
        image, depth = camera.get_frames()
        results = hands.process( cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        hand.update_position(image, depth)
        

        start_time = time.time()
        gaze_finalvalue="Unknown"

        if yolo_elapsed_time.total_seconds() > yolo_updating_time:
            yolo_timer = datetime.now()

            detections = detect_objects(image, model)
            gaze_finalvalue="Unknown"
            dict_3d_positions = estimate_objects_3d_positions(detections, depth, classes)

            resYolo = [model.names[detection["class_id"]] for detection in detections]
            list_of_IDs = [detection["class_id"] for detection in detections]
            new_dict_of_objects = rename_objects(detections, model)
            possible_actions = build_all_possible_goals(list_of_actions, list(new_dict_of_objects.keys()))
            dict_of_objects = rename_objects(detections, model)

             #VLM Preptreatement /thread
            if (datetime.now() - last_vlm_time).total_seconds() > 3:
                print   ("⏳ Starting VLM processing thread...")
                last_vlm_time = datetime.now()
                try:
                    vlmframe = processorLL.convert_image_for_VLM(image) #change resolution to 640x640
                    vlm_result_container = {}
                    dict_of_objects_at_vlm_time = dict(new_dict_of_objects)
                    grouped = group_by_type(dict_of_objects, OBJECT_TYPES)
                    print("Dict of object given to the VLM:",dict_of_objects_at_vlm_time)
                    for typ, objs in grouped.items():
                    	print(f"{typ}: {', '.join(objs)}")
                    vlm_thread = threading.Thread(
                        target=threaded_VLM_wrapper,
                        args=(processorLL, VLM_model_name, shared_caption, vlmframe, dict_of_objects_at_vlm_time, timing_stats, vlm_result_container, list_of_actions)
                    )
                    vlm_thread.start()
                except Exception as e:
                    print(f"Erreur pendant le traitement VLM : {e}")

            if vlm_thread is not None and not vlm_thread.is_alive():
                print("✅ VLM thread finished, retrieving results...")
                result = vlm_result_container.get("result", None)
                if result:
                    if dict_of_objects:
                        list_of_goals = process_multiline_caption(result, smModel, list_of_actions, list(dict_of_objects_at_vlm_time.keys()))

                    else:
                        print("No object detected — skipping similarity_model call.")
                        list_of_goals = []

                    goal_candidate= goals_candidate(dict_3d_positions, hand._3Dpos, list_of_goals, TIME_SPENT_CSV)
                    if goal_candidate:
                        list_of_goals.extend(goal_candidate)

                    if "Undecided" not in list_of_goals:
                        list_of_goals.append("Undecided")

                    object_to_action_ids = defaultdict(list)
                    for goal, ids in current_goals_landmarks.items():
                        if '(' in goal:
                            obj = goal.split('(')[-1].strip(') ')
                            object_to_action_ids[obj].extend(ids)

                    closest_object, diff_distance, new_observation= moving_closer(dict_3d_positions, hand._3Dpos, last_distance)
                    closest_object_observations = generate_observations_live(dict_3d_positions, hand._3Dpos)
                    all_observations=[]
                    all_observations= closest_object_observations + new_observation
                    list_of_observations = match_obs_with_landmarks_id(all_observations, mapping_info)
                    list_of_observations = [obs for obs in list_of_observations if obs != 99]
                    if set(list_of_goals) != set(hmm.goal_beliefs.keys()):
                        hmm.goal_beliefs = {goal: 1 / len(list_of_goals) for goal in list_of_goals}
                        hmm.transition_proba = create_transition_matrix(hmm.goal_beliefs, list_of_goals, loaded_matrix)
                        hmm.current_goals_landmarks = build_current_goals_landmarks(hmm.goal_beliefs, obs_type_to_id, object_to_id)
                        uniqueness = hmm.get_landmarks_uniqueness()
                        hmm.compute_likelihood_table(heuristic_ratio, uniqueness)

                vlm_thread = None

            if len(dict_of_objects) == 0:
                    print("⚠️ No object detected — skipping similarity_model call.")
                    continue
            object_to_action_ids = defaultdict(list)
            for goal, ids in current_goals_landmarks.items():
                if '(' in goal:
                    obj = goal.split('(')[-1].strip(') ')
                    object_to_action_ids[obj].extend(ids)
            closest_object, diff_distance, new_observation = moving_closer(dict_3d_positions, hand._3Dpos, last_distance)
            print("new observation", new_observation)

            closest_object_observations = generate_observations_live(dict_3d_positions, hand._3Dpos)
            all_observations=[]
            all_observations= closest_object_observations + new_observation
            list_of_observations = match_obs_with_landmarks_id(all_observations, mapping_info)
            list_of_observations = [obs for obs in list_of_observations if obs != 99]
            last_distance= save_last_distance(dict_3d_positions, hand._3Dpos)
            goal_candidate= goals_candidate(dict_3d_positions, hand._3Dpos, list_of_goals, TIME_SPENT_CSV)
            if goal_candidate:
                list_of_goals.extend(goal_candidate)
            if "Undecided" not in list_of_goals:
                list_of_goals.append("Undecided")

            # print("-------")


        print("\n" + "="*50)
        print("🎯 TOP 5 GOAL PROBABILITIES")
        print("="*50)
        sorted_goals = sorted(hmm.goal_beliefs.items(), key=lambda x: x[1], reverse=True)
        for i, (goal, probability) in enumerate(sorted_goals[:5], 1):
            bar_length = int(probability * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            print(f"{i}. {goal:30s} │{bar}│ {probability:.2%}")



        if set(list_of_goals) != set(hmm.goal_beliefs.keys()):
            hmm.goal_beliefs = {goal: 1 / len(list_of_goals) for goal in list_of_goals}
            hmm.transition_proba = create_transition_matrix(hmm.goal_beliefs, list_of_goals, loaded_matrix)
            hmm.current_goals_landmarks = build_current_goals_landmarks(hmm.goal_beliefs, obs_type_to_id, object_to_id)
            uniqueness = hmm.get_landmarks_uniqueness()
            hmm.compute_likelihood_table(heuristic_ratio, uniqueness)

        if list_of_observations:
            filtred_observations= filter_observations(list_of_observations, list_of_goals, mapping_info)
            alpha, current_goal = hmm.assisted_teleop(updating_time, memory_loss_value, filtred_observations)

        object_goals = online_goal_estimation(image,  hmm.goal_beliefs, dict_of_objects, gaze_finalvalue, closest_object_observations)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

camera.stop()

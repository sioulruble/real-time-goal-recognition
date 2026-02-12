# === Imports ===

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

# === External Modules ===
from rtgr.inference.hmm import HMM
from rtgr.inference.VLM import VLMProcessor
from langchain_ollama import OllamaLLM as Ollama
from sentence_transformers import SentenceTransformer 
from rtgr.inference.similaritymodel import SentenceSimilarityModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline

import cv2
import time
from collections import defaultdict
from ultralytics import YOLO
import os
import open3d as o3d
from rtgr.inference.hmm_grab import GrabHMM
from rtgr.inference.Hand_position import HandPositionReader
from rtgr.inference.receive_gaze import EyesTracking
from rtgr.visualization.overlays import *
from rtgr.utils import *


def load_object_types(csv_path="goals_type.csv"):
    mapping = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            obj = row["object"].strip()
            typ = row["type"].strip()
            mapping[obj].append(typ)
    return mapping


def group_by_type(objects_dict, OBJECT_TYPES):
    grouped = defaultdict(list)
    for name in objects_dict.keys():
        if name in OBJECT_TYPES:
            for typ in OBJECT_TYPES[name]:
                if name not in grouped[typ]:
                    grouped[typ].append(name)
    return grouped
def find_gaze_value(camera):
    gaze_position=camera.eye_tracker.gaze_position
    return gaze_position

def convert_gaze(frame,gaze_position):
    if frame is None or not isinstance(frame, np.ndarray):
        return None
    h,w = frame.shape[:2]
    u,v= float(gaze_position[0]), float(gaze_position[1])
    x = int(u*w)
    y = int((1-v)*h)
    return x,y

def normalize_goal(goal):
    match = re.match(r"(\w+)\((\w+?)(\d*)\)", goal)
    if match:
        action, obj_base, _ = match.groups()
        return f"{action}({obj_base})"
    return goal


def rename_objects(detections, model):
    object_count = {}
    renamed_objects = {}

    for obj in detections:
        class_id = obj["class_id"]
        if class_id in object_count:
            object_count[class_id] += 1
            renamed_objects[f"{model.names[class_id]}{object_count[class_id]}"] = obj["box"]
        else:
            object_count[class_id] = 1
            renamed_objects[model.names[class_id]] = obj["box"]

    return renamed_objects

def build_all_possible_goals(list_of_actions, list_of_objects):
    all_possible_goals = []
    for action in list_of_actions:
        for obj in list_of_objects:
            goal = action.replace("object1", obj)
            all_possible_goals.append(goal)
    return all_possible_goals



def threaded_VLM_wrapper(processorLL, model_name, caption, frame, objects, timing, result_container, list_of_actions):
    result = processorLL.VLM_process_func(model_name, caption, frame, objects, list_of_actions, timing)
    print("VLM output:", result)
    result_container["result"] = result
    return result_container


def build_current_goals_landmarks(goals_beliefs, obs_type_to_id, object_to_id):
        mapping = {}

        for goal in goals_beliefs:
            this_goal = []
            found = False

            for obj, obj_id in object_to_id.items():
                if obj in goal:
                    for obs_type, obs_type_id in obs_type_to_id.items():
                        landmark_id = obs_type_id + obj_id
                        this_goal.append(landmark_id)


                    mapping[goal] = this_goal
                    found = True
                    break

            if not found:
                for obs_type, obs_type_id in obs_type_to_id.items():
                    landmark_id = obs_type_id + 999  # 999 = placeholder ID for unknown objects
                    this_goal.append(landmark_id)

                mapping[goal] = this_goal
        return mapping

def mapping_infos(obs_type_to_id, object_to_id):
    landmark_info = {}  # Stores landmark_id → {"type": ..., "object": ..., "id": ...}
    for obj, obj_id in object_to_id.items():
        for obs_type, obs_type_id in obs_type_to_id.items():
            landmark_id = obs_type_id + obj_id
            if landmark_id not in landmark_info:
                landmark_info[landmark_id] = {
                    "type": obs_type,
                    "object": obj,
                    "id": landmark_id
                }
    return landmark_info

def moving_closer(dict_3d_positions, hand_position_3d, last_distance):
    print("last distance ", last_distance)
    diff_distance=dict_3d_positions.copy()
    closest_object= []
    new_observation = []
    for name, pos in diff_distance.items():
        print(name)
        if name in last_distance and name != 'person':
            distance = np.linalg.norm(pos -hand_position_3d)
            diff = last_distance[name] - distance
            print ("HAND POS",hand_position_3d )
            print("POS", pos)
            print ("DIFF",diff )
            print("DISTANCE", distance)

            if diff>0 : 
                print("a")
                new_obs= {}
                new_obs['type']= "moving_closer"
                new_obs['object']= name
                new_observation.append(new_obs)
                closest_object.append(name)
            diff_distance[name]= diff
        else:
            print(f"New object detected:", name)

    for name in list(last_distance.keys()):
        if name not in diff_distance:
            del last_distance[name]

    return closest_object, diff_distance, new_observation

def save_last_distance(dict_3d_positions, hand_position_3d):
    last_distance=dict_3d_positions.copy()
    for name, pos in last_distance.items():
        # if "person" in name.lower():
        #     last_distance[name] = float('inf')
        # else:
        
        distance = np.linalg.norm(pos - hand_position_3d)
        last_distance[name] = distance
    return last_distance

def ID_to_text(ID, mapping_info):
    for index, landmark_text in mapping_info.items():
        if index==ID:
            return landmark_text

def generate_observations_live(dict_3d_positions, hand_position_3d, threshold=0.2):
    observations = []
    test_observations = {}
    save_observations = {}
    # Step 1: Find nearby objects
    for name, pos in dict_3d_positions.items():
        # if name != 'person':
        distance = np.linalg.norm(pos - hand_position_3d)
        if distance < threshold:
            save_observations[name] = distance

    # Step 2: Sort objects by distance
    sorted_objects = sorted(save_observations.items(), key=lambda item: item[1])
    print("Sorted nearby objects:", sorted_objects)

    if not sorted_objects:
        print("No objects detected within the threshold.")
        return observations

    closest_object = sorted_objects[0][0]  # name
    observation_type = "closest_object"

    test_observations['type']= observation_type
    test_observations['object']=closest_object
    observations.append(test_observations)

    return observations

def match_obs_with_landmarks_id(observations, mapping_info):
    observations_ID=[]
    for obs in observations:
        observation_type= obs['type']
        type_object= obs['object']
        for lid, info in mapping_info.items():
            if info["type"] == observation_type and info["object"] == type_object:
                observations_ID.append(info["id"])
    return observations_ID

def goals_candidate(dict_3d_positions, hand_position_3d, list_of_goals, path):                    # Add the closest object as goal if it's not already in the list
    if not dict_3d_positions:
        print("Can't have the 3D pos so no possible candidate")
        return None

    if dict_3d_positions:
        distances = {
            name: np.linalg.norm(pos - hand_position_3d)
            for name, pos in dict_3d_positions.items()
            }
        sorted_distances = sorted(distances.items(), key=lambda item: item[1])
        closest_object = sorted_distances[0][0] if sorted_distances else None

        if closest_object:
            object_already_in_goals = any(
            f"({closest_object})" in goal for goal in list_of_goals
        )
        if not object_already_in_goals:
            if path:
                matching_goals = []
                with open(path, mode='r', newline='') as csv_file:
                    reader =csv.DictReader(csv_file)
                    for row in reader:
                        if f"{(closest_object)}" in row ['Goal']:
                            matching_goals.append(row['Goal'])
                print(f"Adding goal from file (unique match): {matching_goals}")
                return matching_goals

            else:
                goal_candidate = f"grab({closest_object})"
                print(f"Adding goal based on proximity: {goal_candidate}")
                return goal_candidate

    return None

def similarities_between_boxes(box1, box2, distance_threshold=150, size_threshold=0.7):

    x1, y1 = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
    x2, y2 = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2
    euclideanDistance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    widthRatio = 0 if max(box1[2] - box1[0], box2[2] - box2[0]) == 0 else min(box1[2] - box1[0], box2[2] - box2[0]) / max(box1[2] - box1[0], box2[2] - box2[0])
    heightRatio = 0 if max(box1[3] - box1[1], box2[3] - box2[1]) == 0 else min(box1[3] - box1[1], box2[3] - box2[1]) / max(box1[3] - box1[1], box2[3] - box2[1])

    return euclideanDistance < distance_threshold and widthRatio > size_threshold and heightRatio > size_threshold

def display_goal_estimation(frame, goals_beliefs, object_boxes, previous_object_goals, shared_caption, timing_stats, fps_times, camera, detections,gaze_finalvalue):
    # Initialize the dictionary if it's the first call
    if previous_object_goals is None:
        previous_object_goals = {}

    # Dictionary to store the most probable goal and its confidence for each object
    object_goals = defaultdict(lambda: [None, -1])


    # Step 1: Match each object to the most likely goal
    for goal, probability in goals_beliefs.items():
        object_names = []
        if '(' in goal and ';' not in goal:
            object_names = [goal.split('(')[-1].strip(') ')]
        elif ';' in goal:
            inner = goal.split('(')[-1].strip(')')
            object_names = [name.strip() for name in inner.split(';')]

        for object_name in object_names:
            if object_name:
                current_prob = object_goals[object_name][1]
                if probability > current_prob:
                    object_goals[object_name] = [goal, probability]

    # Step 2: Draw bounding boxes and goal
    draw_goal_boxes(frame, object_boxes, object_goals)
    draw_gaze(frame, gaze_finalvalue)


    # Step 3: Compute FPS
    fps_times.append(time.time())
    if len(fps_times) >= 2:
        fps = len(fps_times) / (fps_times[-1] - fps_times[0])
    else:
        fps = 0.0

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Step 4: Display timing stats
    y_offset = 100
    for key, val in timing_stats.items():
        cv2.putText(frame, f"{key}: {val:.2f}s", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_offset += 20

    # Step 5: Show current scene caption from VLM
    cv2.putText(frame, shared_caption.value[:60], (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Step 6: Display the annotated frame
    cv2.imshow("Goal Recognition", frame)

    return object_goals

def online_goal_estimation(image, goals_beliefs, list_of_bounding_boxes, gaze_finalvalue, closest_object_observations=None):
    frame = image 
    if frame is None:
        return {}

    object_goals = defaultdict(lambda: [None, -1])

    for goal, probability in goals_beliefs.items():
        if '(' in goal:
            objects = goal.split('(')[1].strip(')').split(';')
            for obj in objects:
                if probability > object_goals[obj][1]:
                    object_goals[obj] = [goal, probability]

    draw_goal_boxes(frame, list_of_bounding_boxes, object_goals)
    if closest_object_observations is not None:
        cv2.putText(frame, f"Closest object: {', '.join(str(closest_object_observations))}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    if gaze_finalvalue is not None:
        draw_gaze(frame, gaze_finalvalue)

    cv2.imshow("Goal Recognition", frame)
    return object_goals

def process_multiline_caption(caption: str, similarity_model: SentenceSimilarityModel, list_of_actions, list_of_objects, top_k=1):

    # Split caption into non-empty lines
    lines = extract_clean_lines(caption)

    selected_goals = []
    all_possible = []
    # Generate all possible actions from current objects
    all_possible = similarity_model.all_possible_actions(list_of_actions, list_of_objects)

    for line in lines:

        # Compute similarity for current line
        paired_sorted = similarity_model.get_paired_sorted(line, all_possible)

        # Select top-k most similar actions
        top_actions = [action for action, score in paired_sorted[:top_k]]

        # Add to global result list
        selected_goals.extend(top_actions)

    return selected_goals

def extract_clean_lines(caption: str):
    return [line.strip().replace("'", "").replace("[", "").replace("]", "") for line in caption.split(',')]

def create_transition_matrix(goals_beliefs, list_of_goals, hmm_transition_matrix):
    transition_matrix = {}
    all_goals = list(hmm_transition_matrix.keys())  # already normalized
    n = len(list_of_goals)

    for goal_from in list_of_goals:
        normalized_from = normalize_goal(goal_from)

        if normalized_from in hmm_transition_matrix:
            full_row = hmm_transition_matrix[normalized_from]
            try:
                # Normalize only for indexing in hmm matrix, keep original goal_to
                row = [full_row[all_goals.index(normalize_goal(goal_to))] for goal_to in list_of_goals]
            except ValueError as e:
                raise ValueError(f"Goal in list_of_goals not found in hmm matrix: {e}")
            total = sum(row)
            row = [val / total for val in row] if total > 0 else [1 / n] * n
        else:
            # If normalized version is not in matrix
            print(f"[WARN] Goal '{goal_from}' (normalized: '{normalized_from}') not in hmm matrix. Uniform distribution applied.")
            row = [1 / n] * n

        transition_matrix[goal_from] = row

    return transition_matrix


def create_z_position_list(dict_3d_positions):

    z_positions = {}
    for (name,pos) in dict_3d_positions.items():
        if pos is not None and len(pos) == 3:
            z_positions[name]= pos[2]  # Extract the Z coordinate
    return z_positions

def load_transition_matrix(path="inference/transition_proba_for_hmm.json"):
    with open(path, "r", encoding="utf-8") as f:
        transition_matrix = json.load(f)
    #print(f"[INFO] Transition matrix loaded from {path}")
    return transition_matrix

def filter_observations(list_of_observations, list_of_goals, mapping_info):
    filtred_observations = []
    for obs in list_of_observations:

        obs_text= ID_to_text(obs, mapping_info)
        for obj in list_of_goals:
            if f"({obs_text['object']})" in obj:
                id_act=(obs)
                build_obs = (id_act, obj)
                filtred_observations.append(build_obs)
    return filtred_observations
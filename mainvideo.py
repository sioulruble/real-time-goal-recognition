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
from orbbec.orbbec_api import OrbbecPlugin as pluginOrbbec
from orbbec.orbbec_api import PointCloudLike
from inference.HMM import HMM
from inference.VLM import VLMProcessor
from langchain_ollama import OllamaLLM as Ollama
from sentence_transformers import SentenceTransformer 
from inference.similaritymodel import SentenceSimilarityModel
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
from inference.HMM_grab import GrabHMM
from inference.Hand_position import HandPositionReader
from inference.receive_gaze import EyesTracking

from utils import *


# === Parameters ===
#video_path = "demo1.svo2"
video_path = "recorded_stream.svo"
yolo_model_path = "yolov8n-seg.pt"
#yolo_model_path = "yolov8n-seg.pt"
# yolo_model_path = "yolov11n-seg.pt"
VLM_model_name = "llava-phi3" 
OBJECT_TYPES = load_object_types("goals_type.csv")

if not os.path.exists(yolo_model_path):
    print("Download YOLOv8n-seg...")
    yolo_model= YOLO(yolo_model_path)
    detectable_classes = list(yolo_model.names.values())

    print("Detectable classes :", detectable_classes)
    print("Model downloaded")

# Timers and thresholds
yolo_timer = datetime.now()
timer = datetime.now()
waiting_timer = datetime.now()
yolo_updating_time = 0.50
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
#hand_position_3d = np.array([-80, 150.0, 470])
#hand_reader = HandPositionReader()
#right_hand = hand_reader.get_right_position()
#print("right hand before", right_hand)
#hand_position_3d = np.array(right_hand)
#print("right hand after", hand_position_3d)

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
# Simple loop pour afficher le flux Orbbec (RGB + overlay gaze si disponible)

camera = pluginOrbbec(yolo_model_path)

# Lancer le tracker des yeux en arrière-plan (optionnel mais utile pour afficher le regard)
tracker = EyesTracking()
eye_thread = threading.Thread(target=tracker.stream_data, daemon=True)
eye_thread.start()
camera.eye_tracker = tracker

# try:
#     while True:
#         image = camera.get_rgb_frame()
#         if image is None:
#             time.sleep(0.01)
#             continue

#         # Si une fonction utilitaire existe pour récupérer la gaze, on l'utilise
#         try:
#             gaze_pos = find_gaze_value(camera)  # retourne typiquement une position ou None
#         except Exception:
#             gaze_pos = None

#         # Affichage du regard sur l'image si disponible
#         if gaze_pos is not None:
#             try:
#                 gaze_xy = convert_gaze(image, gaze_pos)  # convert_gaze doit renvoyer (x,y) ou une string
#             except Exception:
#                 gaze_xy = gaze_pos
#             if isinstance(gaze_xy, (list, tuple)) and len(gaze_xy) >= 2:
#                 cv2.circle(image, (int(gaze_xy[0]), int(gaze_xy[1])), 8, (0, 0, 255), -1)
#             else:
#                 cv2.putText(image, str(gaze_xy), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#         # Essayer d'afficher également la profondeur si disponible
#         if hasattr(camera, "get_depth_frame"):
#             try:
#                 depth = camera.get_depth_frame()
#                 if depth is not None:
#                     depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
#                     depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
#                     combined = cv2.hconcat([cv2.resize(image, (depth_vis.shape[1], depth_vis.shape[0])), depth_vis])
#                     cv2.imshow("Orbbec RGB | Depth", combined)
#                 else:
#                     cv2.imshow("Orbbec RGB", image)
#             except Exception:
#                 cv2.imshow("Orbbec RGB", image)
#         else:
#             cv2.imshow("Orbbec RGB", image)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
# finally:
#     cv2.destroyAllWindows()

# === Main Code ===
if __name__ == "__main__":

    hand_position_3d = np.array([-80, 150.0, 470])

    left_hand_position_3d = np.array([-80, 150.0, 470])




    background = "camera"
    if background == "camera":
        tracker = EyesTracking()
        eye_thread = threading.Thread(target=tracker.stream_data)
        eye_thread.start()
        #if tracker.gaze_position == [0,0,0]:
        	#print("Eyes not detected")
        #else:
            	#print("Eyes tracking data in real time:", tracker.gaze_position)
        camera.eye_tracker = tracker
        print("CAMERA EYES TRACKING TEST:", camera.eye_tracker)
        image = camera.get_rgb_frame()
        if image is None:
            while image is None:
                image = camera.get_rgb_frame()
                time.sleep(0.01)
                continue

        detections, _ = online_objet_recognition(camera)
        classes = camera.classes
        #print("Detectable classes :", classes)
        object_to_id = {obj: idx for idx, obj in enumerate(classes)}
        #print("object_to_id:", object_to_id)

        # Without ZED: create a lightweight holder object.
        # pluginOrbbec.retrieve_measure(...) will attach the methods
        # get_width(), get_height(), get_value(u,v) and get_data() to this object.

        point_cloud = camera.get_3Dpoints()

        if point_cloud is None:
            while point_cloud is None:
                point_cloud = camera.get_3Dpoints()
                print("Waiting for valid point cloud from camera...")
                time.sleep(0.01)
                continue
        dict_3d_positions = {}
        for det in detections:
            name = camera.model.names[det["class_id"]]
            mask = det.get("mask")
            if mask is not None:
                resized_mask = cv2.resize(mask, (point_cloud.shape[1], point_cloud.shape[0]), interpolation=cv2.INTER_NEAREST)
                #visualize_object_from_mask(resized_mask, point_cloud, name)
                pos = compute_3d_position_from_mask(resized_mask, point_cloud)
                print(f"pos: {pos}")
                if pos is not None:
                    print(f"3D position of {name}: {pos}")
                    dict_3d_positions[name] = pos
        resYolo = [camera.model.names[detection["class_id"]] for detection in detections]
    else:
        print("Invalid input. Please enter 'video' or 'camera'.")
        exit()

    # Example: first detection frame (adapt as needed)
    current_time = datetime.now()
    start_yolo = time.time()


    list_of_IDs = [detection["class_id"] for detection in detections]
    dict_of_objects = rename_objects(detections, camera)
    print("Dict of objects:", dict_of_objects)

    possible_actions = []
    possible_actions = build_all_possible_goals(list_of_actions, list(dict_of_objects.keys()))

    # Initialize VLM processor
    processorLL = VLMProcessor()
    vlm_result_container = {}
    round= 0  #use frequency of 5 to update the VLM
    shared_caption = type('', (), {})()
    shared_caption.value = "No description yet"
    vlm_thread = None
    last_vlm_time = datetime.min
    vlm_result_container = {}
    image = camera.get_rgb_frame()
    if image is None:
        while image is None:
            print("Waiting for valid image from camera...")
            image = camera.get_rgb_frame()
            time.sleep(0.01)
            continue
    vlmframe = processorLL.convert_image_for_VLM(image)  # resize to 224x224
    grouped = group_by_type(dict_of_objects, OBJECT_TYPES)
    print("Grouped:", grouped)
    for typ, objs in grouped.items():
        print(f"{typ}: {', '.join(objs)}")
    if "container" in grouped and "contenable" in grouped:
    	list_of_actions = ["grab(object1)", "push(object1)", "place(object1)", "pull(object1)", "press(object1)","pour(object1)"]
    	pour_object = grouped["contenable"]
    	receiver_object = grouped["container"]
    else:
    	list_of_actions = ["grab(object1)", "push(object1)","place(object1)", "pull(object1)", "press(object1)"]
    print("LIST OF ACTIONS", list_of_actions)
    vlm_thread = threading.Thread(
        target=threaded_VLM_wrapper,
        args=(processorLL, VLM_model_name, shared_caption, vlmframe, dict_of_objects, timing_stats, vlm_result_container, list_of_actions)
    )
    vlm_thread.start()
    # wait vlm_thread to finish
    vlm_thread.join()
    result = vlm_result_container.get("result", None)
    if result:
        current_state = result if isinstance(result, list) else [result]
        print(f"✅ current_state initial mis à jour par VLM : {current_state}")

    #Initialize similarity sentence
    smodel = SentenceTransformer("all-MiniLM-L6-v2")
    smModel = SentenceSimilarityModel(smodel)

    all_possible = []
    if dict_of_objects:
        list_of_goals = process_multiline_caption(result, smModel, list_of_actions, list(dict_of_objects.keys()))
    else:
        print("⚠️ No object detected — skipping similarity_model call.")
        list_of_goals = []

    #Add the closest object as goal if it's not already in the list
    goal_candidate=goals_candidate(dict_3d_positions, hand_position_3d, list_of_goals)

    if goal_candidate:
        list_of_goals.extend(goal_candidate)

    # Add "Undecided" to the list of goals
    if "Undecided" not in list_of_goals:
        list_of_goals.append("Undecided")


    #Initialize HMM
    goals_beliefs = {goal: 1 / len(list_of_goals) for goal in list_of_goals}
    loaded_matrix= load_transition_matrix(path="inference/transition_proba_for_hmm.json")
    transition_proba =create_transition_matrix(goals_beliefs, list_of_goals, loaded_matrix)
    current_goals_landmarks = build_current_goals_landmarks(goals_beliefs, obs_type_to_id, object_to_id)
    mapping_info = mapping_infos(obs_type_to_id, object_to_id)
    decreasing_actions = [] # actions that decrease the probability of the goal
    hmm = HMM(goals_beliefs, transition_proba, current_goals_landmarks, decreasing_actions)
    landmark_uniqueness = hmm.get_landmarks_uniqueness() #uniqueness computation (proba d'observer certaines actions selon chaque objectif)
    hmm.compute_likelihood_table(heuristic_ratio, landmark_uniqueness)

###left
    hmm_left = HMM(goals_beliefs, transition_proba, current_goals_landmarks, decreasing_actions)
    landmark_uniqueness = hmm_left.get_landmarks_uniqueness() #uniqueness computation (proba d'observer certaines actions selon chaque objectif)
    hmm_left.compute_likelihood_table(heuristic_ratio, landmark_uniqueness)


    object_to_action_ids = defaultdict(list)
    for goal, ids in current_goals_landmarks.items():
        if '(' in goal:
            obj = goal.split('(')[-1].strip(') ')
            object_to_action_ids[obj].extend(ids)

    closest_object_observations = generate_observations_live(dict_3d_positions, hand_position_3d, object_to_action_ids, mapping_info)
    last_distance= save_last_distance(dict_3d_positions, hand_position_3d)
    obj_moving_closer = []
    list_of_observations = match_obs_with_landmarks_id(closest_object_observations, mapping_info)
    list_of_observations = [obs for obs in list_of_observations if obs != 99]
    filtred_observations = filter_observations(list_of_observations, list_of_goals, mapping_info)
    alpha, current_goal = hmm.assisted_teleop(updating_time, memory_loss_value, filtred_observations)
    closest_object_observations_left = generate_observations_live(dict_3d_positions, left_hand_position_3d, object_to_action_ids, mapping_info)
    last_distance_left= save_last_distance(dict_3d_positions, left_hand_position_3d)
    obj_moving_closer_left = []
    list_of_observations_left = match_obs_with_landmarks_id(closest_object_observations_left, mapping_info)
    list_of_observations_left = [obs for obs in list_of_observations_left if obs != 99]
    filtred_observations_left = filter_observations(list_of_observations_left, list_of_goals, mapping_info)
    alpha_left, current_goal_left = hmm_left.assisted_teleop(updating_time, memory_loss_value, filtred_observations_left)

   
    nbr_of_frames = 0
    every_frame = 1


    while True:
        current_time = datetime.now()
        yolo_elapsed_time = current_time - yolo_timer
        elapsed_time = current_time - timer
        image = camera.get_rgb_frame()
        if image is None:
            time.sleep(0.01)
            continue
        start_time = time.time()
        gaze_position = find_gaze_value(camera)
        print(type(gaze_position))
        if gaze_position is not None:
        	gaze_finalvalue= convert_gaze(image,gaze_position)
        else:
        	gaze_finalvalue="Unknown"

        if yolo_elapsed_time.total_seconds() > yolo_updating_time:
            yolo_timer = datetime.now()

            

            detections, image = online_objet_recognition(camera)
            # Without ZED: create a lightweight holder object.
            # pluginOrbbec.retrieve_measure(...) will attach the methods
            # get_width(), get_height(), get_value(u,v) and get_data() to this object.
            point_cloud = camera.get_3Dpoints()

            if point_cloud is None:
                while point_cloud is None:
                    point_cloud = camera.get_3Dpoints()
                    print("Waiting for valid point cloud from camera...")
                    time.sleep(0.01)
                    continue
            gaze_position = find_gaze_value(camera)
            if gaze_position is not None:
                gaze_finalvalue= convert_gaze(image,gaze_position)
            else:
                gaze_finalvalue="Unknown"
            #print(f"point @ = {point_cloud} — type: {type(point_cloud)}")
            dict_3d_positions = {}
            for det in detections:
                name = camera.model.names[det["class_id"]]
                mask = det.get("mask")
                if mask is not None:
                    resized_mask = cv2.resize(mask, (point_cloud.shape[1], point_cloud.shape[0]), interpolation=cv2.INTER_NEAREST)
                    pos = compute_3d_position_from_mask(resized_mask, point_cloud)
                    print(f"pos: {pos}")
                    if pos is not None:
                        dict_3d_positions[name] = pos
            resYolo = [camera.model.names[detection["class_id"]] for detection in detections]
            list_of_IDs = [detection["class_id"] for detection in detections]
            new_dict_of_objects = rename_objects(detections, camera)
            possible_actions = build_all_possible_goals(list_of_actions, list(new_dict_of_objects.keys()))

            dict_of_objects = rename_objects(detections, camera)

             #VLM Preptreatement /thread
            if (datetime.now() - last_vlm_time).total_seconds() > 3 and image is not None:
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
                    #dict_of_objects = dict(new_dict_of_objects) # update the dictionary of objects
                    vlm_thread.start()
                except Exception as e:
                    print(f"Erreur pendant le traitement VLM : {e}")

            if vlm_thread is not None and not vlm_thread.is_alive():
                print("✅ VLM thread finished, retrieving results...")
                result = vlm_result_container.get("result", None)
                if result:
                    current_state = result if isinstance(result, list) else [result]
                    print(f"current_state mis à jour par VLM : {current_state}")

                    #all_possible = []
                    if dict_of_objects:
                        list_of_goals = process_multiline_caption(result, smModel, list_of_actions, list(dict_of_objects_at_vlm_time.keys()))
                    else:
                        print("No object detected — skipping similarity_model call.")
                        list_of_goals = []

                    goal_candidate= goals_candidate(dict_3d_positions, hand_position_3d, list_of_goals)
                    if goal_candidate:
                        list_of_goals.extend(goal_candidate)

                    if "Undecided" not in list_of_goals:
                        list_of_goals.append("Undecided")


                    if set(list_of_goals) != set(hmm.goal_beliefs.keys()):
                        hmm.goal_beliefs = {goal: 1 / len(list_of_goals) for goal in list_of_goals}
                        hmm.transition_proba = create_transition_matrix(hmm.goal_beliefs, list_of_goals, loaded_matrix)
                        hmm.current_goals_landmarks = build_current_goals_landmarks(hmm.goal_beliefs, obs_type_to_id, object_to_id)
                        uniqueness = hmm.get_landmarks_uniqueness()
                        hmm.compute_likelihood_table(heuristic_ratio, uniqueness)

                    object_to_action_ids = defaultdict(list)
                    for goal, ids in current_goals_landmarks.items():
                        if '(' in goal:
                            obj = goal.split('(')[-1].strip(') ')
                            object_to_action_ids[obj].extend(ids)

                    closest_object, diff_distance, new_observation= moving_closer(dict_3d_positions, hand_position_3d, last_distance)
                    closest_object_observations = generate_observations_live(dict_3d_positions, hand_position_3d, object_to_action_ids, mapping_info)
                    all_observations=[]
                    all_observations= closest_object_observations + new_observation
                    #print("All observations",all_observations)
                    list_of_observations = match_obs_with_landmarks_id(all_observations, mapping_info)
                    #print("All observations1",list_of_observations)
                    last_distance= save_last_distance(dict_3d_positions, hand_position_3d)
                    list_of_observations = [obs for obs in list_of_observations if obs != 99]

                vlm_thread = None



            #compare old object with new object (if the same object is detected in the new frame)
            if new_dict_of_objects.keys() != dict_of_objects.keys():
                common_object_list = []
                for object_name in new_dict_of_objects:
                    if object_name in dict_of_objects:
                        if similarities_between_boxes(dict_of_objects[object_name], new_dict_of_objects[object_name], 200):
                            common_object_list.append(object_name)

                sameobjects = False
                if len(dict_of_objects) == 0:
                    print("No object detected — skipping similarity_model call.")
                    continue

                object_to_action_ids = defaultdict(list)
                for goal, ids in current_goals_landmarks.items():
                    if '(' in goal:
                        obj = goal.split('(')[-1].strip(') ')
                        object_to_action_ids[obj].extend(ids)

                closest_object, diff_distance, new_observation = moving_closer(dict_3d_positions, hand_position_3d, last_distance)
                #print("CCCLOSEST OBJECT:", closest_object)
                closest_object_observations = generate_observations_live(dict_3d_positions, hand_position_3d, object_to_action_ids, mapping_info) #generated observations simulated to estimate the probability of each goal being pursued.
                all_observations=[]
                all_observations= closest_object_observations + new_observation
                list_of_observations = match_obs_with_landmarks_id(all_observations, mapping_info)
                print('dict   jkqKNJCQ      ', dict_3d_positions)
                last_distance= save_last_distance(dict_3d_positions, hand_position_3d)
                goal_candidate= goals_candidate(dict_3d_positions, hand_position_3d, list_of_goals)
                if goal_candidate:
                    list_of_goals.extend(goal_candidate)

                if "Undecided" not in list_of_goals:
                    list_of_goals.append("Undecided")


                # if the list of goals is different from the previous one, update the HMM
                if set(list_of_goals) != set(hmm.goal_beliefs.keys()):
                    # update the goal beliefs
                    hmm.goal_beliefs = {goal: 1 / len(list_of_goals) for goal in list_of_goals}

                    # update the transition probabilities
                    transition_proba = create_transition_matrix(hmm.goal_beliefs, list_of_goals, loaded_matrix)
                    hmm.transition_proba = transition_proba

                    # update the landmarks
                    hmm.current_goals_landmarks = build_current_goals_landmarks(hmm.goal_beliefs, obs_type_to_id, object_to_id)
                    uniqueness = hmm.get_landmarks_uniqueness()
                    hmm.compute_likelihood_table(heuristic_ratio, uniqueness)

                #alpha, current_goal = hmm.assisted_teleop(updating_time, memory_loss_value, list_of_observations)



            if len(dict_of_objects) == 0:
                    print("⚠️ No object detected — skipping similarity_model call.")
                    continue
            object_to_action_ids = defaultdict(list)
            for goal, ids in current_goals_landmarks.items():
                if '(' in goal:
                    obj = goal.split('(')[-1].strip(') ')
                    object_to_action_ids[obj].extend(ids)
            last_distance = save_last_distance(dict_3d_positions, hand_position_3d)

            print( last_distance)
            closest_object, diff_distance, new_observation = moving_closer(dict_3d_positions, hand_position_3d, last_distance)
            closest_object_observations = generate_observations_live(dict_3d_positions, hand_position_3d, object_to_action_ids, mapping_info) #generated observations simulated to estimate the probability of each goal being pursued.
            print("closest_object_observations",closest_object_observations)
            all_observations=[]
            all_observations= closest_object_observations + new_observation
            print("All observations",all_observations)
            list_of_observations = match_obs_with_landmarks_id(all_observations, mapping_info)
            print("All observations1",list_of_observations)
            last_distance= save_last_distance(dict_3d_positions, hand_position_3d)
            print(f"dict_3d_positions before: {dict_3d_positions}")
            z_positions=create_z_position_list(dict_3d_positions)
            print(f"z_positions: {z_positions}")
            min_dist = min(z_positions.values())
            grab_hmm = GrabHMM()
            state, conf = grab_hmm.update(min_dist)
            #print(f"min_dist = {min_dist} → State: {state}, confidence = {conf:.2f}")

            # Add the closest object as goal if it's not already in the list
            goal_candidate= goals_candidate(dict_3d_positions, hand_position_3d, list_of_goals)
            if goal_candidate:
                list_of_goals.extend(goal_candidate)
            if "Undecided" not in list_of_goals:
                list_of_goals.append("Undecided")

            # if the list of goals is different from the previous one, update the HMM
            if set(list_of_goals) != set(hmm.goal_beliefs.keys()):
                # update the goal beliefs
                hmm.goal_beliefs = {goal: 1 / len(list_of_goals) for goal in list_of_goals}

                # update the transition probabilities
                transition_proba = create_transition_matrix(hmm.goal_beliefs, list_of_goals, loaded_matrix)
                hmm.transition_proba = transition_proba

                # update the landmarks
                hmm.current_goals_landmarks = build_current_goals_landmarks(hmm.goal_beliefs, obs_type_to_id, object_to_id)
                uniqueness = hmm.get_landmarks_uniqueness()
                hmm.compute_likelihood_table(heuristic_ratio, uniqueness)

            #Method to update the goals
        if list_of_observations:
           print("list of obs to filter", list_of_observations)
           filtred_observations= filter_observations(list_of_observations, list_of_goals, mapping_info)
           alpha, current_goal = hmm.assisted_teleop(updating_time, memory_loss_value, filtred_observations)

        for value in hmm.goal_beliefs.values():
                if value > threshold_proba:
                    new_goal_achieved = True


        #if background == "video":
        if background == "camera" or background == "video":
            object_goals = online_goal_estimation(camera, hmm.goal_beliefs, dict_of_objects, gaze_finalvalue)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    #zed.close()
    #cap.release()
    cv2.destroyAllWindows()

import pyzed.sl as sl
import cv2
import time
import numpy as np
import re
import json
import csv
import open3d as o3d
from collections import defaultdict


def load_object_types(csv_path="goals_type.csv"):
    mapping = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            obj = row["object"].strip()
            typ = row["type"].strip()
            mapping[obj].append(typ)
    return mapping

OBJECT_TYPES = load_object_types("goals_type.csv")

def group_by_type(objects_dict):
    grouped = defaultdict(list)
    for name in objects_dict.keys():
        if name in OBJECT_TYPES:
            for typ in OBJECT_TYPES[name]:
                if name not in grouped[typ]:
                    grouped[typ].append(name)
    return grouped
def find_gaze_value(camera):
        #print("Eyes tracking data in real time:", camera.eye_tracker.latest_data)
        #print("Eyes tracking pos:", camera.eye_tracker.gaze_position)
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


def video_object_recognition(video_path, yolo_model):

    zed = sl.Camera()
    input_type = sl.InputType()
    input_type.set_from_svo_file(video_path)

    init = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)
    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        print("❌ Unable to open SVO file.")
        exit()

    mat = sl.Mat()

    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(mat, sl.VIEW.LEFT)
        frame = mat.get_data()
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    else:
        print("Unable to grab image from SVO file.")
        return [], None

    detections = yolo_model.detect_objects(frame)

    #cap.release()
    zed.close()
    return detections, frame

def online_objet_recognition(camera_Zed,frame_skip=0, is_video=False):
    # Initialize ZED camera and YOLOv8 model

    runtime_params = sl.RuntimeParameters()
    mat = sl.Mat()

    if camera_Zed.zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        camera_Zed.zed.retrieve_image(mat, sl.VIEW.LEFT)  # image extracted from ZED camera
        frame = mat.get_data() # image in numpy format

        if frame.shape[2] == 4:  # RGBA image
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR) # Conversion in BGR

        detections = camera_Zed.detect_objects(frame) # Run YOLOv8 detection
        if is_video and frame_skip>0:
            current_frame = camera_Zed.zed.get_svo_position()
            camera_Zed.zed.set_svo_position(current_frame + frame_skip)

        return detections, frame

    return [], frame

def rename_objects(detections, camera):
    object_count = {}
    renamed_objects = {}

    for obj in detections:
        class_id = obj["class_id"]
        # Increment count for the object in the dictionary
        if class_id in object_count:
            object_count[class_id] += 1
            renamed_objects[f"{camera.model.names[class_id]}{object_count[class_id]}"] = obj["box"]
        else:
            object_count[class_id] = 1
            renamed_objects[camera.model.names[class_id]] = obj["box"]

    return renamed_objects

def build_all_possible_goals(list_of_actions, list_of_objects):
    all_possible_goals = []
    for action in list_of_actions:
        for obj in list_of_objects:
            goal = action.replace("object1", obj)
            all_possible_goals.append(goal)
    return all_possible_goals

def compute_3d_position_from_mask(mask, point_cloud, stride=8, min_valid_points=20):
    h, w = mask.shape
    points = []

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            if mask[y, x]:
                err, point = point_cloud.get_value(x, y)
                if err == sl.ERROR_CODE.SUCCESS:
                    X, Y, Z, _ = point
                    if not any(map(np.isnan, (X, Y, Z))) and Z > 0 and Z < 8000:
                        points.append([X, Y, Z])

    if len(points) < min_valid_points:
        return None

    return np.median(points, axis=0)

def threaded_VLM_wrapper(model_name, caption, frame, objects, timing, result_container, list_of_actions):
    result = processorLL.VLM_process_func(model_name, caption, frame, objects, list_of_actions, timing)
    print("VLM output:", result)
    result_container["result"] = result

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

            #print("v1:", mapping[goal])

        #print("mapping:", mapping)
        return mapping

def mapping_infos(obs_type_to_id, object_to_id):
    landmark_info = {}  # Stores landmark_id → {"type": ..., "object": ..., "id": ...}
    for obj, obj_id in object_to_id.items():
        for obs_type, obs_type_id in obs_type_to_id.items():
            landmark_id = obs_type_id + obj_id
            # Save landmark info if not already recorded
            if landmark_id not in landmark_info:
                landmark_info[landmark_id] = {
                    "type": obs_type,
                    "object": obj,
                    "id": landmark_id
                }
    #print("landmark_info:", landmark_info)
    return landmark_info

def moving_closer(dict_3d_positions, hand_position_3d, last_distance):
    diff_distance=dict_3d_positions.copy()
    closest_object= []
    new_observation = []

    #print(f"test diff distance 3d avant", diff_distance)
    for name, pos in diff_distance.items():
        if name in last_distance:
            distance = np.linalg.norm(pos -hand_position_3d)
            diff = last_distance[name] - distance
            if diff>0 :
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
            #delete the line of name in diff_distance
            #del diff_distance[name]
            del last_distance[name]
            #print(f"object deleted:", name,"in", last_distance)

    #print(f"test diff distance 3d apres", diff_distance)
    #print(f"closest object", closest_object)
    #print(f"new observations", new_observation)
    return closest_object, diff_distance, new_observation

def save_last_distance(dict_3d_positions, hand_position_3d):
    last_distance=dict_3d_positions.copy()
    #print(f"test print distance 3d avant", last_distance)
    for name, pos in last_distance.items():
        distance = np.linalg.norm(pos -hand_position_3d)
        last_distance[name]= distance
    #print(f"test print distance 3d apres", last_distance)
    return last_distance

def ID_to_text(ID, mapping_info):
    for index, landmark_text in mapping_info.items():
        if index==ID:
            return landmark_text

def generate_observations_live(dict_3d_positions, hand_position_3d, object_to_action_ids, mapping_info, threshold=1000):
    observations = []
    test_observations = {}
    save_observations = {}
    # Step 1: Find nearby objects
    for name, pos in dict_3d_positions.items():
        distance = np.linalg.norm(pos - hand_position_3d)
        if distance < threshold:
            save_observations[name] = distance

    # Step 2: Sort objects by distance
    sorted_objects = sorted(save_observations.items(), key=lambda item: item[1])

    if not sorted_objects:
        print("No objects detected within the threshold.")
        return observations

    # Step 3: Take the closest object
    closest_object = sorted_objects[0][0]  # name
    observation_type = "closest_object"

    test_observations['type']= observation_type
    test_observations['object']=closest_object
    observations.append(test_observations)

    #print("Observations generated:", observations)
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

def goals_candidate(dict_3d_positions, hand_position_3d, list_of_goals, path="time_spent.csv"):                    # Add the closest object as goal if it's not already in the list
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
    camera.draw_goal_boxes(frame, object_boxes, object_goals)
    camera.draw_gaze(frame, gaze_finalvalue)


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

def online_goal_estimation(camera_Zed, goals_beliefs, list_of_bounding_boxes, previous_object_goals, gaze_finalvalue):
    if previous_object_goals is None:
        previous_object_goals = {}

    runtime_params = sl.RuntimeParameters()
    mat = sl.Mat()
    #object_goals = defaultdict(lambda: [None, -1], previous_object_goals)
    object_goals = defaultdict(lambda: [None, -1])

    for goal, probability in goals_beliefs.items():
        object_names = []
        if '(' in goal and ';' not in goal:
            object_names = [goal.split('(')[-1].strip(') ')]
        elif ';' in goal:
            inner = goal.split('(')[-1].strip(')')
            object_names = [name.strip() for name in inner.split(';')]

        # update individual object goals
        for object_name in object_names:
            if object_name:
                current_prob = object_goals[object_name][1]
                if probability > current_prob:
                    object_goals[object_name] = [goal, probability]

    # image treatment
    if camera_Zed.zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        camera_Zed.zed.retrieve_image(mat, sl.VIEW.LEFT)
        frame = mat.get_data()

        if frame.shape[2] == 4:  # RGBA image
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Draw bounding boxes on the frame with custom labels and confidences
        camera_Zed.draw_goal_boxes(frame, list_of_bounding_boxes, object_goals)
        if gaze_finalvalue != "Unknown":
        	camera.draw_gaze(frame, gaze_finalvalue)
        ##
        point_cloud = sl.Mat()
        camera.zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

        mask_overlay = np.zeros_like(frame, dtype=np.uint8)

        for det in detections:
            dict_3d_positions = {}
            mask = det.get("mask")
            if mask is not None:
                resized_mask = cv2.resize(mask, (point_cloud.get_width(), point_cloud.get_height()), interpolation=cv2.INTER_NEAREST)
                pos = compute_3d_position_from_mask(resized_mask, point_cloud)
                #print(f"pos: {pos}")
                if pos is not None:
                    x, y, z = pos
                    #print(f"[3D] {camera.model.names[det['class_id']]} → X={x:.2f}, Y={y:.2f}, Z={z:.2f}")
                    resized_mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                    mask_overlay[resized_mask == 1] = [255, 0, 0]
                    dict_3d_positions[name] = pos
        frame = cv2.addWeighted(frame, 1.0, mask_overlay, 0.5, 0)

        fps_times.append(time.time())
        if len(fps_times) >= 2:
            fps = len(fps_times) / (fps_times[-1] - fps_times[0])
        else:
            fps = 0.0

        # display FPS on the frame
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        y_offset = 100
        for key, val in timing_stats.items():
            cv2.putText(frame, f"{key}: {val:.2f}s", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_offset += 20
        # Display both frames
        cv2.putText(frame, shared_caption.value[:60], (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
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

def visualize_sl_point_cloud(point_cloud_sl_mat):
    """
    Visualizes a ZED SDK point cloud (sl.Mat) using Open3D.

    Parameters:
        point_cloud_sl_mat (sl.Mat): A ZED point cloud filled with XYZRGBA data.
    """
    # Step 1: Extract data from sl.Mat
    pc_np = point_cloud_sl_mat.get_data()  # shape: (H, W, 4), dtype: float32

    # Step 2: Flatten and remove invalid points
    pc_flat = pc_np.reshape(-1, 4)
    valid = np.isfinite(pc_flat[:, 2])  # Only keep points with valid Z
    pc_valid = pc_flat[valid]

    if pc_valid.size == 0:
        #print("No valid 3D points to display.")
        return

    # Step 3: Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_valid[:, :3])  # XYZ

    # Step 4: Extract and normalize RGB from packed float RGBA
    rgba = pc_valid[:, 3].astype(np.uint32).view(np.uint8).reshape(-1, 4)
    rgb = rgba[:, :3] / 255.0
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    # Step 5: Visualize
    o3d.visualization.draw_geometries([pcd])

def visualize_object_from_mask(mask,point_cloud,name="object"):
    h,w = mask.shape
    points = []
    colors = []
    for y in range(h):
        for x in range(w):
            if mask[y,x]:
                err, point =point_cloud.get_value(x, y)
                if err == sl.ERROR_CODE.SUCCESS:
                    X, Y, Z, rgba = point
                    if np.isfinite(X) and np.isfinite(Y) and np.isfinite(Z) and Z>0 and Z<80000000000000000:
                        points.append([X,Y,Z])
                        rgba_int =np.uint32(rgba)
                        r = (rgba_int >> 24) & 0xFF
                        g = (rgba_int >> 16) & 0xFF
                        b = (rgba_int >> 8) & 0xFF
                        colors.append([r/255.0, g/255.0, b/255.0])  # Normalize RGB values
    if not points:
        #print (f"No valid 3D points found for {name}.")
        return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

    #print(f"Visualizing {name} with {len(points)} points.")
    o3d.visualization.draw_geometries([pcd])

def create_z_position_list(dict_3d_positions):

    z_positions = {}
    for (name,pos) in dict_3d_positions.items():
        if pos is not None and len(pos) == 3:
            z_positions[name]= pos[2]  # Extract the Z coordinate
    return z_positions

def load_transition_matrix(path="transition_proba_for_hmm.json"):
    with open(path, "r", encoding="utf-8") as f:
        transition_matrix = json.load(f)
    #print(f"[INFO] Transition matrix loaded from {path}")
    return transition_matrix

def filter_observations(list_of_observations, list_of_goals):
    filtred_observations = []
    for obs in list_of_observations:
        #print("list of observations test",list_of_observations)
        #print("obs test", obs)
        #print("list of goals test", list_of_goals)
        #print("obs:", obs)
        obs_text= ID_to_text(obs, mapping_info)
        #print("obs in words", obs_text["object"])
        for obj in list_of_goals:
            if f"({obs_text['object']})" in obj:
                #print("test working")
                id_act=(obs)
                #print("id_act:",id_act)
                build_obs = (id_act, obj)
                filtred_observations.append(build_obs)
    #print(f"filtred_observations: {filtred_observations}")
    return filtred_observations
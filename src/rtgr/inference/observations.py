import csv
from collections import defaultdict
import numpy as np
from rtgr.config.paths import TIME_SPENT_CSV


def load_object_types(csv_path="goals_type.csv"):
    mapping = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            obj = row["object"].strip()
            typ = row["type"].strip()
            mapping[obj].append(typ)
    return mapping

def moving_closer(objects_3d, hand_position_3d, last_distance):
    if objects_3d is None :
        return []


    new_observation = []
    for name, pos in objects_3d.items():
        if name in last_distance and name != 'person':
            distance = np.linalg.norm(pos -hand_position_3d)
            diff = last_distance[name] - distance
            if diff> 1e-6 : 

                new_obs= {}
                new_obs['type']= "moving_closer"
                new_obs['object']= name
                new_observation.append(new_obs)

    return new_observation

def generate_observations_live(objects_3d, hand_position_3d, threshold=1.0):
    observations = []
    test_observations = {}
    save_observations = {}
    if objects_3d:
        for name, pos in objects_3d.items():
            # if name != 'person':
            distance = np.linalg.norm(pos - hand_position_3d)
            if distance < threshold:
                save_observations[name] = distance

        # Step 2: Sort objects by distance
        sorted_objects = sorted(save_observations.items(), key=lambda item: item[1])

        if not sorted_objects:
            print("No objects detected within the threshold.")
            return observations

        closest_object = sorted_objects[0][0]  # name
        observation_type = "closest_object"

        test_observations['type']= observation_type
        test_observations['object']=closest_object
        observations.append(test_observations)

        return observations
    return []

def match_obs_with_landmarks_id(observations, mapping_info):
    observations_ID=[]
    for obs in observations:
        observation_type= obs['type']
        type_object= obs['object']
        for lid, info in mapping_info.items():
            if info["type"] == observation_type and info["object"] == type_object:
                observations_ID.append(info["id"])
    return observations_ID


def save_last_distance(objects_3d, hand_position_3d):
    if objects_3d :
        for name, pos in objects_3d.items():

            distance = np.linalg.norm(pos - hand_position_3d)
            objects_3d[name] = distance
        return objects_3d
    return None


def ID_to_text(ID, mapping_info):
    for index, landmark_text in mapping_info.items():
        if index==ID:
            return landmark_text
        
def infer_goal_from_closest_objects(objects_3d, hand_position_3d, list_of_goals, path):                    
    if not objects_3d:
        print("Can't have the 3D pos so no possible candidate")
        return None

    if objects_3d:
        distances = {
            name: np.linalg.norm(pos - hand_position_3d)
            for name, pos in objects_3d.items()
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

    return []

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


def build_all_possible_goals(list_of_actions, list_of_objects):
    all_possible_goals = []
    for action in list_of_actions:
        for obj in list_of_objects:
            goal = action.replace("object1", obj)
            all_possible_goals.append(goal)
    return all_possible_goals


def update_hand_object_distances(objects_3d, last_hand_objects_distance, hand):
    new_distances = {}
    getting_closer = {}

    #persistent objects
    common_objects = objects_3d.keys() & last_hand_objects_distance.keys()

    for obj_name in common_objects:
        obj_pos = objects_3d[obj_name]

        new_dist = np.linalg.norm(hand._3Dpos - obj_pos)
        last_dist = last_hand_objects_distance[obj_name]

        new_distances[obj_name] = new_dist

        if hand.is_getting_closer(obj_name, new_dist, last_dist):
            getting_closer[obj_name] = new_dist

    return new_distances, getting_closer

def get_observations(objects_3D, hand_3D, last_distance , mapping_info   ):
    # new_distances, getting_closer = update_hand_object_distances(objects_3D, ancienne_distance, hand)
    # print(new_distances, getting_closer)
    if objects_3D and last_distance:
        new_observation= moving_closer(objects_3D, hand_3D, last_distance)
        closest_object_observations = generate_observations_live(objects_3D, hand_3D)
        all_observations= closest_object_observations + new_observation
        list_of_observations = match_obs_with_landmarks_id(all_observations, mapping_info)
        list_of_observations = [obs for obs in list_of_observations if obs != 99]
        return list_of_observations
    return []


def update_goals_from_proximity (goals, objects_3D, hand_3D, hmm):
    goal_candidate= infer_goal_from_closest_objects(objects_3D, hand_3D, goals, TIME_SPENT_CSV)
    if goal_candidate:
        goals.extend(goal_candidate)
    if "Undecided" not in goals:
        goals.append("Undecided")

    if set(goals) != set(hmm.goal_beliefs.keys()):
        hmm.reset_goal_inference(goals)
from ultralytics import YOLO

import pandas as pd
import json
import math

def extract_pourable_objects(csv_path: str):
    """
    Lit goals_type.csv et renvoie les objets dont type == 'pourable'.
    Attendu: colonnes 'object' et 'type'.
    """
    df = pd.read_csv(csv_path)
    if 'object' not in df.columns or 'type' not in df.columns:
        raise ValueError("goals_type.csv doit contenir les colonnes 'object' et 'type'.")
    mask = df['type'].astype(str).str.lower() == 'pourable'
    return set(df.loc[mask, 'object'].astype(str).str.strip())

def get_yolo_object_list(model_path: str):
    model = YOLO(model_path)                 # ex: 'yolov8n.pt' ou 'best.pt'
    names = getattr(model, "names", None)
    if names is None:
        names = getattr(getattr(model, "model", None), "names", None)
    if isinstance(names, dict):
        return [names[k] for k in sorted(names.keys())]
    return list(names)



def load_goal_durations(csv_path):

    df = pd.read_csv(csv_path)

    if 'Goal' not in df.columns or 'Time(seconds)' not in df.columns:
        raise ValueError("CSV must contain 'Goal' and 'Time(seconds)' columns.")

    return dict(zip(df["Goal"], df["Time(seconds)"]))


def compute_transition_matrix(goal_durations, delta_t=0.1, p_interrupt=0.0):

    transition_matrix = {}

    for goal, duration in goal_durations.items():
        if duration <= 0:
            p_switch = 1.0
        else:
            #p_stay_natural = 1 - delta_t / duration
            #p_stay = max(0.0, p_stay_natural * (1 - p_interrupt))
            p_switch= delta_t/ duration
            #x=-delta_t/ duration
            #p_stay = math.exp(x)

        p_stay = 1 - p_switch

        transition_matrix[goal] = {
            goal: round(p_stay, 4),     # Stay in same goal
            "other": round(p_switch, 4)  # Switch to other goal
        }

    return transition_matrix

def build_all_possible_goals(list_of_actions, list_of_objects):
    all_possible_goals = []
    for action in list_of_actions:
        for obj in list_of_objects:
            goal = action.replace("object1", obj)
            all_possible_goals.append(goal)
    return all_possible_goals

def create_full_transition_matrix(transition_matrix, not_found_goals):

    for goal in not_found_goals:
        if goal not in transition_matrix:
            transition_matrix[goal] = {goal: 1.0, "other": 0.0}
    return transition_matrix


def inject_related_transitions(transition_matrix, related_goals_path, p_related_ratio=0.9):

    # Load related goals (JSON Lines format: one JSON object per line)
    related_goals = {}
    with open(related_goals_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                line_dict = json.loads(line)
                related_goals.update(line_dict)
            except json.JSONDecodeError as e:
                print(f"[ERROR] JSON decode error: {e}\nLine: {line}")
                continue

    enhanced_matrix = {}

    for goal, transitions in transition_matrix.items():
        p_stay = transitions.get(goal, 0.0)
        p_switch = transitions.get("other", 0.0)
        related = related_goals.get(goal, {})

        new_transitions = {goal: p_stay}

        if related:
            total_weight = sum(float(v) for v in related.values())
            p_related_total = p_switch * p_related_ratio

            for related_goal, weight in related.items():
                new_transitions[related_goal] = round(p_related_total * float(weight) / total_weight, 6)

            p_unrelated = round(p_switch * (1 - p_related_ratio), 6)
            new_transitions["other"] = p_unrelated
        else:
            # No related goals found — keep original "other"
            new_transitions["other"] = p_switch

        enhanced_matrix[goal] = new_transitions

    return enhanced_matrix

def create_goal_to_goal_matric(enhanced_matrix, all_goals):
    transition_proba={}
    n= len(all_goals)
    for goal in all_goals:
        transitions = enhanced_matrix.get(goal,None)

        if transitions is None:
            p_uniform= round(1/n, 6)
            transition_proba[goal] = [p_uniform for _ in all_goals]
            continue

        row =[]

        known_transitions={g:p for g,p in transitions.items() if g !="other"}

        p_other = transitions.get("other", 0.0)

        missing_goals = [g for g in all_goals if g !=goal and g not in known_transitions]

        p_unrelated = p_other/ len(missing_goals) if missing_goals else 0.0

        for target_goal in all_goals:
            if target_goal == goal:
                row.append(transitions.get(goal,0.0))
            elif target_goal in known_transitions:
                row.append(known_transitions[target_goal])
            else:
                row.append(round(p_unrelated, 6))
        
        transition_proba[goal]=row
    return transition_proba


def save_transition_matrix(transition_matrix, path="transition_proba_for_hmm.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(transition_matrix, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Transition matrix saved to {path}")




if __name__ == "__main__":
    csv_file = "time_spent.csv"  # Replace with your actual file path
    delta_t = 0.1                # HMM update every 0.1s
    p_interrupt = 0.05            # 5% chance user may change goal
    path_HMM="transition_proba_for_hmm.json"  
    list_of_actions = ["grab(object1)", "push(object1)", "place(object1)", "pull(object1)", "press(object1)"]
    pourable_csv_file = "goals_type.csv"   # CSV des types (object,type)
    yolo_model_path = "yolov11n-seg.pt"
    object_list = get_yolo_object_list(yolo_model_path)
    print(f"[INFO] YOLO classes loaded: {len(object_list)} objects")

    """
    object_list = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    """
    
    durations = load_goal_durations(csv_file)
    transition_matrix = compute_transition_matrix(durations, delta_t, p_interrupt)
    all_possible_goals = build_all_possible_goals(list_of_actions, object_list)
    pourable_objects = extract_pourable_objects(pourable_csv_file)
    all_possible_goals.extend([f"pour({o})" for o in pourable_objects])
    all_possible_goals.append("Undecided")
    print("Transition matrix:", transition_matrix)
    print("All possible goals:", all_possible_goals)
    path= "related_goals.json"
    enhanced_matrix=inject_related_transitions(transition_matrix, path)
    print("enhanced_matrix", enhanced_matrix)
    not_found_goals = [goal for goal in all_possible_goals if goal not in transition_matrix]
    if not_found_goals:
        print("Warning: The following goals were not found in the transition matrix:")
        for goal in not_found_goals:
            print(goal)
    #build full matrix for all_possible_goals updated by enhanced_matrix value
    transition_proba_for_hmm = create_goal_to_goal_matric(enhanced_matrix, all_possible_goals)
    print("transition proba for hmm", transition_proba_for_hmm)
    save_transition_matrix(transition_proba_for_hmm,path_HMM)





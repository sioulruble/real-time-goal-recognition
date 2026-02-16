import re
import json
import pandas as pd


'''
a bunch of functions for compute transitions in hmm
'''


def extract_pourable_objects(csv_path: str):
    df = pd.read_csv(csv_path)
    if 'object' not in df.columns or 'type' not in df.columns:
        raise ValueError("goals_type.csv doit contenir les colonnes 'object' et 'type'.")
    mask = df['type'].astype(str).str.lower() == 'pourable'
    return set(df.loc[mask, 'object'].astype(str).str.strip())

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


def create_full_transition_matrix(transition_matrix, not_found_goals):

    for goal in not_found_goals:
        if goal not in transition_matrix:
            transition_matrix[goal] = {goal: 1.0, "other": 0.0}
    return transition_matrix

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

def save_transition_matrix(transition_matrix, path="transition_proba_for_hmm.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(transition_matrix, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Transition matrix saved to {path}")


def normalize_goal(goal):
    match = re.match(r"(\w+)\((\w+?)(\d*)\)", goal)
    if match:
        action, obj_base, _ = match.groups()
        return f"{action}({obj_base})"
    return goal

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


def load_transition_matrix(path):
    with open(path, "r", encoding="utf-8") as f:
        transition_matrix = json.load(f)
    return transition_matrix


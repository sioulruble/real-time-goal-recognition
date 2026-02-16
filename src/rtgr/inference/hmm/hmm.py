import math
from typing import Dict, List, Tuple
from .utils import *
from rtgr.config.paths import TRANSITION_MATRIX, TIME_SPENT_CSV
from rtgr.config.constants import OBS_TYPE_TO_ID, HMM_MEMORY_LOSS_VALUE, HMM_UPDATING_TIME, HMM_MIN_PERSISTENCE, HMM_HEURISTIC_RATIO
from rtgr.perception.object_detection import detect_objects
from rtgr.inference.observations import infer_goal_from_closest_objects
import numpy as np 
class HMM:
    def __init__(self, goal_beliefs: Dict[str, float], current_goals_landmarks: Dict[str, int], objects_labels , heuristic_ratio = HMM_HEURISTIC_RATIO,
                 updating_time=HMM_UPDATING_TIME, memory_loss_value=HMM_MEMORY_LOSS_VALUE, min_persistence=HMM_MIN_PERSISTENCE):

        self.goal_beliefs = goal_beliefs
        self.decreasing_actions = []
        self.current_goals_landmarks = current_goals_landmarks
        self.likelihood_table = {}
        self.action_counter = {}
        self.min_persistence = min_persistence
        self.updating_time = updating_time
        self.memory_loss_value = memory_loss_value
        self.heuristic_ratio = heuristic_ratio
    
        self.loaded_matrix= load_transition_matrix(TRANSITION_MATRIX)
        self.transition_proba =create_transition_matrix(self.goal_beliefs, self.goal_beliefs , self.loaded_matrix)
        self.object_label_to_id = {obj: idx for idx, obj in enumerate(objects_labels)}
        self.heuristic_ratio = heuristic_ratio
        self.uniqueness = self.get_landmarks_uniqueness()
        

    def get_landmarks_uniqueness(self):
        # Initialize landmarks_uniqueness
        list_of_landmarks = []
        for landmarks in self.current_goals_landmarks.values():
            for elem in landmarks:
                if elem not in list_of_landmarks:
                    list_of_landmarks.append(elem)
        landmarks_uniqueness = {landmark:0.0 for landmark in list_of_landmarks}

        # Update landmarks_uniqueness based on current_goals_landmarks
        for goal in self.goal_beliefs:
            tmp_landmarks = self.current_goals_landmarks[goal]
            for landmark in tmp_landmarks:
                landmarks_uniqueness[landmark] += 1

        # Adjust uniqueness values
        for landmark, value in landmarks_uniqueness.items():
            if value > 0:
                landmarks_uniqueness[landmark] = 1.0 / value

        return landmarks_uniqueness
    
    def compute_likelihood_table(self, ratio = 0.6):
        self.likelihood_table = {}
        number_of_goals = len(self.goal_beliefs)

        # prepare the list of all action ids
        all_action_ids = set()
        for landmarks in self.current_goals_landmarks.values():
            all_action_ids.update(landmarks)

        # Initialize for all action
        for action_id in all_action_ids:
            uq = self.uniqueness.get(action_id, 0.0)
            value0 = max(ratio * math.exp(uq - 1), (1 - ratio) / number_of_goals)
            value1 = (1 - ratio) / number_of_goals
            self.likelihood_table[action_id] = [value0, value1]
        
    def get_likelihood(self, memory_loss: float, list_of_observations: List[Tuple[int, str]]):
        likelihood = {}
        n = len(self.goal_beliefs)
        min_likelihood = 1.0 / n if n > 0 else 0.0
        current_actions = [obs[0] for obs in list_of_observations]

        for action in current_actions:
            self.action_counter[action] = self.action_counter.get(action, 0) + 1

        # reset unseen actions
        for action in list(self.action_counter.keys()):
            if action not in current_actions:
                self.action_counter[action] = 0

        for action, observed_goal in list_of_observations:
            print("----")
            print(action , observed_goal)
            likelihood[action] = []

            # persistence factor (0 → 1)
            persistence = min(
                1.0,
                self.action_counter[action] / self.min_persistence
            )

            if action in self.decreasing_actions:

                added_value = (
                    self.likelihood_table[action][0]
                    - self.likelihood_table[action][0] * memory_loss
                ) / len(self.goal_beliefs)

                for current_goal in self.goal_beliefs:
                    if observed_goal == current_goal:
                        likelihood[action].append(
                            self.likelihood_table[action][0] * persistence
                        )
                    else:
                        likelihood[action].append(
                            self.likelihood_table[action][1] * persistence
                        )

                # decay likelihood memory
                self.likelihood_table[action][0] = max(
                    self.likelihood_table[action][0] * memory_loss,
                    min_likelihood
                )
                self.likelihood_table[action][1] = min(
                    self.likelihood_table[action][1] + added_value,
                    min_likelihood
                )

            else:
                for current_goal in self.goal_beliefs:
                    if observed_goal == current_goal:
                        likelihood[action].append(
                            self.likelihood_table[action][0] * persistence
                        )
                    else:
                        # print("action:", action)
                        # print("LIKELIHOOD :", likelihood)

                        # print("LIKELIHOOD table:", self.likelihood_table)
                        likelihood[action].append(
                            self.likelihood_table[action][1] * persistence
                        )

        return likelihood

    def assisted_teleop(self, list_of_observations: List[Tuple[int, str]]):
        

        number_of_goals = len(self.goal_beliefs)
        if number_of_goals == 0:
            return 0.0, "Undecided"
        memory_loss = math.pow(self.memory_loss_value, 1.0 / (1.0 / self.updating_time))
        likelihood = self.get_likelihood(memory_loss, list_of_observations)
        sum_beliefs = 0
        previous_beliefs = self.goal_beliefs.copy()

        for i, goal in enumerate(self.goal_beliefs):
            sum_proba = sum(
                self.transition_proba[goal][k] * previous_beliefs[goal_bis]
                for k, goal_bis in enumerate(self.goal_beliefs)
            )
            product_proba = 1.0
            for observation in list_of_observations:
                product_proba *= likelihood[observation[0]][i]

            res = product_proba * sum_proba
            self.goal_beliefs[goal] = res
            sum_beliefs += res

        sorted_goals = sorted(
            self.goal_beliefs.items(), key=lambda x: x[1], reverse=True
        )
        current_goal, max_belief = sorted_goals[0]
        if len(sorted_goals) > 1:
            second_goal, second_max_belief = sorted_goals[1]

        for goal in self.goal_beliefs:
            self.goal_beliefs[goal] /= sum_beliefs if sum_beliefs > 0 else 1/number_of_goals

        alpha = 0
        delta1 = 0.2
        delta2 = 0.75

        max_entropy = -math.log(1.0 / number_of_goals)
        entropy = -sum(b * math.log(b) for b in self.goal_beliefs.values() if b > 0)
        if number_of_goals > 1:
            confidence = 1 - entropy / max_entropy
        else:
            confidence = max_belief

        if confidence > delta1:
            alpha = min(confidence, delta2)

        if current_goal == "Undecided":
            alpha = 0

        # return alpha, current_goal
    
    def reset_goal_inference(self, goal_hypotheses):
            self.goal_beliefs = {goal: 1 / len(goal_hypotheses) for goal in goal_hypotheses}
            self.transition_proba = create_transition_matrix(self.goal_beliefs, goal_hypotheses, self.loaded_matrix)
            self.current_goals_landmarks = build_current_goals_landmarks(self.goal_beliefs, OBS_TYPE_TO_ID, self.object_label_to_id)
            self.uniqueness = self.get_landmarks_uniqueness()
            self.compute_likelihood_table()     


    



def init_hmm(image, depth, model):
    _, objects_3D = detect_objects(image, depth, model)
    hand_3D = np.array([np.inf, np.inf, np.inf])

    goals = infer_goal_from_closest_objects(objects_3D, hand_3D, [], TIME_SPENT_CSV)
    goals = goals or []
    goals.append("Undecided")

    beliefs = {g: 1/len(goals) for g in goals}
    mapping = mapping_infos(OBS_TYPE_TO_ID, {o:i for i,o in enumerate(model.names.values())})
    classes = list(model.names.values()) if hasattr(model, "names") else []
    object_to_id=  {obj: idx for idx, obj in enumerate(classes)}
    hmm = HMM(beliefs, build_current_goals_landmarks(beliefs, OBS_TYPE_TO_ID, object_to_id), classes)
    hmm.compute_likelihood_table()
    print(hmm.likelihood_table)

    return hmm, goals, mapping, hand_3D 

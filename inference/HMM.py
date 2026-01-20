import math
from typing import Dict, List, Tuple

class HMM:
    def __init__(self, goal_beliefs: Dict[str, float], transition_proba: Dict[str, List[float]], current_goals_landmarks: Dict[str, int], decreasing_actions: List[int]):
 
        self.goal_beliefs = goal_beliefs
        self.decreasing_actions = decreasing_actions
        self.transition_proba = transition_proba
        self.current_goals_landmarks = current_goals_landmarks
        self.likelihood_table = {}


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
    
    def compute_likelihood_table(self, ratio: float, landmark_uniqueness: Dict[int, float]):
        self.likelihood_table = {}
        number_of_goals = len(self.goal_beliefs)

        # prepare the list of all action ids
        all_action_ids = set()
        for landmarks in self.current_goals_landmarks.values():
            all_action_ids.update(landmarks)

        # Initialize for all action
        for action_id in all_action_ids:
            uniqueness = landmark_uniqueness.get(action_id, 0.0)
            value0 = max(ratio * math.exp(uniqueness - 1), (1 - ratio) / number_of_goals)
            value1 = (1 - ratio) / number_of_goals
            self.likelihood_table[action_id] = [value0, value1]
        
        #print("Likelihood table initialized:", self.likelihood_table)


    def get_likelihood(self, memory_loss: float, list_of_observations: List[Tuple[int, str]]):
        likelihood = {}
        min_likelihood = 1.0 / len(self.goal_beliefs)

        for action, goal in list_of_observations:
            #print("GET LIKELIHOOD ACTION:", action)
            #print("GET LIKELIHOOD GOAL", goal)
            likelihood[action] = []

            if action in self.decreasing_actions:

                added_value = (self.likelihood_table[action][0] - self.likelihood_table[action][0] * memory_loss) / len(self.goal_beliefs)
                for current_goal in self.goal_beliefs:
                    if goal == current_goal:
                        likelihood[action].append(self.likelihood_table[action][0])
                        #print("LIKELIHOOD true:", self.likelihood_table[action][0])

                    else:
                        likelihood[action].append(self.likelihood_table[action][1])
                        #print("LIKELIHOOD false:", self.likelihood_table[action][1])

                # Update likelihood table
                self.likelihood_table[action][0] = max(self.likelihood_table[action][0] * memory_loss, min_likelihood)
                self.likelihood_table[action][1] = min(self.likelihood_table[action][1] + added_value, min_likelihood)

            else:
                for current_goal in self.goal_beliefs:
                    if goal == current_goal:
                        likelihood[action].append(self.likelihood_table[action][0])
                    else:
                        likelihood[action].append(self.likelihood_table[action][1])

        #print("Likelihood table updated:", self.likelihood_table)
        return likelihood


    def assisted_teleop(self, update_time: float, memory_loss_value: float, list_of_observations: List[Tuple[int, str]]):
        number_of_goals = len(self.goal_beliefs)
        memory_loss = math.pow(memory_loss_value, 1.0 / (1.0 / update_time))
        likelihood = self.get_likelihood(memory_loss, list_of_observations)
        #print("Likelihood table:", likelihood)

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

        #print("Current goal:", current_goal)
        #print("goal beliefs:", self.goal_beliefs)
        return alpha, current_goal

import math

class GrabHMM:
    def __init__(self, init_grab=0.1, init_not_grab=0.9, threshold_distance=20):
        self.state = ["not_grab", "grab"]
        self.beliefs = {"not_grab": init_not_grab, "grab": init_grab}
        self.transition = {
            "not_grab": {"not_grab": 0.9, "grab": 0.1},
            "grab": {"not_grab": 0.1, "grab": 0.9}
        }
        self.threshold_distance = threshold_distance
    
    def compute_likelihood(self, min_dist):
        grab_likelihood = math.exp(-min_dist / self.threshold_distance)
        not_grab_likelihood = 1 - grab_likelihood
        return {"not_grab": not_grab_likelihood, "grab": grab_likelihood}

    def update(self,min_dist):
        likelihood = self.compute_likelihood(min_dist)
        new_beliefs = {}
        
        for state in self.state:
            total = 0
            for prev_state in self.state:
                trans = self.transition[prev_state][state]
                total += trans * self.beliefs[prev_state]
            new_beliefs[state] = total * likelihood[state]

        total_sum = sum(new_beliefs.values())
        if total_sum > 0:
            for state in self.state:
                new_beliefs[state] /= total_sum
        
        self.beliefs = new_beliefs
        current_state = max(self.beliefs, key=self.beliefs.get)
        confidence = self.beliefs[current_state]
        return current_state, confidence
    

if __name__ == "__main__":

    distances = {'person': 901.32, 'refrigerator': 1378.95}

    min_dist = min(distances.values())
    #state, confidence = hmm.update(min_dist)
    hmm = GrabHMM()

    print("grab detection with HMM")
    state, conf = hmm.update(min_dist)
    print(f"min_dist = {min_dist} → State: {state}, confidence = {conf:.2f}")

import cv2
import numpy as np
import mediapipe as mp
from rtgr.perception.hand_tracking.hand_tracker import HandTracker    

class MediaPipeHand(HandTracker):
    def __init__(self):
        super().__init__()
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self._3Dpos = np.array([0,0,0])
        self.last_3Dpos = self._3Dpos


    def update_position(self, image, points3d):
        self.last_3Dpos = self._3Dpos
        self.results = self.hands.process(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        )

        if not self.results.multi_hand_landmarks:
            self._3Dpos = np.array([0,0,0])
            return np.array([0,0,0])

        hand = self.results.multi_hand_landmarks[0]

        xs = [lm.x for lm in hand.landmark]
        ys = [lm.y for lm in hand.landmark]

        cx = int(np.mean(xs) * image.shape[1])
        cy = int(np.mean(ys) * image.shape[0])

        self._3Dpos= points3d[cy, cx]
        cv2.circle(image, (cx, cy), 8, (0, 0, 255), -1)
    
    def get_position3d(self):
        return self._3Dpos


    def is_tracked(self):
        if self.results.multi_hand_landmarks == np.array([0,0,0]):
            return False
        return True
    
    def is_getting_closer(self, obj_name, obj_3d_pos, last_dist, eps=1e-3):


        if self._3Dpos is None or np.all(self._3Dpos == 0):
            return False

        if obj_3d_pos is None or np.any(np.isnan(obj_3d_pos)):
            return False

        curr_dist = np.linalg.norm(self._3Dpos - obj_3d_pos)
        last_dist = np.linalg.norm(self.last_3Dpos - last_dist)

        return curr_dist < (last_dist - eps)

    
    

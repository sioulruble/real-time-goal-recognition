# hand_ros_tracker.py

from rtgr.perception.hand_tracking.hand_tracker import HandTracker
import subprocess
import threading
import re
import numpy as np

class ROSHandTracker(HandTracker):
    """hand tracker using ros tf interface"""
    
    def __init__(self, 
                 hand_side="right",  # "left" ou "right"
                 from_frame="real/azure_color_frame", 
                 left_frame="real/L_HAND_LINK", 
                 right_frame="real/R_HAND_LINK"):
        super().__init__()
        self.hand_side = hand_side
        self.from_frame = from_frame
        self.left_frame = left_frame
        self.right_frame = right_frame
        
        self.left_position = None
        self.right_position = None
        self.running = True
        
        # Threads pour les deux mains
        self.left_thread = threading.Thread(target=self._read_tf, args=(self.left_frame, 'left'), daemon=True)
        self.right_thread = threading.Thread(target=self._read_tf, args=(self.right_frame, 'right'), daemon=True)
        self.left_thread.start()
        self.right_thread.start()
    
    def _read_tf(self, to_frame, hand):
        cmd = ["rosrun", "tf", "tf_echo", self.from_frame, to_frame]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        pattern = re.compile(r'Translation:\s+\[([-\d\.]+), ([-\d\.]+), ([-\d\.]+)\]')
        
        for line in process.stdout:
            match = pattern.search(line)
            if match:
                position = tuple(map(float, match.groups()))
                if hand == 'left':
                    self.left_position = position
                else:
                    # Conversion en mm si nécessaire
                    self.right_position = tuple(x * 1000 for x in position)
            
            if not self.running:
                break
        
        process.terminate()
    
    def update_position(self, image=None, points3d=None):

        if self.hand_side == "left":
            position = self.left_position
        else:
            position = self.right_position
        
        if position is not None:
            self._last_position_3d = np.array(position)
        else:
            self._last_position_3d = np.array([np.inf, np.inf, np.inf])
    
    def get_left_position(self):
        """Méthode legacy pour compatibilité"""
        return self.left_position
    
    def get_right_position(self):
        print("RIGHT HAND", self.right_position)
        return self.right_position
    
    def stop(self):
        self.running = False
        self.left_thread.join()
        self.right_thread.join()
    
    def __del__(self):
        self.stop()
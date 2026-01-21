
from inference.Hand_position import HandPositionReader
import time

hand_reader = HandPositionReader()

try:
    while True:
        left_hand = hand_reader.get_left_position()
        right_hand = hand_reader.get_right_position()        
        if left_hand:
            print(f"Left hand: {left_hand}")
        if right_hand:
            print(f"Right hand: {right_hand}")
            
        time.sleep(0.1)

except KeyboardInterrupt:
    hand_reader.stop()
    print("Stop")
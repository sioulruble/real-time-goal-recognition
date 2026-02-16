import cv2
from collections import defaultdict

def draw_bounding_boxes(frame, detections, classes, labels=None, confidences=None):
    for detection in detections:
        x, y, w, h = detection["box"]
        class_id = detection["class_id"]
        confidence = detection["confidence"]
        label = labels[class_id] if labels else f"{classes[class_id]}"
        display_confidence = confidences[class_id] if confidences and class_id in confidences else confidence
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f"{label}: {display_confidence:.2f}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

def draw_goal_boxes(frame, object_boxes, object_goals):
    if object_boxes is not None :
        for obj in object_boxes:
            x, y, w, h = object_boxes[obj]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            if obj not in object_goals or object_goals[obj][0] is None:
                label = f"analyzing({obj}) (?)"
                display_confidence = 0.0
            else:
                label = object_goals[obj][0]
                display_confidence = object_goals[obj][1]
            cv2.putText(frame, f"{label}: {display_confidence:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
def draw_gaze(frame, gaze_position, size=12, color=(0,255,255), thickness=2):
    if gaze_position is None:
        return
    try:
        x, y = map(int, gaze_position)
    except (ValueError, TypeError):
        return
    cv2.line(frame, (x-size,y), (x+size,y), color, thickness)
    cv2.line(frame, (x,y-size), (x,y+size), color, thickness)

def print_goals(goal_beliefs):
        print("\n" + "="*50)
        print("TOP 5 GOAL PROBABILITIES")
        print("="*50)
        sorted_goals = sorted(goal_beliefs.items(), key=lambda x: x[1], reverse=True)
        for i, (goal, probability) in enumerate(sorted_goals[:5], 1):
            bar_length = int(probability * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            print(f"{i}. {goal:30s} │{bar}│ {probability:.2%}")

def visualize(image, goals_beliefs, list_of_bounding_boxes, watch_goals = True):
    frame = image 
    if frame is None:
        return {}

    object_goals = defaultdict(lambda: [None, -1])

    for goal, probability in goals_beliefs.items():
        if '(' in goal:
            objects = goal.split('(')[1].strip(')').split(';')
            for obj in objects:
                if probability > object_goals[obj][1]:
                    object_goals[obj] = [goal, probability]

    draw_goal_boxes(frame, list_of_bounding_boxes, object_goals)
    cv2.imshow("Goal Recognition", frame)
    if watch_goals:
        print_goals(goals_beliefs)

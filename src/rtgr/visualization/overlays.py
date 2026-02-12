import cv2

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
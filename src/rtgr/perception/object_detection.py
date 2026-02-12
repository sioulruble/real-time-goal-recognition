import cv2
import numpy as np
from collections import defaultdict

def detect_objects( frame, model, prediction_conf_threshold=0.8):
    results = model.predict(source=frame, conf=prediction_conf_threshold, verbose=False)[0]
    detections = []
    for result in results:
        boxes = result.boxes 
        for i, box in enumerate(boxes):
            class_id = int(box.cls[0]) 
            prediction_confidence = float(box.conf[0]) 
            if prediction_confidence > prediction_conf_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                mask = result.masks.data[i].cpu().numpy() if result.masks is not None else None
                detections.append({
                    "class_id": class_id,
                    "confidence": prediction_confidence,
                    "box": [x1, y1, x2 - x1, y2 - y1], 
                    "mask": mask
                })
    # apply NMS
    if detections:
        boxes = np.array([[d["box"][0], d["box"][1], d["box"][0] + d["box"][2], d["box"][1] + d["box"][3]] for d in detections])
        confidences = np.array([d["confidence"] for d in detections])
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), prediction_conf_threshold, nms_threshold=0.4)
        detections = [detections[i] for i in indices.flatten()]
    return detections

def get_3Dpos_object(mask, depth):
    pts = depth[mask > 0].astype(np.float32)  # (N,3)

    pts = pts[np.all(pts != 0, axis=1)]
    if pts.shape[0] == 0:
        return None  # ou np.array([np.nan, np.nan, np.nan])

    return np.median(pts, axis=0)

def rename_objects(detections, model):
    object_count = {}
    renamed_objects = {}
    for obj in detections:
        class_id = obj["class_id"]
        if class_id in object_count:
            object_count[class_id] += 1
            renamed_objects[f"{model.names[class_id]}{object_count[class_id]}"] = obj["box"]
        else:
            object_count[class_id] = 1
            renamed_objects[model.names[class_id]] = obj["box"]
    return renamed_objects


def estimate_objects_3d_positions(detections, depth, class_names):
    h, w = depth.shape[:2]
    positions_3d = {}
    instance_counters = defaultdict(int)

    for det in detections:
        class_id = det["class_id"]
        base_name = class_names[class_id]
        mask = det.get("mask")

        if mask is None:
            continue

        # Instance naming (O(1))
        instance_counters[base_name] += 1
        name = (
            base_name
            if instance_counters[base_name] == 1
            else f"{base_name}{instance_counters[base_name]}"
        )

        # Resize mask only once, nearest to preserve labels
        resized_mask = cv2.resize(
            mask, (w, h), interpolation=cv2.INTER_NEAREST
        )

        if not np.any(resized_mask):
            continue

        pos = get_3Dpos_object(resized_mask, depth)
        if pos is not None:
            positions_3d[name] = pos

    return positions_3d
    
import cv2
import numpy as np
from collections import defaultdict
from datetime import datetime

def detect_objects( image, depth, model, prediction_conf_threshold=0.8):
    results = model.predict(source=image, conf=prediction_conf_threshold, verbose=False)[0]
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
        objects_2D = [detections[i] for i in indices.flatten()]
        class_names = list(model.names.values()) if hasattr(model, "names") else []
        objects_3D = estimate_objects_3d_positions(objects_2D, depth, class_names)
        objects_2D = rename_objects(objects_2D, model)
        return objects_2D, objects_3D     
    return None, None

def get_3Dpos_object(mask, depth):
    pts = depth[mask > 0].astype(np.float32) 

    pts = pts[np.all(pts != 0, axis=1)] 
    if pts.shape[0] == 0:
        return None  #np.array([np.nan, np.nan, np.nan])

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

        instance_counters[base_name] += 1
        name = (
            base_name
            if instance_counters[base_name] == 1
            else f"{base_name}{instance_counters[base_name]}"
        )

        resized_mask = cv2.resize(
            mask, (w, h), interpolation=cv2.INTER_NEAREST
        )

        if not np.any(resized_mask):
            continue

        pos = get_3Dpos_object(resized_mask, depth)
        if pos is not None:
            positions_3d[name] = pos

    return positions_3d
    

def rename_objects_3d_from_2d(objects_2d_named, objects_3d):
    renamed_3d = {}
    for name, obj_2d in objects_2d_named.items():
        obj_id = obj_2d.get("id")  # si tu as un id commun
        renamed_3d[name] = objects_3d.get(obj_id) if isinstance(objects_3d, dict) else None
    return renamed_3d


def make_yolo_detector(model, period_sec=0.3, conf=0.8):
    """
    YOLO is temporally gated to ensure stable object labels.
    Running detection at every frame causes class oscillations (YOLO is not a tracker)
    """
    last_run = None
    last_result = (None, None)

    def detect(image, depth):
        nonlocal last_run, last_result
        now = datetime.now()

        if last_run is None or (now - last_run).total_seconds() >= period_sec:
            last_run = now
            last_result = detect_objects(
                image=image,
                depth=depth,
                model=model,
                prediction_conf_threshold=conf
            )

        return last_result

    return detect
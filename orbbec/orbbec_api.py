import cv2
import numpy as np
import time

from pyorbbecsdk import (
    Pipeline,
    Config,
    OBSensorType,
    OBStreamType,
    AlignFilter
)

from .utils import frame_to_bgr_image
from ultralytics import YOLO
import threading


class TemporalFilter:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.prev = None

    def process(self, frame):
        frame = frame.astype(np.float32)
        if self.prev is None:
            self.prev = frame.copy()
            return frame
        out = cv2.addWeighted(frame, self.alpha, self.prev, 1 - self.alpha, 0)
        self.prev = out.copy()
        return out



class PointCloudLike:
    """
    Replacement for ZED point_cloud.get_value(u, v)

    Usage:
        X, Y, Z = pc.get_xyz(u, v)
    """

    def __init__(self, depth_mm, intrinsics):
        self.depth = depth_mm  # uint16 depth in mm
        self.fx = intrinsics["fx"]
        self.fy = intrinsics["fy"]
        self.cx = intrinsics["cx"]
        self.cy = intrinsics["cy"]


    def get_xyz(self, u, v):
        if (
            u < 0 or v < 0 or
            v >= self.depth.shape[0] or
            u >= self.depth.shape[1]
        ):
            return None

        Z = self.depth[v, u] / 1000.0  # mm → meters
        if Z <= 0:
            return None

        X = (u - self.cx) * Z / self.fx
        Y = (v - self.cy) * Z / self.fy

        return X, Y, Z


def get_color_intrinsics(pipeline):
    cam_param = pipeline.get_camera_param()
    intr = cam_param.rgb_intrinsic


    return {
        "fx": intr.fx,
        "fy": intr.fy,
        "cx": intr.cx,
        "cy": intr.cy
    }


def init_orbbec():
    try:
        pipeline = Pipeline()
        config = Config()

        # --- Depth stream ---
        depth_profiles = pipeline.get_stream_profile_list(
            OBSensorType.DEPTH_SENSOR
        )
        depth_profile = depth_profiles.get_default_video_stream_profile()
        config.enable_stream(depth_profile)

        # --- Color stream ---
        color_profiles = pipeline.get_stream_profile_list(
            OBSensorType.COLOR_SENSOR
        )
        color_profile = color_profiles.get_default_video_stream_profile()
        config.enable_stream(color_profile)

        pipeline.start(config)

        cam_param = pipeline.get_camera_param()
        depth_intr = cam_param.depth_intrinsic
        color_intr = cam_param.rgb_intrinsic
        fx, fy = depth_intr.fx, depth_intr.fy

        cx, cy = depth_intr.cx, depth_intr.cy
        width, height = depth_intr.width, depth_intr.height

        # --- fx, fy, cx, cy recalés si resize ---
        fx_scaled = (width / depth_intr.width) * depth_intr.fx
        fy_scaled = (height / depth_intr.height) * depth_intr.fy
        cx_scaled = width / 2
        cy_scaled = height / 2


    except Exception as e:
        print("❌ Erreur lors de l'initialisation Orbbec :", e)
        return None, None, None, None

    temporal_filter = TemporalFilter(alpha=0.5)
    align_filter = AlignFilter(
        align_to_stream=OBStreamType.COLOR_STREAM
    )

    print("✅ Orbbec initialisé avec succès.")
    return pipeline, temporal_filter, align_filter, (fx, fy, cx, cy, fx_scaled, fy_scaled, cx_scaled, cy_scaled)




def object_xyz_from_mask(mask, depth, pc_like, min_points=50):
    ys, xs = np.where(mask > 0)
    if len(xs) < min_points:
        return None

    zs = depth[ys, xs]
    zs = zs[zs > 0]
    if len(zs) < min_points:
        return None

    Z = np.median(zs) / 1000.0  # meters
    u = int(xs.mean())
    v = int(ys.mean())

    xyz = pc_like.get_xyz(u, v)
    if xyz is None:
        return None

    X, Y, _ = xyz
    return X, Y, Z


class OrbbecPlugin:
    """
    Adapter minimal pour utiliser Orbbec avec l'API attendue par mainvideo.py.

    - .model : YOLO model
    - .classes : list of class names
    - .pipeline, .temporal_filter, .align_filter, .intrinsics : orbbec internals
    - Methods:
        - detect_objects(frame, prediction_conf_threshold)
        - draw_gaze(frame, gaze_position, ...)
        - draw_bounding_boxes(frame, detections, classes, labels=None, confidences=None)
        - draw_goal_boxes(frame, object_boxes, object_goals)
        - retrieve_image(mat) -> fills mat.get_data() with BGR numpy array
        - retrieve_measure(point_cloud_mat) -> attaches get_width/get_height/get_value/get_data
    """
    def __init__(self, model_path, open_camera=True, width=640, height=480, fps=30):
        self.pipeline, self.temporal_filter, self.align_filter, self.intrinsics = init_orbbec()
        cam_param = self.pipeline.get_camera_param()
        self.width = width
        self.height = height
        self.fps = fps
        self.model = YOLO(model_path)
        self.classes = list(self.model.names.values()) if hasattr(self.model, "names") else []
        self.fx = self.intrinsics[0]
        self.fy = self.intrinsics[1]
        self.cx = self.intrinsics[2]
        self.cy = self.intrinsics[3]
        self.fx_scaled = self.intrinsics[4]
        self.fy_scaled = self.intrinsics[5]
        self.cx_scaled = self.intrinsics[6]
        self.cy_scaled = self.intrinsics[7]
        # cache last frames
        self._last_color = None
        self._last_depth = None
        self._last_3d = None
        self._frame_lock = threading.Lock()

    def _update_frames(self):
        rgb, depth = self.get_frames(min_depth=20, max_depth=10000)
        with self._frame_lock:
            self._last_color = rgb
            self._last_depth = depth
            if depth is not None:
                self._depth_3Dpoints = self.depth_to_3dpoints()
                self._pc_like = PointCloudLike(depth, {
                    "fx": self.fx_scaled,
                    "fy": self.fy_scaled,
                    "cx": self.cx_scaled,
                    "cy": self.cy_scaled
                })
            else:
                self._pc_like = None
                self._depth_3Dpoints = None

    def retrieve_image(self, mat, view=None):
        """
        Populate mat with current BGR image similar to ZED sl.Mat.get_data()
        """
        self._update_frames()
        with self._frame_lock:
            if self._last_color is None:
                return False
            # provide get_data method
            def get_data():
                return self._last_color
            mat.get_data = get_data
        return True

    def retrieve_measure(self, point_cloud_mat, measure_type=None):
        """
        Attach ZED-like point-cloud API to point_cloud_mat:
            - get_width(), get_height(), get_value(u,v), get_data()
        """
        self._update_frames()
        with self._frame_lock:
            if self._last_depth is None or self._pc_like is None:
                return False

            depth = self._last_depth
            color = self._last_color
            pc_like = self._pc_like

            h, w = depth.shape[:2]

            def get_width():
                return w

            def get_height():
                return h

            def get_data():
                # return raw depth array for convenience
                return depth

            def get_value(u, v):
                # Mirror ZED behavior: return (err_code, (X,Y,Z, rgba_packed))
                xyz = pc_like.get_xyz(u, v) if pc_like is not None else None
                if xyz is None:
                    return False, (np.nan, np.nan, np.nan, 0.0)
                X, Y, Z = xyz  # meters
                # pack color as 0xRRGGBBAA in float-compatible int
                if color is not None and 0 <= v < color.shape[0] and 0 <= u < color.shape[1]:
                    b, g, r = map(int, color[v, u].tolist())
                else:
                    r, g, b = 0, 0, 0
                rgba = (r << 24) | (g << 16) | (b << 8) | 255
                return True, (float(X*1000.0), float(Y*1000.0), float(Z*1000.0), float(rgba))

            point_cloud_mat.get_width = get_width
            point_cloud_mat.get_height = get_height
            point_cloud_mat.get_data = get_data
            point_cloud_mat.get_value = get_value

        return True

    def draw_gaze(self, frame, gaze_position, size=12, color=(0,255,255), thickness=2):
        if gaze_position is None:
            return
        try:
            x, y = map(int, gaze_position)
        except (ValueError, TypeError):
            return
        cv2.line(frame, (x-size,y), (x+size,y), color, thickness)
        cv2.line(frame, (x,y-size), (x,y+size), color, thickness)

    def detect_objects(self, frame, prediction_conf_threshold=0.5):
        results = self.model.predict(source=frame, conf=prediction_conf_threshold, verbose=False)[0]

        detections = []
        for result in results:
            # Parse detection results
            boxes = result.boxes  # Get bounding boxes
            for i, box in enumerate(boxes):
                class_id = int(box.cls[0])  # Class index
                prediction_confidence = float(box.conf[0])  # Confidence
                if prediction_confidence > prediction_conf_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get coordinates of bounding box
                    mask = result.masks.data[i].cpu().numpy() if result.masks is not None else None
                    #print(f"mask initial: {mask}")
                    detections.append({
                        "class_id": class_id,
                        "confidence": prediction_confidence,
                        "box": [x1, y1, x2 - x1, y2 - y1],  # Format as [x, y, w, h]
                        "mask": mask
                    })
        return detections

    def draw_bounding_boxes(self, frame, detections, classes, labels=None, confidences=None):
        for detection in detections:
            x, y, w, h = detection["box"]
            class_id = detection["class_id"]
            confidence = detection["confidence"]
            label = labels[class_id] if labels else f"{classes[class_id]}"
            display_confidence = confidences[class_id] if confidences and class_id in confidences else confidence
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f"{label}: {display_confidence:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    def draw_goal_boxes(self, frame, object_boxes, object_goals):
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

    def get_rgb_frame(self):
        """
        Retourne la dernière image BGR (numpy array) ou None.
        Met à jour les frames si nécessaire.
        """
        self._update_frames()
        with self._frame_lock:
            if self._last_color is None:
                return None
            return self._last_color.copy()

    def get_depth_frame(self):
        """
        Retourne la dernière depth map (uint16 numpy) ou None.
        """
        self._update_frames()
        with self._frame_lock:
            if self._last_depth is None:
                return None
            return self._last_depth.copy()


    def get_3Dpoints(self):
        """
        Retourne les points 3D de la dernière image depth.
        """
        self._update_frames()
        with self._frame_lock:
            return self._depth_3Dpoints if hasattr(self, '_depth_3Dpoints') else None
        
    def depth_to_xyz(self, u, v):
        if (
            u < 0 or v < 0 or
            v >= self.depth.shape[0] or
            u >= self.depth.shape[1]
        ):
            return None

        Z = self.depth[v, u]
        if Z <= 0:
            return None

        fx_scaled, fy_scaled = self.intrinsics[4], self.intrinsics[5]
        cx_scaled, cy_scaled = self.intrinsics[6], self.intrinsics[7]

        X = (u - cx_scaled) * Z / fx_scaled
        Y = (v - cy_scaled) * Z / fy_scaled

        return X, Y, Z

    def depth_to_3dpoints(self):
        H, W = self.depth.shape

        fx = self.intrinsics[4]
        fy = self.intrinsics[5]
        cx = self.intrinsics[6]
        cy = self.intrinsics[7]

        # grille pixel
        xs, ys = np.meshgrid(
            np.arange(W),
            np.arange(H)
        )

        Z = self.depth.astype(np.float32)
        mask = Z > 0

        X = (xs - cx) * Z / fx
        Y = (ys - cy) * Z / fy

        depth_3dpoints = np.zeros((H, W, 3), dtype=np.float32)
        depth_3dpoints[..., 0] = X
        depth_3dpoints[..., 1] = Y
        depth_3dpoints[..., 2] = Z

        depth_3dpoints[~mask] = 0.0

        return depth_3dpoints

        
    def get_frames(self, min_depth=20, max_depth=10000):
        """
        Instance method version of the previous get_frames function.
        Returns: (rgb_bgr, depth_uint16, pc_like, depth_3Dpoints)
        """
        frames = self.pipeline.wait_for_frames(100)
        if frames is None:
            return None, None

        # Align depth to color
        if self.align_filter:
            frames = self.align_filter.process(frames)
        if frames is None:
            return None, None

        frames = frames.as_frame_set()

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if depth_frame is None or color_frame is None:
            return None, None

        # --- RGB ---
        rgb = frame_to_bgr_image(color_frame)

        # --- Depth ---
        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
        depth_data = depth_data.reshape((depth_frame.get_height(), depth_frame.get_width()))
        depth_data = depth_data.astype(np.float32) * depth_frame.get_depth_scale()
        depth_data = np.where((depth_data > min_depth) & (depth_data < max_depth), depth_data, 0)

        # --- Filtre temporel ---
        if self.temporal_filter:
            depth_data = self.temporal_filter.process(depth_data)
        depth = depth_data.astype(np.uint16)

        if self.temporal_filter:
            depth = self.temporal_filter.process(depth)

        depth = depth.astype(np.uint16)
        depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        self.depth = depth
        self.rgb = rgb

        return rgb, depth
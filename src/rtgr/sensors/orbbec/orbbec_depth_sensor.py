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
from rtgr.sensors.depth_sensor_interface import DepthSensor
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


class OrbbecDepthSensor(DepthSensor):
    def __init__(self):
        self.pipeline = None
        self.running = False
        self._rgb = None
        self._depth = None

    def start(self):
        self.pipeline, self.temporal_filter, self.align_filter, self.intrinsics = init_orbbec()
        self.running = True
        self.fx = self.intrinsics[0]
        self.fy = self.intrinsics[1]
        self.cx = self.intrinsics[2]
        self.cy = self.intrinsics[3]
        self.fx_scaled = self.intrinsics[4]
        self.fy_scaled = self.intrinsics[5]
        self.cx_scaled = self.intrinsics[6]
        self.cy_scaled = self.intrinsics[7]
        self._last_rgb = None
        self._last_depth = None
        self._last_3d = None
        self._frame_lock = threading.Lock()

    def get_frames(self, min_depth=20, max_depth=10000):
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

        if self.temporal_filter:
            depth_data = self.temporal_filter.process(depth_data)
            # depth_data = depth_data.astype(np.uint16)
            depth_resized= cv2.resize(depth_data, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            self.rgb = rgb
            H, W = depth_resized.shape[:2]
            xs, ys = np.meshgrid(
            np.arange(W),
            np.arange(H)
            )

            Z = depth_resized.astype(np.float32) / 1000.0  # in meters
            X = (xs - self.cx_scaled) * Z / self.fx_scaled
            Y = (ys - self.cy_scaled) * Z / self.fy_scaled

            depth = np.zeros((H, W, 3), dtype=np.float32)
            depth[..., 0] = X
            depth[..., 1] = Y
            depth[..., 2] = Z
            self._last_rgb = rgb
            self._last_depth = depth

        return rgb, depth
    
    def get_depth(self):
        return super().get_depth()
        
    def get_rgb(self):
        return super().get_rgb()
    
    def get_intrinsics(self):
        return super().get_intrinsics()
    
    def stop(self):
        return super().stop()
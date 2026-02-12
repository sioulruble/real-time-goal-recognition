from abc import ABC, abstractmethod
import numpy as np

class DepthSensor(ABC):
    """
    minimal interface for 3d sensor
    """
    def __init__(self):
        self._last_depth = None
        self._last_rgb = None
        self.intrinsics = None


    @abstractmethod
    def start(self) -> None:
        """Open hardware / start streams"""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Release hardware / stop streams"""
        pass

    @abstractmethod
    def get_rgb(self) -> np.ndarray | None:
        return self._last_rgb

    @abstractmethod
    def get_depth(self) -> np.ndarray | None:
        return self._last_depth


    @abstractmethod
    def get_intrinsics(self) -> dict:
        """
        fx, fy, cx, cy (image space)
        """
        return self.intrinsics

    @abstractmethod
    def get_frames(self):
        """
        Return rgb and 3d depth information
        """
        pass
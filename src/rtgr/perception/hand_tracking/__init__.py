# hand_tracking/__init__.py

from .hand_tracker import HandTracker
from .hand_mediapipe import MediaPipeHand
from .hand_ros_tracker import ROSHandTracker

def create_hand_tracker(tracker_type="mediapipe", **kwargs):
    """Factory pour créer le bon tracker"""
    trackers = {
        "mediapipe": MediaPipeHand,
        "ros": ROSHandTracker,
    }
    if tracker_type not in trackers:
        raise ValueError(f"Unknown tracker: {tracker_type}")
    return trackers[tracker_type](**kwargs)

__all__ = ["HandTracker", "MediaPipeHand", "ROSHandTracker", "create_hand_tracker"]
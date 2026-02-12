from abc import ABC, abstractmethod

class HandTracker(ABC):
    def __init__(self):
        self._last_position_3d = None

    def get_position_3d(self):
        return self._last_position_3d

    @abstractmethod
    def update(self, image, points3d):
        """
        Met à jour la position 3D de la main
        image : image RGB ou BGR
        points3d : carte 3D alignée (H, W, 3)
        """
        pass

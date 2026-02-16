import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import cv2
from rtgr.inference.hmm.hmm import  init_hmm
from rtgr.config.system import init_system
from rtgr.perception.object_detection import make_yolo_detector
from rtgr.inference.observations import update_goals_from_proximity, get_observations, filter_observations, save_last_distance
from rtgr.visualization.overlays import visualize

def main():
    model, camera, hand, vlm_handler = init_system(depth_sensor= "orbbec", hand_tracker="mediapipe")
    yolo_detect = make_yolo_detector(model, period_sec=0.25)
    image, depth = camera.get_frames()
    hmm, goals, mapping, hand_3D = init_hmm(image, depth, model)
    last_distance = {}

    while True:
        image, depth = camera.get_frames()
        hand.update_position(image, depth)
        hand_3D = hand.get_position3d()
        objects_2D, objects_3D = yolo_detect(image, depth)
        if objects_2D is None :
            visualize(image, hmm.goal_beliefs, {})
            continue

        goals =  vlm_handler.update(
            image=image,
            objects_2D=objects_2D            
        )
        # fuse VLM goal hypotheses with proximity-based priors
        update_goals_from_proximity(goals, objects_3D, hand_3D, hmm)

        observations = get_observations(objects_3D, hand_3D, last_distance , mapping)     
        if observations:
            obs = filter_observations(observations, goals, mapping)
            # update goal beliefs using temporally filtered observations
            hmm.assisted_teleop(obs)
        last_distance= save_last_distance(objects_3D, hand_3D)

        visualize(image,  hmm.goal_beliefs, objects_2D)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    camera.stop()

if __name__ == "__main__" :
    main()

from sentence_transformers import SentenceTransformer, util
from rtgr.inference.vlm.similaritymodel import SentenceSimilarityModel
from rtgr.perception.hand_tracking import create_hand_tracker
from ultralytics import YOLO
from rtgr.config.paths import  YOLO_MODEL, VLM_MODEL, SENTENCE_MODEL
from rtgr.sensors.orbbec.orbbec_depth_sensor import *
from rtgr.config.constants import ACTIONS
from rtgr.inference.vlm.vlm_handler import *
from rtgr.inference.vlm.similaritymodel import process_multiline_caption


def init_system(depth_sensor, hand_tracker, yolo_model=YOLO_MODEL, vlm_model= VLM_MODEL, sentence_model = SENTENCE_MODEL, actions = ACTIONS):
    model = YOLO(yolo_model)
    camera = None
    if depth_sensor == "orbbec":
        camera = OrbbecDepthSensor()
        camera.start()

    hand = create_hand_tracker(hand_tracker)

    smodel = SentenceTransformer(sentence_model)
    smModel = SentenceSimilarityModel(smodel)

    vlm_handler = VLMHandler(
        processorLL=VLMProcessor(),
        VLM_model_name= vlm_model,
        threaded_VLM_wrapper=threaded_VLM_wrapper,
        process_multiline_caption=process_multiline_caption,
        smModel=smModel,
        list_of_actions=actions,
        period_sec=3.0
    )

    return model, camera, hand, vlm_handler

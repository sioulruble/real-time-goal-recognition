from PIL import Image
import numpy as np
import base64
import io
import requests
import cv2
import json
import time
import threading
from datetime import datetime
from rtgr.config.constants import TIMING_STATS

class VLMProcessor:
    def __init__(self, ollama_url="http://localhost:11434/api/chat"):
        self.ollama_url = ollama_url

    def convert_image_for_VLM(self, frame):
        """
        Convertit une image numpy en format PIL compatible avec LLaVA.
        """
        if frame is None:
            raise ValueError("❌  Received image is None.")
        if not isinstance(frame, np.ndarray):
            raise ValueError("The input must be a numpy array.")
        # Resize
        frame_resized = cv2.resize(frame, (224, 224))

        return Image.fromarray(frame_resized)

    def encode_image_base64(self, image_pil):
        """
        Encode une image PIL en base64 pour Ollama.
        """
        buffered = io.BytesIO()
        image_pil.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def VLM_process_func(self, model_name, shared_caption, image, new_dict_of_objects, list_of_actions , pour_object= None, timing_stats=TIMING_STATS):
        try:
            start_vlm = time.time()
            shared_caption.value = self.generate_description_with_VLM(model_name, image, new_dict_of_objects, list_of_actions)
            if timing_stats is not None:
                timing_stats["VLM"] = time.time() - start_vlm
                print(timing_stats["VLM"])
            return shared_caption.value
        except Exception as e:
            print(f"Erreur dans le processus LLaVA : {e}")
    
    def generate_description_with_VLM(self, model_name, frame, new_dict_of_objects, list_of_actions):
        if not new_dict_of_objects:
            return "[]"  

        image_pil = frame  
        image_b64 = self.encode_image_base64(image_pil)

        # prompt creation
        objects = list(new_dict_of_objects.keys())
        object_list = ", ".join(objects)
        messages = [
            {
                "role": "system",
                "content":
                "You are a visual reasoning assistant specialized in robot task planning.\n\n"
                "You will be given:\n"
                "- An image representing the scene.\n"
                "- A list of visible objects.\n"
                "- A list of possible actions, formatted as function calls.\n"
                "Your job is to return a Python list of all logically feasible actions based on the image, visible objects.\n\n"
                "Each action must follow the function format: grab(object1), push(object1), pull(object1), press(object1), place(object1).\n"
                "Only return actions that are possible in the scene. For example, press(button) is not allowed if the object is not a button.\n\n"
                "❗ Output ONLY a Python list. No explanations. No natural language. No quotes around object names."
            },
            {
                "role": "user",
                "content":
                "Visible objects: [bowl, cup, bottle, ball]\n"
                "Possible actions: [grab(object1), push(object1), pull(object1)]\n"
            },
            {
                "role": "assistant",
                "content":
                "[grab(bowl), grab(cup), grab(bottle), grab(ball), push(bowl), push(cup), push(bottle), push(ball), pull(bowl), pull(cup), pull(bottle), pull(ball)]"
            },
            {
                "role": "user",
                "content":
                "Visible objects: [button, drawer, box]\n"
                "Possible actions: [grab(object1), push(object1), pull(object1), press(object1)]\n"
            },
            {
                "role": "assistant",
                "content":
                "[grab(button), grab(drawer), grab(box), push(button), push(drawer), push(box), pull(drawer), pull(box), press(button)]"
            },
            {
                "role": "user",
                "content":
                f"Visible objects: [{', '.join(new_dict_of_objects.keys())}]\n"
                f"Possible actions: {list_of_actions}\n"
            }

        ]

        payload = {
            "model": model_name,
            "messages": messages,
            "images": [image_b64],
            "stream": False
        }

        response = requests.post(self.ollama_url, json=payload)

        if response.status_code == 200:
            try:
                result = response.json()
                message = result.get("message", {}).get("content", "").strip()
                return message
            except json.JSONDecodeError:
                print("❌ Erreur de décodage JSON. Contenu brut de la réponse :")
                print(response.text)
                return "[]"
        else:
            error_msg = f"Error: {response.status_code} - {response.text}"
            print(error_msg)
            return "[]"
            
    def generate_description_with_VLM_pour(self, model_name, frame, new_dict_of_objects, list_of_actions):
        if not new_dict_of_objects:
            return "[]"  

        image_pil = frame  
        image_b64 = self.encode_image_base64(image_pil)

        # prompt creation
        objects = list(new_dict_of_objects.keys())
        object_list = ", ".join(objects)
        messages = [
            {
                "role": "system",
                "content":
                "You are a visual reasoning assistant specialized in robot task planning.\n\n"
                "You will be given:\n"
                "- An image representing the scene.\n"
                "- A list of visible objects.\n"
                "- A list of possible actions, formatted as function calls.\n"
                "Your job is to return a Python list of all logically feasible actions based on the image, visible objects.\n\n"
                "Each action must follow the function format: grab(object1), push(object1), pull(object1), press(object1), place(object1), pour(object1).\n"
                "Only return actions that are possible in the scene. For example, press(button) is not allowed if the object is not a button.\n\n"
                "❗ Output ONLY a Python list. No explanations. No natural language. No quotes around object names."
            },
            {
                "role": "user",
                "content":
                "Visible objects: [bowl, cup, bottle, ball]\n"
                "Possible actions: [grab(object1), push(object1), pull(object1)]\n"
            },
            {
                "role": "assistant",
                "content":
                "[grab(bowl), grab(cup), grab(bottle), grab(ball), push(bowl), push(cup), push(bottle), push(ball), pull(bowl), pull(cup), pull(bottle), pull(ball)]"
            },
            {
                "role": "user",
                "content":
                "Visible objects: [button, drawer, box]\n"
                "Possible actions: [grab(object1), push(object1), pull(object1), press(object1)]\n"
            },
            {
                "role": "assistant",
                "content":
                "[grab(button), grab(drawer), grab(box), push(button), push(drawer), push(box), pull(drawer), pull(box), press(button)]"
            },
            {
                "role": "user",
                "content":
                f"Visible objects: [{', '.join(new_dict_of_objects.keys())}]\n"
                f"Possible actions: {list_of_actions}\n"
            }

        ]

        payload = {
            "model": model_name,
            "messages": messages,
            "images": [image_b64],
            "stream": False
        }

        response = requests.post(self.ollama_url, json=payload)

        if response.status_code == 200:
            try:
                result = response.json()
                message = result.get("message", {}).get("content", "").strip()

                return message
            except json.JSONDecodeError:
                print("❌ Erreur de décodage JSON. Contenu brut de la réponse :")
                print(response.text)
                return "[]"
        else:
            error_msg = f"Error: {response.status_code} - {response.text}"
            print(error_msg)
            return "[]"

def threaded_VLM_wrapper(processorLL, model_name, caption, frame, objects, timing, result_container, list_of_actions):
    result = processorLL.VLM_process_func(model_name, caption, frame, objects, list_of_actions, timing)
    print("VLM output:", result)
    result_container["result"] = result
    return result_container





class VLMHandler:
    def __init__(
        self,
        processorLL,
        VLM_model_name,
        threaded_VLM_wrapper,
        process_multiline_caption,
        smModel,
        list_of_actions,
        period_sec=3.0
    ):
        """
        Gestionnaire du VLM (thread + timing + résultats)
        """
        self.processorLL = processorLL
        self.VLM_model_name = VLM_model_name
        self.threaded_VLM_wrapper = threaded_VLM_wrapper
        self.process_multiline_caption = process_multiline_caption
        self.smModel = smModel
        self.list_of_actions = list_of_actions

        self.period_sec = period_sec

        # État interne
        self.last_vlm_time = datetime.min
        self.vlm_thread = None
        self.vlm_result_container = {}

        # Caption partagée (mutée par le thread)
        self.shared_caption = type('', (), {})()
        self.shared_caption.value = "No description yet"

        self.list_of_goals = []

    # -----------------------------------------------------
    # Lancement du thread VLM si nécessaire
    # -----------------------------------------------------
    def maybe_start_vlm(
        self,
        image,
        objects_2D,
        timing_stats
    ):
        """
        Lance le thread VLM si le délai est dépassé
        """
        if (datetime.now() - self.last_vlm_time).total_seconds() <= self.period_sec:
            return

        if self.vlm_thread is not None and self.vlm_thread.is_alive():
            return

        print("⏳ Starting VLM processing thread...")
        self.last_vlm_time = datetime.now()

        try:
            vlmframe = self.processorLL.convert_image_for_VLM(image)
            self.vlm_result_container = {}
            dict_of_objects_at_vlm_time = dict(objects_2D)

            self.vlm_thread = threading.Thread(
                target=self.threaded_VLM_wrapper,
                args=(
                    self.processorLL,
                    self.VLM_model_name,
                    self.shared_caption,
                    vlmframe,
                    dict_of_objects_at_vlm_time,
                    timing_stats,
                    self.vlm_result_container,
                    self.list_of_actions,
                ),
                daemon=True
            )
            self.vlm_thread.start()

        except Exception as e:
            print(f"❌ Erreur pendant le traitement VLM : {e}")

    # -----------------------------------------------------
    # Récupération des résultats si le thread est fini
    # -----------------------------------------------------
    def maybe_collect_results(self, objects_2D):
        """
        Récupère les résultats si le thread est terminé
        """
        if self.vlm_thread is None:
            return self.list_of_goals

        if self.vlm_thread.is_alive():
            return self.list_of_goals

        print("✅ VLM thread finished, retrieving results...")
        result = self.vlm_result_container.get("result", None)

        if result:
            if objects_2D:
                self.list_of_goals = self.process_multiline_caption(
                    result,
                    self.smModel,
                    self.list_of_actions,
                    list(objects_2D.keys())
                )
            else:
                print("No object detected — skipping similarity_model call.")
                self.list_of_goals = []

        self.vlm_thread = None
        return self.list_of_goals

    # -----------------------------------------------------
    # Méthode pratique : à appeler dans la loop principale
    # -----------------------------------------------------
    def update(
        self,
        image,
        objects_2D,
        timing_stats = TIMING_STATS
    ):
        """
        Appel unique par frame
        """
        self.maybe_start_vlm(image, objects_2D, timing_stats)
        return self.maybe_collect_results(objects_2D)

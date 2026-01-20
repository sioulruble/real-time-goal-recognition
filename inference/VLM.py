from PIL import Image
import numpy as np
import base64
import io
import requests
import cv2
import json
import time

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

    def VLM_process_func(self, model_name, shared_caption, image, new_dict_of_objects, list_of_actions , pour_object= None, timing_stats=None):
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

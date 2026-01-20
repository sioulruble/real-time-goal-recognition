import requests
import json
import logging
import re
import csv
import json
import ast
import os

def ensure_csv(csv_path):
    print("Step 1")
    print(f"Le fichier existe ? {os.path.abspath(csv_path)}")   
    if not os.path.exists(csv_path):
        print("Step 2")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["object", "type"])

def merge_goals(primary, extra):
    seen = set()
    merged = []
    for g in primary + extra:
        if g not in seen:
            merged.append(g)
            seen.add(g)
    return merged

def append_unique_row(csv_path, obj, typ):
    key = (obj.strip().lower(), typ.strip().lower())
    existing = set()
    print("Step 3")
    if os.path.exists(csv_path):
        print("Step 4")
        with open(csv_path, newline="", encoding="utf-8") as f:
            print("Step 5")
            reader = csv.DictReader(f)
            for r in reader:
                existing.add((r["object"].strip().lower(), r["type"].strip().lower()))
                print("Step 6")
    if key not in existing:
        print("Step 7")
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            print("Step 8")
            writer = csv.writer(f)
            writer.writerow([obj, typ])
        print(f"add to the CSV: {obj}, {typ}")
    else:
        print(f"already in the CSV: {obj}, {typ}")

def create_all_feasible_goals(feasible_objects, list_of_actions):
    goals = []
    for action in list_of_actions:
        for obj in feasible_objects:
            goal = action.replace("object1", obj)
            goals.append(goal)
    print(f"Feasible goals created: {len(goals)}")
    print(f"Feasible goals: {goals}")
    return goals

def create_single_pour_goals(csv_goal_path):
    pourable_objects = []
    if not os.path.exists(csv_goal_path):
        print(f":warning: Fichier {csv_goal_path} introuvable.")
        return []
    with open(csv_goal_path, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            obj = row.get("object", "").strip()
            typ = row.get("type", "").strip().lower()
            if typ == "pourable" and obj:
                pourable_objects.append(obj)

    goals = [f"pour({obj})" for obj in pourable_objects]
    print(f"Objets pourable trouvés: {pourable_objects}")
    print(f"Pour goals générés: {goals}")
    return goals

class LLMClassification:
    def __init__(self, model, host):
        self.model = model
        self.host = host.rstrip('/') + "/api/chat"
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)

    def classify_container(self, objects):
        """
        Classify objects as Containers vs Non-Containers using the LLM model.
        :param objects: List of objects to classify.
        :return: Classification results as a string (two lists) like the model returns.
        """
        if not objects:
            self.logger.warning("No objects provided for classisfication.")
            return {}
        messages =[
            {
                "role": "system",
                "content":"""Your job is to return exactly two clean, alphabetical lists:
                Containers
                Non-Containers
                Do not explain anything. Just return the two clean lists.
                I have a list of objects detected by a vision model. Each object is labeled with a name.
                I want to split this list into two categories:
                Containers: Objects that are primarily designed or commonly used to hold, store, serve, or carry other items or substances. This includes:
                - Open or closed vessels and enclosures (glass, cup, mug, bowl, bottle, jar, box, can, bin, basket, bucket, bag, backpack, pouch, purse, wallet, suitcase, crate, chest, drawer, cabinet, envelope, case, toolbox, pot, pan, tube, tank, trash can).
                - Flat or open objects that can hold or carry things on them (plates, trays, dishes, platters).
                Non-Containers: Objects that are not intended to hold or carry items. This includes solid items (like chairs, tables, pens, books), living beings, infrastructure elements, locations, and abstract concepts.
                Assume the objects are real-world, full-sized versions, unless stated otherwise.
                Examples:
                    - A glass is a container because you can put liquid inside it.
                    - A backpack is a container because you can store objects inside it.
                    - A plate is a container because you can place food on it.
                    
                    - A chair is not a container because it is not designed to hold items.
                    - A pen is not a container because it does not carry other objects.
                Here is the full list of object classes (from an object detection model like COCO): {objects}
                """
            },
            {
                "role": "user",
                "content": """Example input: glass, chair, backpack, book, plate"""
            },
            {
                "role": "assistant",
                "content": """Containers: [backpack,glass,plate]
                Non-Containers: [book,chair]"""
            },
            {
                "role": "user",
                "content": f"Here is the list of objects: {objects}"
            }

        ]

        payload = {
            "model": self.model,
            "stream": False,
            "messages": messages
        }

        try:
            response = requests.post(self.host, json=payload)
            response.raise_for_status()
            result =response.json()
            content =result.get("message", {}).get("content", "").strip()
        except Exception as e:
            self.logger.error(f"Error querying model: {e}")
            return {}
        
        self.logger.debug(f"Raw modeln output: {content}")

        return content

    def classify_contenable(self, objects):
        """
        Classify objects as Contenable vs Non-Contenable using the LLM model.
        """
        if not objects:
            self.logger.warning("No objects provided for classification.")
            return {}
        messages = [
            {
                "role": "system",
                "content": """Your job is to return exactly two clean, alphabetical lists:
                Contenable
                Non-Contenable
                Do not explain anything. Just return the two clean lists.
                Definitions:
                - Contenable: Objects that can realistically be placed inside another object (like a bottle inside a bag, water in a glass, fruit in a bowl, book in a backpack, etc.) AND can be comfortably grasped and carried with ONE HAND by an average adult.
                - Non-Contenable: Objects that cannot reasonably be put inside another (like a table, chair, couch, very large/fixed objects, or anything requiring two hands to carry).
                Assume the objects are real-world, full-sized versions, unless stated otherwise.
                Examples:
                - A bottle is contenable because it can go inside a bag and is carried with one hand.
                - A glass is contenable because it can be placed inside a cupboard or a box, and held with one hand.
                - A banana is contenable because it can be put in a bowl and held with one hand.
                - A chair is non-contenable because you cannot put it inside another object in normal use, and it usually requires two hands to move.
                - A suitcase is non-contenable because although it can hold items, it cannot itself be easily placed inside another container with one hand.
                - A laptop is non-contenable if considered full size, because it usually requires a bag or case and is not trivially "one-hand contenable"."""
            },
            {
                "role": "user",
                "content": """Example input: bottle, chair, banana, couch"""
            },
            {
                "role": "assistant",
                "content": """Contenable: [banana,bottle]
                Non-Contenable: [chair,couch]"""
            },
            {
                "role": "user",
                "content": f"Here is the list of objects: {objects}"
            }
        ]
        payload = {
            "model": self.model,
            "stream": False,
            "messages": messages
        }
        try:
            response = requests.post(self.host, json=payload)
            response.raise_for_status()
            result = response.json()
            content = result.get("message", {}).get("content", "").strip()
        except Exception as e:
            self.logger.error(f"Error querying model: {e}")
            return {}
        self.logger.debug(f"Raw model output: {content}")
        return content
    
    def classify_twohands(self, objects):
        """
        Classify objects as requiring two hands vs not requiring two hands.
        """
        if not objects:
            self.logger.warning("No objects provided for classification.")
            return {}
        messages = [
            {
                "role": "system",
                "content": """Your job is to return exactly two clean, alphabetical lists:
                Two-Handed
                Not-Two-Handed
                Do not explain anything. Just return the two lists.
                Assumptions:
                - Consider full-size, real-world household/office items (not miniatures).
                - Judge what an average adult would use for a safe, controlled lift/carry/manipulation in typical use.
                - If uncertain or borderline, default to Two-Handed.
                Definitions:
                - Two-Handed: Items that typically require BOTH HANDS to safely lift, carry, or manipulate due to size, weight, bulk, awkward shape, fragility, or stability needs.
                Examples (Two-Handed): chair, couch/sofa, dining table/table, suitcase, snowboard, skateboard, bench, large potted plant, monitor/TV, microwave, oven, refrigerator, desktop computer/tower, laptop, keyboard (full-size), printer, box/crate when full, heavy kitchen pots/pans when full.
                - Not-Two-Handed: Items that can be comfortably and safely handled with ONE HAND in normal use.
                Examples (Not-Two-Handed): bottle, glass, cup, banana, apple, spoon, fork, knife, book (paperback/hardback), phone, remote, mouse, scissors, wallet, keys.
                Edge rules:
                - Electronics that are thin/fragile or wide (e.g., laptop, monitor, keyboard) → classify as Two-Handed for safe handling.
                - Furniture-like items and large containers (chair, couch, table, suitcase) → Two-Handed.
                - Small utensils, small food items, small tools → Not-Two-Handed.
                - Sports gear: long/board-like or bulky (snowboard, skateboard) → Two-Handed.
                - When in doubt → Two-Handed.
                Output format (exactly two lines):
                Two-Handed: [item1,item2,...]
                Not-Two-Handed: [item3,item4,...]"""
            },
            {
                "role": "user",
                "content": """Example input: chair, bottle, banana, suitcase, laptop, keyboard, book"""
            },
            {
                "role": "assistant",
                "content": """Two-Handed: [chair,keyboard,laptop,suitcase]
                Not-Two-Handed: [banana,book,bottle]"""
            },
            {
                "role": "user",
                "content": f"Here is the list of objects: {objects}"
            }
        ]
        payload = {
            "model": self.model,
            "stream": False,
            "messages": messages
        }
        try:
            response = requests.post(self.host, json=payload)
            response.raise_for_status()
            result = response.json()
            content = result.get("message", {}).get("content", "").strip()
        except Exception as e:
            self.logger.error(f"Error querying model: {e}")
            return {}
        self.logger.debug(f"Raw model output: {content}")
        return content

    def classify_pourable(self, objects):
        """
        Classify objects as Pourable vs Non-Pourable.
        """
        if not objects:
            self.logger.warning("No objects provided for classification.")
            return {}
        messages = [
            {
                "role": "system",
                "content": """Return exactly two clean, alphabetical lists:
                Pourable
                Non-Pourable
                Do not explain anything. Just return the two lists.
                Definitions:
                - Pourable: Vessels commonly used to HOLD and/or POUR liquids (or small pourable solids): bottle, cup, mug, glass, bowl, jar, kettle, teapot, pitcher/jug, thermos, can, tube (liquid), carton, measuring cup, wine glass, vase (if typical household vase), pot/pan with lip, dispenser, soap bottle, squeeze bottle.
                - Non-Pourable: Solid items, tools, electronics, furniture, foods that are not vessels, flat items, living beings, fixed infrastructure, etc.
                Edge rules:
                - Plates/trays are NOT pourable (surface, no lip).
                - Laptops/phones/remotes/keyboards → Non-Pourable.
                - Food (banana, apple, pizza, etc.) → Non-Pourable.
                - If uncertain → Non-Pourable.
                Output format (two lines exactly):
                Pourable: [item1,item2,...]
                Non-Pourable: [item3,item4,...]"""
            },
            {"role": "user", "content": "Example input: bottle, bowl, plate, laptop, cup, vase"},
            {"role": "assistant", "content": "Pourable: [bottle,bowl,cup,vase]\nNon-Pourable: [laptop,plate]"},
            {"role": "user", "content": f"Here is the list of objects: {objects}"}
        ]
        payload = {"model": self.model, "stream": False, "messages": messages}
        try:
            response = requests.post(self.host, json=payload)
            response.raise_for_status()
            result = response.json()
            content = result.get("message", {}).get("content", "").strip()
        except Exception as e:
            self.logger.error(f"Error querying model: {e}")
            return {}
        self.logger.debug(f"Raw model output: {content}")
        return content


    def classify(self, objects):
        """
        Classify objects using the LLM model.
        :param objects: List of objects to classify.
        :return: Classification results as a dictionary.
        """
        if not objects:
            self.logger.warning("No objects provided for classisfication.")
            return {}
        messages =[
            {
                "role": "system",
                "content":"""Your job is to return exactly two clean, alphabetical lists:"
                Feasible Objects
                Unfeasible Objects
                Do not explain anything. Just return the two clean lists.
                I have a list of objects detected by a vision model. Each object is labeled with a name. I want to split this list into two categories:
                Feasible: Objects that a humanoid robot with human-level dexterity and size can realistically interact with during teleoperation. That includes grabbing, pushing, placing, or pressing the object — like a human would.
                Unfeasible: Objects that are physically impossible or illogical for a humanoid robot to interact with (e.g. too large, fixed in place, living beings like animals or humans, infrastructure elements, or abstract/non-graspable things).
                Assume the objects are real-world, full-sized versions, unless stated otherwise. Ignore non-physical interactions (like speaking or observing).
                Also, assume that any object graspable by a human (like a banana, teddy bear) is also graspable by the humanoid robot, since it has human-level hands and motor control.
                Examples:
                    -A chair is feasible because it can be grabbed or moved by a humanoid robot.
                    -A car is unfeasible because it is too large and heavy for a humanoid robot to manipulate directly.
                Here is the full list of object classes (from an object detection model like COCO):{objects}
                """
            },
            {
                "role": "user",
                "content": """Example input: chair, car"""
            },
            {
                "role": "assistant",
                "content": """Feasible Objects: [chair]
                Unfeasible Objects: [car]"""
            },
            {
                "role": "user",
                "content": f"Here is the list of objects: {objects}"
            }

        ]

        payload = {
            "model": self.model,
            "stream": False,
            "messages": messages
        }

        try:
            response = requests.post(self.host, json=payload)
            response.raise_for_status()
            result =response.json()
            content =result.get("message", {}).get("content", "").strip()
        except Exception as e:
            self.logger.error(f"Error querying model: {e}")
            return {}
        
        self.logger.debug(f"Raw modeln output: {content}")

        return content
    
    
    def give_time(self,list_of_goals):
        """
        Classify objects using the LLM model.
        :param objects: List of objects to classify.
        :return: Classification results as a dictionary.
        """
        if not list_of_goals:
            self.logger.warning("No objects provided")
            return {}
        messages =[
            {
                "role": "system",
                "content":"""Your task is to return the estimated execution time (in seconds) for each goal in the list I will provide. Each goal is formatted like: action(object), for example: grab(bottle).
                The time must reflect the realistic duration for a humanoid robot to execute that action via teleoperation for the specific object. Take into account the type, size, shape, and difficulty of the object when performing the action.
                Provide only a clean JSON dictionary mapping each goal to its estimated time in seconds. Do not explain anything. Do not describe assumptions.
                Example input: ['grab(bottle)', 'push(couch)']
                Expected output: {'grab(bottle)': value, 'push(couch)': value}"""
            },
            {
                "role": "user",
                "content": """Example input: ["grab(bottle)"]"""
            },
            {
                "role": "assistant",
                "content": """["grab(bottle)":2]"""
            },
            {
                "role": "user",
                "content": f"Here is the list of goals: {list_of_goals}"
            }

        ]

        payload = {
            "model": self.model,
            "stream": False,
            "messages": messages
        }

        try:
            response = requests.post(self.host, json=payload)
            response.raise_for_status()
            result =response.json()
            content =result.get("message", {}).get("content", "").strip()
        except Exception as e:
            self.logger.error(f"Error querying model: {e}")
            return {}
        
        self.logger.debug(f"Raw modeln output: {content}")

        return content
    
    def related_goals(self, list_of_goals, current_goal):
        if not list_of_goals:
            self.logger.warning("No goals provided")
            return {}
        messages=[
            {
                "role": "system",
                "content": """You are a context-aware robotic planning assistant.

                Your job is to give the most plausible next goals from a provided list that a human might perform immediately after the current goal — based on human-like intention, task continuity, physical plausibility, and scene consistency.

                You are given:
                1. A list of possible goals, each expressed as an atomic action on an object, in the format: action(object). Examples: grab(cup), place(cup), press(remote), etc.
                2. A single current goal that has just been completed.

                Your task is to choose the most likely next goals from the list, considering how humans act as part of short-term, coherent tasks (like serving a drink, setting a table, or operating a device).

                Important logical constraints:

                - Use only goals from the provided list. Do not invent new actions or new objects.
                - Do not repeat the current goal. Self-loops are not allowed.
                - Do not suggest place(X) unless grab(X) occurred earlier.
                - After place(X), do not immediately suggest grab(X) again. The object has just been placed, so the human is likely done with it for now.
                - However, it is logical to grab a different object after placing one. For example:
                - Valid: place(cup) followed by grab(bottle)
                - Invalid: place(cup) followed by grab(cup)

                Continuity and context rules:

                - Prefer next goals that are part of the same task or physical context.
                - Only suggest actions on objects that are **functionally or contextually linked** to the object in the current goal. For example:
                - cup ↔ bottle, spoon, plate
                - book ↔ pen, notebook
                - mouse ↔ keyboard, computer
                - Avoid switching to unrelated objects or scenes (e.g., cup → book) unless no related options are available.

                Example of invalid transition:
                - Current goal: place(cup)
                - Possible goals: [grab(book), grab(bottle)]
                - Suggested: {"grab(book)": 1.0}
                - This is invalid because "book" is unrelated to the recently placed object "cup".

                Example of valid transition:
                - Current goal: place(cup)
                - Possible goals: [grab(bottle), grab(book)]
                - Suggested: {"grab(bottle)": 1.0}

                Output instructions:

                Return a JSON dictionary with at most 3 or 4 next goals and their associated probabilities. All probabilities must sum to exactly 1.0.

                Example output:
                {
                "place(cup)": 0.6,
                "grab(bottle)": 0.4
                }
                "If the current action is push(X) or pull(X), and no directly related object Y exists in the list (e.g., part of same functional group as X), return {}. Do not try to force a next step. For example do not return 'grab(bottle)' unless bottle is clearly related with the object X.
                If no logical next goal exists, return an empty dictionary like this:
                {}
                Do not add any explanation, comment, or formatting outside the JSON.
                """
            },
            {
                "role": "user",
                "content": """Example input:  
                -Possible goals:["grab(cup)", "place(cup)", "grab(bottle)", "press(remote)"]
                - Current goal: "grab(cup)" """
            },
            {
                "role": "assistant",
                "content": """"
                {
                "place(cup)": 0.6,
                "grab(bottle)": 0.4
                }"""
            },
            {
                "role": "user",
                "content": f"Here is the list of possible goals: {list_of_goals} and here is the current goal: {current_goal}"
            }
        ]

        payload = {
            "model": self.model,
            "stream": False,
            "messages": messages
        }

        try:
            response = requests.post(self.host, json=payload)
            response.raise_for_status()
            result =response.json()
            content =result.get("message", {}).get("content", "").strip()
        except Exception as e:
            self.logger.error(f"Error querying model: {e}")
            return {}
        
        self.logger.debug(f"Raw modeln output: {content}")

        return content   


if __name__ == "__main__":
    #LLM_model="deepseek-r1:14B" 
    #LLM_model = "llama3:latest"

    LLM_model = "qwen3:latest"
    host_url="http://localhost:11434"
    llm_class= LLMClassification(LLM_model, host_url)
   
    list_of_actions = ["grab(object1)", "push(object1)", "place(object1)", "pull(object1)", "press(object1)"]
    
    object_list = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    list_of_goals = ['place(cup)', 'grab(bench)', 'grab(bear)', 'grab(backpack)', 'grab(umbrella)', 'grab(handbag)', 'grab(tie)', 'grab(suitcase)', 'grab(frisbee)', 'grab(skis)', 'grab(snowboard)', 'grab(sports ball)', 'grab(kite)', 'grab(baseball bat)', 'grab(baseball glove)', 'grab(skateboard)', 'grab(tennis racket)', 'grab(bottle)', 'grab(wine glass)', 'grab(cup)', 'grab(fork)', 'grab(knife)', 'grab(spoon)', 'grab(bowl)', 'grab(banana)', 'grab(apple)', 'grab(sandwich)', 'grab(orange)', 'grab(broccoli)', 'grab(carrot)', 'grab(hot dog)', 'grab(pizza)', 'grab(donut)', 'grab(cake)', 'grab(chair)', 'grab(couch)', 'grab(potted plant)', 'grab(dining table)', 'grab(laptop)', 'grab(mouse)', 'grab(remote)', 'grab(keyboard)', 'grab(cell phone)', 'grab(toaster)', 'grab(book)', 'grab(vase)', 'grab(scissors)', 'grab(teddy bear)', 'grab(hair drier)', 'grab(toothbrush)', 'push(bench)', 'push(bear)', 'push(backpack)', 'push(umbrella)', 'push(handbag)', 'push(tie)', 'push(suitcase)', 'push(frisbee)', 'push(skis)', 'push(snowboard)', 'push(sports ball)', 'push(kite)', 'push(baseball bat)', 'push(baseball glove)', 'push(skateboard)', 'push(tennis racket)', 'push(bottle)', 'push(wine glass)', 'push(cup)', 'push(fork)', 'push(knife)', 'push(spoon)', 'push(bowl)', 'push(banana)', 'push(apple)', 'push(sandwich)', 'push(orange)', 'push(broccoli)', 'push(carrot)', 'push(hot dog)', 'push(pizza)', 'push(donut)', 'push(cake)', 'push(chair)', 'push(couch)', 'push(potted plant)', 'push(dining table)', 'push(laptop)', 'push(mouse)', 'push(remote)', 'push(keyboard)', 'push(cell phone)', 'push(toaster)', 'push(book)', 'push(vase)', 'push(scissors)', 'push(teddy bear)', 'push(hair drier)', 'push(toothbrush)', 'place(bench)', 'place(bear)', 'place(backpack)', 'place(umbrella)', 'place(handbag)', 'place(tie)', 'place(suitcase)', 'place(frisbee)', 'place(skis)', 'place(snowboard)', 'place(sports ball)', 'place(kite)', 'place(baseball bat)', 'place(baseball glove)', 'place(skateboard)', 'place(tennis racket)', 'place(bottle)', 'place(wine glass)', 'place(fork)', 'place(knife)', 'place(spoon)', 'place(bowl)', 'place(banana)', 'place(apple)', 'place(sandwich)', 'place(orange)', 'place(broccoli)', 'place(carrot)', 'place(hot dog)', 'place(pizza)', 'place(donut)', 'place(cake)', 'place(chair)', 'place(couch)', 'place(potted plant)', 'place(dining table)', 'place(laptop)', 'place(mouse)', 'place(remote)', 'place(keyboard)', 'place(cell phone)', 'place(toaster)', 'place(book)', 'place(vase)', 'place(scissors)', 'place(teddy bear)', 'place(hair drier)', 'place(toothbrush)', 'pull(bench)', 'pull(bear)', 'pull(backpack)', 'pull(umbrella)', 'pull(handbag)', 'pull(tie)', 'pull(suitcase)', 'pull(frisbee)', 'pull(skis)', 'pull(snowboard)', 'pull(sports ball)', 'pull(kite)', 'pull(baseball bat)', 'pull(baseball glove)', 'pull(skateboard)', 'pull(tennis racket)', 'pull(bottle)', 'pull(wine glass)', 'pull(cup)', 'pull(fork)', 'pull(knife)', 'pull(spoon)', 'pull(bowl)', 'pull(banana)', 'pull(apple)', 'pull(sandwich)', 'pull(orange)', 'pull(broccoli)', 'pull(carrot)', 'pull(hot dog)', 'pull(pizza)', 'pull(donut)', 'pull(cake)', 'pull(chair)', 'pull(couch)', 'pull(potted plant)', 'pull(dining table)', 'pull(laptop)', 'pull(mouse)', 'pull(remote)', 'pull(keyboard)', 'pull(cell phone)', 'pull(toaster)', 'pull(book)', 'pull(vase)', 'pull(scissors)', 'pull(teddy bear)', 'pull(hair drier)', 'pull(toothbrush)', 'press(bench)', 'press(bear)', 'press(backpack)', 'press(umbrella)', 'press(handbag)', 'press(tie)', 'press(suitcase)', 'press(frisbee)', 'press(skis)', 'press(snowboard)', 'press(sports ball)', 'press(kite)', 'press(baseball bat)', 'press(baseball glove)', 'press(skateboard)', 'press(tennis racket)', 'press(bottle)', 'press(wine glass)', 'press(cup)', 'press(fork)', 'press(knife)', 'press(spoon)', 'press(bowl)', 'press(banana)', 'press(apple)', 'press(sandwich)', 'press(orange)', 'press(broccoli)', 'press(carrot)', 'press(hot dog)', 'press(pizza)', 'press(donut)', 'press(cake)', 'press(chair)', 'press(couch)', 'press(potted plant)', 'press(dining table)', 'press(laptop)', 'press(mouse)', 'press(remote)', 'press(keyboard)', 'press(cell phone)', 'press(toaster)', 'press(book)', 'press(vase)', 'press(scissors)', 'press(teddy bear)', 'press(hair drier)', 'press(toothbrush)']

    #finally_feasible = []
    #finally_unfeasible = []
    time_spent = []

    finally_feasible = ['bench', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'dining table', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'toaster', 'book', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    finally_unfeasible = ['person', 'bear', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'zebra', 'giraffe', 'surfboard', 'bed', 'toilet', 'tv', 'microwave', 'oven', 'sink', 'refrigerator', 'clock']


    if not finally_feasible and not finally_unfeasible:
        for obj in object_list:
            in_feasible = []
            in_unfeasible = []
            while obj not in in_feasible and obj not in in_unfeasible:
                objects = [obj]
                results = llm_class.classify(objects)
            
                feasible_match = re.search(r"Feasible Objects:\s*\[(.*?)\]", results, re.DOTALL)
                unfeasible_match = re.search(r"Unfeasible Objects:\s*\[(.*?)\]", results, re.DOTALL)

                print(f"feasible_match: {feasible_match}")
                print(f"unfeasible_match: {unfeasible_match}")

                feasible_objects = []
                unfeasible_objects = []

                if unfeasible_match:
                    unfeasible_objects = [obj.strip().strip("'\"") for obj in unfeasible_match.group(1).split(",") if obj.strip()] if unfeasible_match else []
                if feasible_match:
                    feasible_objects = [obj.strip().strip("'\"") for obj in feasible_match.group(1).split(",") if obj.strip()] if feasible_match else []

                print(f"Feasible Objects: {feasible_objects}")
                print(f"Unfeasible Objects: {unfeasible_objects}")

                # Check which known objects are in each category
                in_feasible = [obj for obj in object_list if obj in feasible_objects and obj not in unfeasible_objects]
                in_unfeasible = [obj for obj in object_list if obj in unfeasible_objects and obj not in feasible_objects]

                print(f"Objects in Feasible: {in_feasible}")
                print(f"Objects in Unfeasible: {in_unfeasible}")

                if obj in in_feasible:
                    finally_feasible.append(obj)
                elif obj in in_unfeasible:
                    finally_unfeasible.append(obj)

                print(f"Finally Feasible: {finally_feasible}")
                print(f"Finally Unfeasible: {finally_unfeasible}")
    else:
        print("Feasible objects already defined:")
        print(finally_feasible)
        print("Unfeasible objects already defined:")
        print(finally_unfeasible)
    
    print(f"Here is the final list of unfeasible objects: {finally_unfeasible} do you want to use one of them to create a goal? (yes/no)")
    user_input = input().strip().lower()

    while user_input == "yes":
        print("Please enter the object you want to use:")
        object_input = input().strip()
        if object_input in finally_unfeasible:
            print(f"You have chosen: {object_input}")
            finally_feasible.append(object_input)
            finally_unfeasible.remove(object_input)
            print(f"Updated feasible objects: {finally_feasible}")
            print(f"Do you want to add an other object :{finally_unfeasible}? (yes/no)")
            user_input = input().strip().lower()
        else:
            print(f"{object_input} is not in the list of unfeasible objects.")
            print(f"Do you want to retry to add an object from this list:{finally_unfeasible}? (yes/no)")
            user_input = input().strip().lower()
    
    print(f"Here is the final list of feasible objects: {finally_feasible} do you want to delete one of them? (yes/no)")
    delete_input = input().strip().lower()
    while delete_input == "yes":
        print("Please enter the object you want to delete:")
        object_input = input().strip()
        if object_input in finally_feasible:
            print(f"You have chosen to delete: {object_input}")
            finally_feasible.remove(object_input)
            finally_unfeasible.append(object_input)
            print(f"Updated unfeasible objects: {finally_unfeasible}")
            print(f"Do you want to delete an other object from this list:{finally_feasible}? (yes/no)")
            delete_input = input().strip().lower()
        else:
            print(f"{object_input} is not in the list of feasible objects.")
            print(f"Do you want to retry to delete an object from this list:{finally_feasible}? (yes/no)")
            delete_input = input().strip().lower()
    
    
    csv_goal_path="goals_type.csv"
    ensure_csv(csv_goal_path)

    feasible_twohands = []
    feasible_not_twohands = []
    for obj in finally_feasible:
        out_txt = llm_class.classify_twohands([obj])
        m_two  = re.search(r"(?mi)^\s*Two-Handed:\s*\[(.*?)\]\s*$", out_txt)
        m_not  = re.search(r"(?mi)^\s*Not-Two-Handed:\s*\[(.*?)\]\s*$", out_txt)
        twohands = [x.strip().strip("'\"") for x in m_two.group(1).split(",") if x.strip()] if m_two else []
        not_twohands = [x.strip().strip("'\"") for x in m_not.group(1).split(",") if x.strip()] if m_not else []
        if obj in twohands:
            feasible_twohands.append(obj)
            print(f"Two-Handed (feasible): {feasible_twohands}")
            append_unique_row(csv_goal_path, obj, "twohands")
        elif obj in not_twohands:
            feasible_not_twohands.append(obj)
            print(f"Not-Two-Handed (feasible): {feasible_not_twohands}")
        else:
            feasible_not_twohands.append(obj)
            print(f"Other (treated as Not-Two-Handed): {feasible_not_twohands}")
    print(f"Two-Handed (feasible): {feasible_twohands}")
    print(f"Not-Two-Handed (feasible): {feasible_not_twohands}")


    feasible_pourable = []
    feasible_non_pourable = []
    for obj in feasible_not_twohands:
        out_txt = llm_class.classify_pourable([obj])
        m_pour = re.search(r"(?mi)^\s*Pourable:\s*\[(.*?)\]\s*$", out_txt)
        m_non  = re.search(r"(?mi)^\s*Non-Pourable:\s*\[(.*?)\]\s*$", out_txt)
        pourables = [x.strip().strip("'\"") for x in m_pour.group(1).split(",") if x.strip()] if m_pour else []
        non_pourables = [x.strip().strip("'\"") for x in m_non.group(1).split(",") if x.strip()] if m_non else []
        if obj in pourables:
            feasible_pourable.append(obj)
            print(f"Pourable (feasible): {feasible_pourable}")
            append_unique_row(csv_goal_path, obj, "pourable")
        elif obj in non_pourables:
            feasible_non_pourable.append(obj)
            print(f"Non-Pourable (feasible): {feasible_non_pourable}")
        else:
            feasible_non_pourable.append(obj)
            print(f"Other (treated as Non-Pourable): {feasible_non_pourable}")
    print(f"Pourable (feasible): {feasible_pourable}")
    print(f"Non-Pourable (feasible): {feasible_non_pourable}")

    """
    feasible_containers = []
    feasible_non_containers = []
    for obj in feasible_not_twohands:
       
        out_txt = llm_class.classify_container([obj])
        m_cont = re.search(r"(?mi)^\s*Containers:\s*\[(.*?)\]\s*$", out_txt)
        m_non  = re.search(r"(?mi)^\s*Non-Containers:\s*\[(.*?)\]\s*$", out_txt)
        containers = []
        non_containers = []
        if m_cont:
            containers = [x.strip().strip("'\"") for x in m_cont.group(1).split(",") if x.strip()]
        if m_non:
            non_containers = [x.strip().strip("'\"") for x in m_non.group(1).split(",") if x.strip()]

        if obj in containers:
            feasible_containers.append(obj)
            print(f"Containers (feasible): {feasible_containers}")
            typ="container"
            append_unique_row(csv_goal_path, obj, typ)
        elif obj in non_containers:
            feasible_non_containers.append(obj)
            print(f"Non-Containers (feasible): {feasible_non_containers}")
        else:
            feasible_non_containers.append(obj)
            print(f"Other: {feasible_non_containers}")
    print(f"Containers (feasible): {feasible_containers}")
    print(f"Non-Containers (feasible): {feasible_non_containers}")

    feasible_contenable = []
    feasible_non_contenable = []
    for obj in feasible_not_twohands:
        out_txt = llm_class.classify_contenable([obj])
        m_cont = re.search(r"(?mi)^\s*Contenable:\s*\[(.*?)\]\s*$", out_txt)
        m_non  = re.search(r"(?mi)^\s*Non-Contenable:\s*\[(.*?)\]\s*$", out_txt)
        contenables = []
        non_contenables = []
        if m_cont:
            contenables = [x.strip().strip("'\"") for x in m_cont.group(1).split(",") if x.strip()]
        if m_non:
            non_contenables = [x.strip().strip("'\"") for x in m_non.group(1).split(",") if x.strip()]
        if obj in contenables:
            feasible_contenable.append(obj)
            print(f"Contenable (feasible): {feasible_contenable}")
            append_unique_row(csv_goal_path, obj, "contenable")
        elif obj in non_contenables:
            feasible_non_contenable.append(obj)
            print(f"Non-Contenable (feasible): {feasible_non_contenable}")
        else:
            feasible_non_contenable.append(obj)
            print(f"Other: {feasible_non_contenable}")
    print(f"Contenable (feasible): {feasible_contenable}")
    print(f"Non-Contenable (feasible): {feasible_non_contenable}")
    """
    print(f"Here is the final list of feasible objects: {finally_feasible}")
    #feasible_goals = create_all_feasible_goals(finally_feasible, list_of_actions)
    print("Here is the final list of feasible goals:")
    #list_of_goals = create_all_feasible_goals(finally_feasible, list_of_actions)
    #print(list_of_goals)

    #single_pour_goals = create_single_pour_goals(csv_goal_path)
    #list_of_goals = create_single_pour_goals(csv_goal_path)


    feasible_goals = create_all_feasible_goals(finally_feasible, list_of_actions)
    pour_goals     = create_single_pour_goals(csv_goal_path) 

    list_of_goals = merge_goals(feasible_goals, pour_goals)

    print("Here is the final list of feasible objects:", finally_feasible)
    print("Here is the final list of goals:", list_of_goals)

    csv_filename = "time_spent.csv"
    missing_goals = []
       
    if not os.path.exists(csv_filename):
        time_spent = llm_class.give_time(list_of_goals)

        matches = re.findall(r"\{.*?\}", time_spent, re.DOTALL)
        if matches:
            last_json_block = matches[-1]
            try:
                time_spent_dict = json.loads(last_json_block)
            except json.JSONDecodeError as e:
                raise  ValueError(f"Error decoding JSON: {e}")
        else:
            raise ValueError("Could not extract a valid JSON block from the response.")
            
        print(f"time spent on each goal: {time_spent_dict}")
        csv_filename = "time_spent.csv"
        with open(csv_filename, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Goal", "Time(seconds)"])
            for goal, time in time_spent_dict.items():
                writer.writerow([goal,time])
        print(f"time spent data written to {csv_filename}")

    while True:
        if os.path.exists(csv_filename):
            with open(csv_filename, mode= 'r', newline='') as csv_file:
                reader =csv.DictReader(csv_file)
                if reader.fieldnames is None or "Goal" not in reader.fieldnames or "Time(seconds)" not in reader.fieldnames:
                    print("Header missing or corrupted in time_spent.csv — rewriting it.")
                    with open(csv_filename, mode='w', newline='') as rewrite_file:
                        writer = csv.writer(rewrite_file)
                        writer.writerow(["Goal", "Time(seconds)"])
                    existing_goals = set()
                existing_goals = {row["Goal"] for row in reader if "Goal" in row }

            missing_goals = [goal for goal in list_of_goals if goal not in existing_goals]
            hallucinated_goals = [goal for goal in existing_goals if goal not in list_of_goals]
        if hallucinated_goals:
            print(f"Hallucinated goals detected in CSV: {hallucinated_goals}")

            with open(csv_filename, mode='r', newline='') as csv_file:
                reader = csv.DictReader(csv_file)
                all_rows = [row for row in reader if row["Goal"] in list_of_goals]

            with open(csv_filename, mode='w', newline='') as csv_file:
                fieldnames = ["Goal", "Time(seconds)"]  
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_rows)

            print(f"Removed hallucinated goals. CSV now only contains valid entries.")

        if missing_goals:
            print(f"Missing goals: {missing_goals}")
            time_spent = llm_class.give_time(missing_goals)
            matches = re.findall(r"\{.*?\}", time_spent, re.DOTALL)
            if matches:
                last_json_block = matches[-1]
                try:
                    time_spent_dict = json.loads(last_json_block)
                except json.JSONDecodeError as e:
                    raise  ValueError(f"Error decoding JSON: {e}")
            else:
                raise ValueError("Could not extract a valid JSON block from the response.")
                
            print(f"time spent on each goal: {time_spent_dict}")
            csv_filename = "time_spent.csv"
            with open(csv_filename, mode='a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                for goal, time in time_spent_dict.items():
                    writer.writerow([goal,time])
            print(f"missing goals append to {csv_filename}")
        
        if not missing_goals and not hallucinated_goals:
            print(f"Time spent data already exists in {csv_filename}. No need to query the model again.")
            break


    ##with open("related_goals.csv", "w") as f_csv:
        ##f_csv.write("current_goal,next_goal,probability\n")

    # Ensure CSV has a header if it's newly created
    csv_path = "related_goals.csv"
    if not os.path.exists(csv_path):
        with open(csv_path, "w") as f_csv:
            f_csv.write("current_goal,next_goal,probability\n")
    
    # Step 1: Load already processed goals
    processed_goals = set()
    if os.path.exists("related_goals.json"):
        with open("related_goals.json", "r") as f_jsonl:
            for line in f_jsonl:
                try:
                    data = json.loads(line)
                    processed_goals.update(data.keys())
                except json.JSONDecodeError:
                    continue

    # Step 2: Identify missing goals
    missing_related_goals = [g for g in list_of_goals if g not in processed_goals]
    print(f"list_of_goals: {list_of_goals}")
    print(f"Missing related goals: {missing_related_goals}")

    while missing_related_goals: #change to while(missing_related_goals):
        current_goal =missing_related_goals[0]
        related_goals = llm_class.related_goals(list_of_goals, current_goal)
        print(f"Related goals: {related_goals}")

        match = re.search(r'{[^}]+}', related_goals, re.DOTALL)
        if match:
            try:
                goals_dict = json.loads(match.group())

                # Wrap result using the current goal as key
                structured_result = {
                    current_goal: goals_dict
                }

                # Save to JSON
                with open("related_goals.json", "a") as f_jsonl:
                    f_jsonl.write(json.dumps(structured_result) + "\n")
                    #json.dump(structured_result, f_json, indent=4)

                # Save to CSV
                with open("related_goals.csv", "a") as f_csv:
                    for goal, prob in goals_dict.items():
                        f_csv.write(f"{current_goal},{goal},{prob}\n")

                print("✅ Saved for current goal:", current_goal)
                missing_related_goals.pop(0)  # Remove the current goal from the missing_related_goals list
                #take off the current goals from the missing_related_goals list

            except json.JSONDecodeError as e:
                print(f"❌ Failed to decode JSON: {e}")
        else:
            print("❌ No valid JSON block found in content")


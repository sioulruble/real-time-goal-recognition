import os
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch


class SentenceSimilarityModel:
    def __init__(self, model):
        self.model = model

    def compute_similarity(self, sentence1, sentence2):
        """
        Compute the similarity between two sentences using the provided model.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        embedding1 = self.model.encode(sentence1, convert_to_tensor=True)
        embedding2 = self.model.encode(sentence2, convert_to_tensor=True)

        embedding1 = embedding1.to(device)
        embedding2 = embedding2.to(device)


        # Compute cosine similarity
        similarity = util.cos_sim(embedding1, embedding2)
        
        return similarity
    
    def all_possible_actions(self, list_of_actions, list_of_objects):
        """
        Generate all possible actions by replacing placeholders in action templates with actual object names.
        """
        All_possible_actions = []
        for action in list_of_actions:
            for obj in list_of_objects:
                All_possible_actions.append(action.replace("object1", obj))
        return All_possible_actions
    
    def get_paired_sorted(self, sentence1, All_possible_actions):
        """
        Get the sorted list of actions based on their similarity to the input sentence.
        """
        similarity = self.compute_similarity(sentence1, All_possible_actions).tolist()
        paired_similarity = list(zip(All_possible_actions, similarity[0]))
        paired_sorted = sorted(paired_similarity, key=lambda x: x[1], reverse=True)
        #print(f"Similarity: {paired_sorted}")
        return paired_sorted
    
    def get_list_of_actions(self, paired_sorted):
        """
        Extract the list of actions from the paired sorted list.
        """
        list_of_actions = [action for action, _ in paired_sorted]
        return list_of_actions
    
    def similarity_model(self, sentence1, list_of_actions, list_of_objects):
        """
        Main function to compute the similarity model.
        """
        All_possible_actions = self.all_possible_actions(list_of_actions, list_of_objects)

        paired_sorted = self.get_paired_sorted(sentence1, All_possible_actions)

        # Get the list of actions from the paired sorted list
        list_final = self.get_list_of_actions(paired_sorted)
        
        return list_final
    
def process_multiline_caption(caption: str, similarity_model: SentenceSimilarityModel, list_of_actions, list_of_objects, top_k=1):
    # Split caption into non-empty lines
    lines = extract_clean_lines(caption)

    selected_goals = []
    all_possible = []
    # Generate all possible actions from current objects
    all_possible = similarity_model.all_possible_actions(list_of_actions, list_of_objects)

    for line in lines:
        paired_sorted = similarity_model.get_paired_sorted(line, all_possible)
        top_actions = [action for action, score in paired_sorted[:top_k]]
        selected_goals.extend(top_actions)

    return selected_goals

def extract_clean_lines(caption: str):
    return [line.strip().replace("'", "").replace("[", "").replace("]", "") for line in caption.split(',')]





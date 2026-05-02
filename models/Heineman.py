"""Heineman category-transition models and variants."""

from pylab import *
import numpy as np
np.random.seed(42)
import sys
import os
import pickle as pk
from Model import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))
from utils import *
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Heineman(Model):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        self.model_class = "heineman"
        self.cat_trans = self.get_category_transition_matrices()
        self.num_total_weights = 3
        self.cat_trans_terms = {}
        self.split_ind = 0
     
    def create_models(self):
        """Instantiate enabled subclasses for the current config."""
        self.models = {subclass.__name__: subclass(self) for subclass in Heineman.__subclasses__() if self.modelstorun.get(subclass.__name__) == 1}
    
    def get_category_transition_matrices(self):
        """Compute per-fold category transition matrices."""
        normalized_transition_matrices = np.zeros((self.config["cv"], self.num_categories, self.num_categories))
        if self.config["fitting"] == "group":
            for i, (train_seqs, _) in enumerate(self.splits):
                transition_matrix = np.zeros((self.num_categories, self.num_categories))
                for seq in train_seqs:
                    for r in range(1, len(seq)):
                        previous_categories = self.response_to_category[seq[r - 1]]
                        categories = self.response_to_category[seq[r]]
                        
                        for prev in previous_categories:
                            for curr in categories:
                                try:
                                    transition_matrix[prev, curr] += 1
                                except:
                                    continue  # when NaN
                normalized_transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
                normalized_transition_matrices[i] = normalized_transition_matrix

        else:
            transition_matrix = np.zeros((self.num_categories, self.num_categories))
            self.data["categories"] = self.data["response"].map(self.response_to_category)
            self.data["previous_categories"] = self.data.groupby("pid")["categories"].shift()
            data_of_interest = self.data.dropna(subset=["previous_categories"])
            for _, row in data_of_interest.iterrows():
                for prev in row["previous_categories"]:
                    for curr in row["categories"]:
                        try:
                            transition_matrix[prev, curr] += 1
                        except:
                            continue  # when NaN
            normalized_transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
            normalized_transition_matrices[0] = normalized_transition_matrix
        
        return normalized_transition_matrices
    
    def get_category_cue(self, response, previous_response):
        cats_prev = set(self.response_to_category[previous_response])
        cats_curr = set(self.response_to_category[response])
        cats_intersection = cats_prev.intersection(cats_curr)

        if len(cats_intersection) != 0:
            return max([self.cat_trans[self.split_ind, c, c] for c in cats_intersection])
        else:
            return max([self.cat_trans[self.split_ind, c1, c2] for c1 in cats_prev for c2 in cats_curr])

    def allweights(self, weights=None):
        """Returns a vector of all weights, with 0s or constants in non-trainable positions."""
        w = torch.zeros(self.num_total_weights, device=device)
        if self.num_weights > 0:
            if weights is None:
                w[self.weight_indices] = self.weights
            else:
                w[self.weight_indices] = weights
        return w
    
    def get_nll(self, seq, weightsfromarg=None):
        nll = 0
        # Similarity and category-transition cues for each step.
        sim_terms = torch.stack([self.d2ts(self.sim_mat[r]) for r in seq[1:-1]]).to(device=device)                      # shape: (len_seq - 2, num_resp)
        try:
            cat_trans_terms = self.cat_trans_terms[' '.join(seq)]
        except:
            cat_trans_terms = torch.stack([torch.tensor([self.get_category_cue(r_, r) for r_ in self.unique_responses], dtype=float, device=device) for r in seq[1:-1]]).to(device=device)                      # shape: (len_seq - 2, num_resp)
            self.cat_trans_terms[' '.join(seq)] = cat_trans_terms

        # Prevent revisiting previously used responses.
        mask = np.ones((len(seq) - 2, len(self.unique_responses)))
        for i in range(2, len(seq)):
            visited_responses = np.array([self.unique_response_to_index[resp] for resp in seq[:i]])
            mask[i - 2, visited_responses] = 0

        weightstouse = self.allweights(weightsfromarg)
        # Combine frequency, similarity, and category-transition cues.
        logits = (
            weightstouse[0] * self.d2ts(self.freq).unsqueeze(0) +                                                    # shape: (1, num_resp)
            weightstouse[1] * sim_terms +                                                                            # shape: (len_seq - 2, num_resp)
            weightstouse[2] * cat_trans_terms
        ) 
        
        if self.config["mask"]:
            mask = torch.tensor(mask, dtype=torch.bool, device=device)
            logits[mask == 0] = float('-inf')                                                                                                    # shape: (len_seq - 2, num_resp)
        log_probs = F.log_softmax(logits, dim=1)                                                                        # shape: (len_seq - 2, num_resp)
        
        if weightsfromarg is not None:
            return log_probs
        
        targets = torch.tensor([self.unique_response_to_index[r] for r in seq], device=device)
        nll = F.nll_loss(log_probs, targets[2:], reduction='sum')

        return nll

class Freq_Sim_Subcategory(Heineman, nn.Module):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        nn.Module.__init__(self)
        self.num_weights = 3
        self.weight_indices = torch.tensor([0, 1, 2], device=device)

        self.weights = nn.Parameter(torch.tensor([self.init_val] * self.num_weights, device=device))
        
class Subcategory(Heineman, nn.Module):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        nn.Module.__init__(self)
        self.num_weights = 1
        self.weight_indices = torch.tensor([2], device=device)

        self.weights = nn.Parameter(torch.tensor([self.init_val] * self.num_weights, device=device))

class Freq_Subcategory(Heineman, nn.Module):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        nn.Module.__init__(self)
        self.num_weights = 2
        self.weight_indices = torch.tensor([0, 2], device=device)

        self.weights = nn.Parameter(torch.tensor([self.init_val] * self.num_weights, device=device))

class Sim_Subcategory(Heineman, nn.Module):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        nn.Module.__init__(self)
        self.num_weights = 2
        self.weight_indices = torch.tensor([1, 2], device=device)

        self.weights = nn.Parameter(torch.tensor([self.init_val] * self.num_weights, device=device))

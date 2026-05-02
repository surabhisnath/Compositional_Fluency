"""Hills family models for sequence prediction and weighting variants."""

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
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Hills(Model):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        self.model_class = "hills"
        self.num_total_weights = 3
     
    def create_models(self):
        """Instantiate enabled subclasses for the current config."""
        self.models = {subclass.__name__: subclass(self) for subclass in Hills.__subclasses__() if self.modelstorun.get(subclass.__name__) == 1}
    
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
        # Similarity terms for each step in the sequence (excluding start/end).
        sim_terms = torch.stack([self.d2ts(self.sim_mat[r]) for r in seq[1:-1]]).to(device=device)                              # shape: (len_seq - 2, num_resp)
        # sim_terms_2step = torch.stack([self.d2ts(self.sim_mat[r]) for r in seq[:-2]]).to(device=device)                         # shape: (len_seq - 2, num_resp)
        
        sim_terms_mask = torch.ones(len(seq) - 2, dtype=torch.int8, device=device).unsqueeze(1)  # Default: all ones
        if self.dynamic:
            if self.dynamic_cat:
                if seq[-1] == "":  # --- Simulation mode: predict next response
                    prev_resp = seq[-2]
                    prev_cat = set(self.response_to_category[prev_resp])
                    sim_terms_mask = torch.tensor([bool(prev_cat & set(self.response_to_category[r])) for r in self.unique_responses], dtype=torch.int8, device=device).unsqueeze(0)  # shape: (1, num_resp)
                else:
                    sim_terms_mask = torch.tensor([ bool(set(self.response_to_category[seq[i]]) & set(self.response_to_category[seq[i - 1]])) for i in range(2, len(seq))], dtype=torch.int8, device=device).unsqueeze(1)   # shape: (len(seq) - 2, 1)

            # elif self.sim_drop:
            #     sim1 = torch.tensor([self.sim_mat[seq[i - 2]][seq[i - 1]] for i in range(2, len(seq) - 1)], dtype=torch.float16, device=device)
            #     sim2 = torch.tensor([self.sim_mat[seq[i - 1]][seq[i]] for i in range(2, len(seq) - 1)], dtype=torch.float16, device=device)
            #     sim3 = torch.tensor([self.sim_mat[seq[i]][seq[i + 1]] for i in range(2, len(seq) - 1)], dtype=torch.float16, device=device)
            #     # sim_drops = ((sim1 > sim2) & (sim2 < sim3)).to(torch.int8)                                 # (len(seq) - 3,)
            #     sim_drops = (~((sim1 > sim2) & (sim2 < sim3))).to(torch.int8)                              
            #     sim_terms_mask = torch.cat([sim_drops, torch.tensor([0], dtype=torch.int8, device=device)])  # Pad to length (len(seq) - 2)

        # Prevent revisiting previously used responses.
        mask = np.ones((len(seq) - 2, len(self.unique_responses)))
        for i in range(2, len(seq)):
            visited_responses = np.array([self.unique_response_to_index[resp] for resp in seq[:i]])
            mask[i - 2, visited_responses] = 0

        weightstouse = self.allweights(weightsfromarg)
        
        # a = weightstouse[0] * self.d2ts(self.freq).unsqueeze(0)
        # b = weightstouse[1] * sim_terms * sim_terms_mask.unsqueeze(1).float()
        # c = weightstouse[2] * sim_terms_2step
        
        # print(dict(zip(seq[1:], [self.response_to_category[seq[i]] for i in range(1,len(seq))])))
        # print("b vector")
        # print(b)
        
        # Combine frequency and similarity cues into logits.
        logits = (
            weightstouse[0] * self.d2ts(self.freq).unsqueeze(0) +               # 1, num_responses
            weightstouse[1] * sim_terms * sim_terms_mask.float() #+             # 
            # weightstouse[2] * sim_terms_2step
        )
        
        if self.config["mask"]:
            mask = torch.tensor(mask, dtype=torch.bool, device=device)
            logits[mask == 0] = float('-inf')
        log_probs = F.log_softmax(logits, dim=1)                                                  # shape: (len_seq - 2, num_resp)

        if weightsfromarg is not None:
            return log_probs
        
        targets = torch.tensor([self.unique_response_to_index[r] for r in seq], device=device)
        nll = F.nll_loss(log_probs, targets[2:], reduction='sum')

        return nll

class OneCueStaticLocal(Hills, nn.Module):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        nn.Module.__init__(self)
        self.num_weights = 1
        self.weight_indices = torch.tensor([1], device=device)

        self.weights = nn.Parameter(torch.tensor([self.init_val] * self.num_weights, device=device))

class OneCueStaticLocalWeighted(Hills, nn.Module):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        nn.Module.__init__(self)
        self.num_weights = self.num_embedding_dims
        self.weights = nn.Parameter(torch.tensor([self.init_val] * self.num_weights, device=device))
        self.onlyforgroup = True

    def get_nll(self, seq, weightsfromarg=None):
        if weightsfromarg is not None:
            weightstouse = weightsfromarg
        else:
            weightstouse = self.weights

        nll = 0
        prev_embeddings = torch.stack([torch.tensor(self.embeddings[r], device=device) for r in seq[1:-1]]).to(device=device)    # Shape: (L, D) where L = len(seq) - 2
        all_embeddings = torch.stack([torch.tensor(self.embeddings[r], device=device) for r in self.unique_responses])           # Shape: (N, D) where N = num unique responses
        all_dots = (prev_embeddings.unsqueeze(1) * all_embeddings.unsqueeze(0)).float()  # Shape: (L, N, D)
        
        mask = np.ones((len(seq) - 2, len(self.unique_responses)))
        for i in range(2, len(seq)):
            visited_responses = np.array([self.unique_response_to_index[resp] for resp in seq[:i]])
            mask[i - 2, visited_responses] = 0

        logits = torch.einsum("lnd,d->ln", all_dots, weightstouse)                                                                        # Shape: (L, N)
        
        if self.config["mask"]:
            mask = torch.tensor(mask, dtype=torch.bool, device=device)
            logits[mask == 0] = float('-inf')
        log_probs = F.log_softmax(logits, dim=1)

        if weightsfromarg is not None:
            return log_probs

        targets = torch.tensor([self.unique_response_to_index[r] for r in seq], device=device)
        nll = F.nll_loss(log_probs, targets[2:], reduction='sum')
        return nll

class OneCueStaticLocalWeightedActivity(Hills, nn.Module):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        nn.Module.__init__(self)
        self.num_weights = self.num_embedding_dims * 2
        self.weights = nn.Parameter(torch.tensor([self.init_val] * self.num_weights, device=device))
        self.onlyforgroup = True

    def get_nll(self, seq, weightsfromarg=None):
        if weightsfromarg is not None:
            weightstouse = weightsfromarg
        else:
            weightstouse = self.weights

        nll = 0
        prev_embeddings = torch.stack([torch.tensor(self.embeddings[r], device=device) for r in seq[1:-1]]).to(device=device)    # Shape: (L, D) where L = len(seq) - 2
        all_embeddings = torch.stack([torch.tensor(self.embeddings[r], device=device) for r in self.unique_responses])           # Shape: (N, D) where N = num unique responses
        all_dots = (prev_embeddings.unsqueeze(1) * all_embeddings.unsqueeze(0)).float()  # Shape: (L, N, D)
        
        mask = np.ones((len(seq) - 2, len(self.unique_responses)))
        for i in range(2, len(seq)):
            visited_responses = np.array([self.unique_response_to_index[resp] for resp in seq[:i]])
            mask[i - 2, visited_responses] = 0

        logits = torch.einsum("lnd,d->ln", all_dots, weightstouse[:self.num_weights//2]) + torch.einsum("nd,d->n", all_embeddings.to(torch.float), weightstouse[self.num_weights//2:]).unsqueeze(0)                                                                       # Shape: (L, N)
        
        if self.config["mask"]:
            mask = torch.tensor(mask, dtype=torch.bool, device=device)
            logits[mask == 0] = float('-inf')
        log_probs = F.log_softmax(logits, dim=1)

        if weightsfromarg is not None:
            return log_probs

        targets = torch.tensor([self.unique_response_to_index[r] for r in seq], device=device)
        nll = F.nll_loss(log_probs, targets[2:], reduction='sum')
        return nll

class OneCueStaticLocal_2step(Hills, nn.Module):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        nn.Module.__init__(self)
        self.num_weights = 1
        self.weight_indices = torch.tensor([2], device=device)

        self.weights = nn.Parameter(torch.tensor([self.init_val] * self.num_weights, device=device))

class OneCueStaticLocal_12step(Hills, nn.Module):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        nn.Module.__init__(self)
        self.num_weights = 2
        self.weight_indices = torch.tensor([1, 2], device=device)

        self.weights = nn.Parameter(torch.tensor([self.init_val] * self.num_weights, device=device))

class CombinedCueStaticWeightedActivity(Hills, nn.Module):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        nn.Module.__init__(self)
        self.num_weights = self.num_embedding_dims * 2 + 1
        self.weights = nn.Parameter(torch.tensor([self.init_val] * self.num_weights, device=device))
        self.onlyforgroup = True

    def get_nll(self, seq, weightsfromarg=None):
        if weightsfromarg is not None:
            weightstouse = weightsfromarg
        else:
            weightstouse = self.weights

        nll = 0
        prev_embeddings = torch.stack([torch.tensor(self.embeddings[r], device=device) for r in seq[1:-1]]).to(device=device)    # Shape: (L, D) where L = len(seq) - 2
        all_embeddings = torch.stack([torch.tensor(self.embeddings[r], device=device) for r in self.unique_responses])           # Shape: (N, D) where N = num unique responses
        all_dots = (prev_embeddings.unsqueeze(1) * all_embeddings.unsqueeze(0)).float()  # Shape: (L, N, D)
        
        mask = np.ones((len(seq) - 2, len(self.unique_responses)))
        for i in range(2, len(seq)):
            visited_responses = np.array([self.unique_response_to_index[resp] for resp in seq[:i]])
            mask[i - 2, visited_responses] = 0

        logits = weightstouse[0] * self.d2ts(self.freq).unsqueeze(0) + torch.einsum("lnd,d->ln", all_dots, weightstouse[1:1 + self.num_weights//2]) + torch.einsum("nd,d->n", all_embeddings.to(torch.float), weightstouse[1 + self.num_weights//2:]).unsqueeze(0)                                                                       # Shape: (L, N)
        
        if self.config["mask"]:
            mask = torch.tensor(mask, dtype=torch.bool, device=device)
            logits[mask == 0] = float('-inf')
        log_probs = F.log_softmax(logits, dim=1)

        if weightsfromarg is not None:
            return log_probs

        targets = torch.tensor([self.unique_response_to_index[r] for r in seq], device=device)
        nll = F.nll_loss(log_probs, targets[2:], reduction='sum')
        return nll

class CombinedCueStaticWeighted(Hills, nn.Module):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        nn.Module.__init__(self)
        self.num_weights = self.num_embedding_dims + 1
        self.weights = nn.Parameter(torch.tensor([self.init_val] * self.num_weights, device=device))
        self.onlyforgroup = True

    def get_nll(self, seq, weightsfromarg=None):
        if weightsfromarg is not None:
            weightstouse = weightsfromarg
        else:
            weightstouse = self.weights

        nll = 0
        prev_embeddings = torch.stack([torch.tensor(self.embeddings[r], device=device) for r in seq[1:-1]]).to(device=device)    # Shape: (L, D) where L = len(seq) - 2
        all_embeddings = torch.stack([torch.tensor(self.embeddings[r], device=device) for r in self.unique_responses])           # Shape: (N, D) where N = num unique responses
        all_dots = (prev_embeddings.unsqueeze(1) * all_embeddings.unsqueeze(0)).float()  # Shape: (L, N, D)
        
        mask = np.ones((len(seq) - 2, len(self.unique_responses)))
        for i in range(2, len(seq)):
            visited_responses = np.array([self.unique_response_to_index[resp] for resp in seq[:i]])
            mask[i - 2, visited_responses] = 0

        logits = weightstouse[0] * self.d2ts(self.freq).unsqueeze(0) + torch.einsum("lnd,d->ln", all_dots, weightstouse[1:])                                                                        # Shape: (L, N)
        
        if self.config["mask"]:
            mask = torch.tensor(mask, dtype=torch.bool, device=device)
            logits[mask == 0] = float('-inf')
        log_probs = F.log_softmax(logits, dim=1)

        if weightsfromarg is not None:
            return log_probs

        targets = torch.tensor([self.unique_response_to_index[r] for r in seq], device=device)
        nll = F.nll_loss(log_probs, targets[2:], reduction='sum')
        return nll

class CombinedCueStatic(Hills, nn.Module):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        nn.Module.__init__(self)
        self.num_weights = 2
        self.weight_indices = torch.tensor([0, 1], device=device)

        self.weights = nn.Parameter(torch.tensor([self.init_val] * self.num_weights, device=device))

class CombinedCueStatic_12step(Hills, nn.Module):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        nn.Module.__init__(self)
        self.num_weights = 3
        self.weight_indices = torch.tensor([0, 1, 2], device=device)

        self.weights = nn.Parameter(torch.tensor([self.init_val] * self.num_weights, device=device))

class CombinedCueDynamicCat(Hills, nn.Module):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        nn.Module.__init__(self)
        self.num_weights = 2
        self.weight_indices = torch.tensor([0, 1], device=device)
        self.dynamic = True
        self.dynamic_cat = True

        self.weights = nn.Parameter(torch.tensor([self.init_val] * self.num_weights, device=device))

class CombinedCueDynamicSimdrop(Hills, nn.Module):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        nn.Module.__init__(self)
        self.num_weights = 2
        self.weight_indices = torch.tensor([0, 1], device=device)
        self.dynamic = True
        self.dynamic_cat = False
        self.sim_drop = True

        self.weights = nn.Parameter(torch.tensor([self.init_val] * self.num_weights, device=device))

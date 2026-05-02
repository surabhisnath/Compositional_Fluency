"""Feature-based model family with static and activity-weighted variants."""

from pylab import *
import numpy as np
np.random.seed(42)
import sys
import os
import pickle as pk
from Model import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))
from collections import OrderedDict
from utils import *
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Ours(Model):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        self.model_class = "ours"
        if self.config["remove_weights_that_donot_recover"]:
            self.feature_names_HS, self.features_HS, self.feature_names_Act, self.features_Act = self.get_features()
            self.num_features_HS = len(self.feature_names_HS)
            self.num_features_Act = len(self.feature_names_Act)
            print(self.num_features_HS)
            print(self.num_features_Act)
            self.sim_mat = self.get_feature_sim_mat_recoverable()
        else:
            self.feature_names, self.features = self.get_features()
            self.num_features = len(self.feature_names)
            print(self.num_features)
            self.sim_mat = self.get_feature_sim_mat()
        self.pers_mat = self.get_feature_pers_mat()
        # self.all_features = torch.stack([self.features[r] for r in self.unique_responses])  # shape: [R, D]
        self.num_total_weights = 3
        self.onlyforgroup = False

    def create_models(self):
        """Instantiate enabled subclasses, respecting fitting mode."""
        if self.config["fitting"] == "individual":
            subclasses = [subclass for subclass in Ours.__subclasses__() if self.modelstorun.get(subclass.__name__) == 1 and not getattr(instance := subclass(self), 'onlyforgroup', False)]
        else:
            subclasses = [subclass for subclass in Ours.__subclasses__() if self.modelstorun.get(subclass.__name__) == 1]
        self.models = {cls.__name__: cls(self) for cls in subclasses}

    def get_features(self):
        """Load binary feature vectors for each response."""
        featuredict = pk.load(open(f"../files/features_{self.config['featurestouse']}.pk", "rb"))
        feature_names = list(next(iter(featuredict.values())).keys())
        parameter_recovery_df = pd.read_csv("../csvs/parameter_recovery.csv")
        if self.config["remove_features_that_donot_recover"]:
            unrecovered_features = [f"feature_{feat}" for feat in parameter_recovery_df.loc[(parameter_recovery_df["R2_HS"] < 0.5) & (parameter_recovery_df["R2_Activity"] < 0.5), "Feature"]] 
            print(unrecovered_features)
            feature_names_that_recover = [f for f in feature_names if f not in unrecovered_features]
            return feature_names_that_recover, {self.corrections.get(k, k): torch.tensor([1 if values.get(f, "").lower()[:4] == "true" else 0 for f in feature_names_that_recover], dtype=torch.int8, device=device) for k, values in featuredict.items()}
        if self.config["remove_weights_that_donot_recover"]:
            unrecovered_weights_HS = [f"feature_{feat}" for feat in parameter_recovery_df.loc[parameter_recovery_df["R2_HS"] < 0.5, "Feature"]]
            unrecovered_weights_Act = [f"feature_{feat}" for feat in parameter_recovery_df.loc[parameter_recovery_df["R2_Activity"] < 0.5, "Feature"]]
            print(unrecovered_weights_HS)
            print(unrecovered_weights_Act)
            feature_names_that_recover_HS = [f for f in feature_names if f not in unrecovered_weights_HS]
            feature_names_that_recover_Act = [f for f in feature_names if f not in unrecovered_weights_Act]
            return feature_names_that_recover_HS, {self.corrections.get(k, k): torch.tensor([1 if values.get(f, "").lower()[:4] == "true" else 0 for f in feature_names_that_recover_HS], dtype=torch.int8, device=device) for k, values in featuredict.items()}, \
                   feature_names_that_recover_Act, {self.corrections.get(k, k): torch.tensor([1 if values.get(f, "").lower()[:4] == "true" else 0 for f in feature_names_that_recover_Act], dtype=torch.int8, device=device) for k, values in featuredict.items()}, \

        return feature_names, {self.corrections.get(k, k): torch.tensor([1 if values.get(f, "").lower()[:4] == "true" else 0 for f in feature_names], dtype=torch.int8, device=device) for k, values in featuredict.items()}

    def get_feature_sim_mat(self):
        """Compute pairwise similarity as mean feature overlap."""
        sim_matrix = {response: {} for response in self.unique_responses}
        self.not_change_mat = torch.zeros((len(self.unique_responses), len(self.unique_responses), self.num_features), dtype=torch.int8)
        for i, resp1 in enumerate(self.unique_responses):
            for j, resp2 in enumerate(self.unique_responses):
                feat1 = self.features[resp1]
                feat2 = self.features[resp2]
                equal_feats = (feat1 == feat2).to(torch.int8)  # (D,)
                sim = equal_feats.float().mean().item()
                sim_matrix[resp1][resp2] = sim
                sim_matrix[resp2][resp1] = sim
                self.not_change_mat[i, j] = equal_feats  # (D,)
        return sim_matrix

    def get_feature_sim_mat_recoverable(self):
        """Compute pairwise similarity as mean feature overlap."""
        sim_matrix = {response: {} for response in self.unique_responses}
        self.not_change_mat = torch.zeros((len(self.unique_responses), len(self.unique_responses), self.num_features_HS), dtype=torch.int8)
        for i, resp1 in enumerate(self.unique_responses):
            for j, resp2 in enumerate(self.unique_responses):
                feat1 = self.features_HS[resp1]
                feat2 = self.features_HS[resp2]
                equal_feats = (feat1 == feat2).to(torch.int8)  # (D,)
                sim = equal_feats.float().mean().item()
                sim_matrix[resp1][resp2] = sim
                sim_matrix[resp2][resp1] = sim
                self.not_change_mat[i, j] = equal_feats  # (D,)
        return sim_matrix
    
    def get_feature_pers_mat(self):
        # return torch.einsum('ijd,jkd->ijk', self.not_change_mat, self.not_change_mat)   # (N, N, N) / (N, N, 1) = (N, N, N)
        return torch.einsum('ijd,jkd->ijk', self.not_change_mat, self.not_change_mat)  / (self.not_change_mat.sum(dim=2).unsqueeze(2) + 1e-6)   # (N, N, N) / (N, N, 1) = (N, N, N)

    def allweights(self, weights=None):
        """Returns a vector of all weights, with 0s or constants in non-trainable positions."""
        w = torch.zeros(self.num_total_weights, device=device)
        if self.num_weights > 0:
            if weights is None:
                w[self.weight_indices] = self.weights
            else:
                w[self.weight_indices] = weights
        return w
    
    def get_nll(self, seq, weightsfromarg=None, getnll=None):
        nll = 0
        # Similarity cues derived from feature overlap.
        sim_terms = torch.stack([self.d2ts(self.sim_mat[r]) for r in seq[1:-1]]).to(device=device)                      # shape: (len_seq - 2, num_resp)

        previous_responses = [self.unique_response_to_index[r] for r in seq[1:-1]]
        previous_previous_responses = [self.unique_response_to_index[r] for r in seq[:-2]]
        pers_terms = self.pers_mat[previous_previous_responses, previous_responses]                                     # shape: (len_seq - 2, num_resp)

        # mask = np.ones((len(seq) - 2, len(self.unique_responses)))
        # for i in range(2, len(seq)):
        #     visited_responses = np.array([self.unique_response_to_index[resp] for resp in seq[:i]])
        #     mask[i - 2, visited_responses] = 0
        # Mask previously visited responses in each sequence position.
        mask = torch.ones((len(seq) - 2, len(self.unique_responses)), device=device)
        for i in range(2, len(seq)):
            visited_responses = torch.tensor([self.unique_response_to_index[resp] for resp in seq[:i]], device=device)
            mask[i - 2, visited_responses] = 0.0

        weightstouse = self.allweights(weightsfromarg)
        logits = (
            weightstouse[0] * self.d2ts(self.freq).unsqueeze(0).expand(sim_terms.shape) +                             # shape: (1, num_resp)
            weightstouse[1] * sim_terms                                                                          # shape: (len_seq - 2, num_resp)
            + weightstouse[2] * self.np2ts(pers_terms)
        )

        # if self.config["mask"]:
        #     mask = torch.tensor(mask, dtype=torch.bool, device=device)
        #     logits[mask == 0] = float('-inf')
        if self.config["mask"]:
            logits = logits.masked_fill(mask == 0, float('-inf'))

        log_probs = F.log_softmax(logits, dim=1)

        if weightsfromarg is not None and getnll is None:
            return log_probs
        
        targets = torch.tensor([self.unique_response_to_index[r] for r in seq], device=device)
        nll = F.nll_loss(log_probs, targets[2:], reduction='sum')

        return nll
      
class Random(Ours, nn.Module):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        nn.Module.__init__(self)
        self.num_weights = 0
        self.weight_indices = torch.tensor([], device=device)

class Freq(Ours, nn.Module):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        nn.Module.__init__(self)
        self.num_weights = 1
        self.weight_indices = torch.tensor([0], device=device)

        self.weights = nn.Parameter(torch.tensor([self.init_val] * self.num_weights, device=device))

class HS(Ours, nn.Module):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        nn.Module.__init__(self)
        self.num_weights = 1
        self.weight_indices = torch.tensor([1], device=device)

        self.weights = nn.Parameter(torch.tensor([self.init_val] * self.num_weights, device=device))

class Freq_HS(Ours, nn.Module):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        nn.Module.__init__(self)
        self.num_weights = 2
        self.weight_indices = torch.tensor([0, 1], device=device)

        self.weights = nn.Parameter(torch.tensor([self.init_val] * self.num_weights, device=device))

class Freq_HS_Pers(Ours, nn.Module):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        nn.Module.__init__(self)
        self.num_weights = 3
        self.weight_indices = torch.tensor([0, 1, 2], device=device)

        self.weights = nn.Parameter(torch.tensor([self.init_val] * self.num_weights, device=device))

class WeightedHS(Ours, nn.Module):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        nn.Module.__init__(self)
        self.num_weights = self.num_features
        self.weights = nn.Parameter(torch.tensor([self.init_val] * self.num_weights, device=device))
        self.onlyforgroup = True

    def get_nll(self, seq, weightsfromarg=None):
        if weightsfromarg is not None:
            weightstouse = weightsfromarg
        else:
            weightstouse = self.weights

        nll = 0
        prev_feats = torch.stack([torch.tensor(self.features[r], dtype=torch.int8, device=device) for r in seq[1:-1]]).to(device=device)    # Shape: (L, D) where L = len(seq) - 2
        all_feats = torch.stack([torch.tensor(self.features[r], dtype=torch.int8, device=device) for r in self.unique_responses])           # Shape: (N, D) where N = num unique responses
        all_diffs = (all_feats.unsqueeze(0) == prev_feats.unsqueeze(1)).float()                                                             # Output: (L, N, D)
        
        mask = np.ones((len(seq) - 2, len(self.unique_responses)))
        for i in range(2, len(seq)):
            visited_responses = np.array([self.unique_response_to_index[resp] for resp in seq[:i]])
            mask[i - 2, visited_responses] = 0

        logits = torch.einsum("lnd,d->ln", all_diffs, weightstouse)                                                                        # Shape: (L, N)
        
        if self.config["mask"]:
            mask = torch.tensor(mask, dtype=torch.bool, device=device)
            logits[mask == 0] = float('-inf')
        log_probs = F.log_softmax(logits, dim=1)

        if weightsfromarg is not None:
            return log_probs

        targets = torch.tensor([self.unique_response_to_index[r] for r in seq], device=device)
        nll = F.nll_loss(log_probs, targets[2:], reduction='sum')
        return nll
    
class FreqWeightedHS(Ours, nn.Module):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        nn.Module.__init__(self)
        self.num_weights = self.num_features + 1
        self.weights = nn.Parameter(torch.tensor([self.init_val] * self.num_weights, device=device))
        self.onlyforgroup = True

    def get_nll(self, seq, weightsfromarg=None):
        if weightsfromarg is not None:
            weightstouse = weightsfromarg
        else:
            weightstouse = self.weights

        nll = 0
        prev_feats = torch.stack([torch.tensor(self.features[r], dtype=torch.int8, device=device) for r in seq[1:-1]]).to(device=device)    # Shape: (L, D) where L = len(seq) - 2
        all_feats = torch.stack([torch.tensor(self.features[r], dtype=torch.int8, device=device) for r in self.unique_responses])           # Shape: (N, D) where N = num unique responses
        all_diffs = (all_feats.unsqueeze(0) == prev_feats.unsqueeze(1)).float()                                                             # Output: (L, N, D)
        
        mask = np.ones((len(seq) - 2, len(self.unique_responses)))
        for i in range(2, len(seq)):
            visited_responses = np.array([self.unique_response_to_index[resp] for resp in seq[:i]])
            mask[i - 2, visited_responses] = 0

        logits = weightstouse[0] * self.d2ts(self.freq).unsqueeze(0) + torch.einsum("lnd,d->ln", all_diffs, weightstouse[1:])                                                                        # Shape: (L, N)
        
        if self.config["mask"]:
            mask = torch.tensor(mask, dtype=torch.bool, device=device)
            logits[mask == 0] = float('-inf')
        log_probs = F.log_softmax(logits, dim=1)
        
        if weightsfromarg is not None:
            return log_probs
                                                                                                                                            # weights: (D,)
        targets = torch.tensor([self.unique_response_to_index[r] for r in seq], device=device)
        nll = F.nll_loss(log_probs, targets[2:], reduction='sum')
        return nll

class Activity(Ours, nn.Module):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        nn.Module.__init__(self)
        self.num_weights = self.num_features
        self.weights = nn.Parameter(torch.tensor([self.init_val] * self.num_weights, device=device))
        self.onlyforgroup = True

    def get_nll(self, seq, weightsfromarg=None):
        if weightsfromarg is not None:
            weightstouse = weightsfromarg
        else:
            weightstouse = self.weights

        nll = 0
        prev_feats = torch.stack([torch.tensor(self.features[r], dtype=torch.int8, device=device) for r in seq[1:-1]]).to(device=device)    # Shape: (L, D) where L = len(seq) - 2
        all_feats = torch.stack([torch.tensor(self.features[r], dtype=torch.int8, device=device) for r in self.unique_responses])           # Shape: (N, D) where N = num unique responses
        all_diffs = (all_feats.unsqueeze(0) == prev_feats.unsqueeze(1)).float()                                                             # Output: (L, N, D)
        
        mask = np.ones((len(seq) - 2, len(self.unique_responses)))
        for i in range(2, len(seq)):
            visited_responses = np.array([self.unique_response_to_index[resp] for resp in seq[:i]])
            mask[i - 2, visited_responses] = 0

        logits = torch.stack([torch.einsum("nd,d->n", all_feats.to(torch.float), weightstouse) for _ in range(len(seq) - 2)], dim=0)         # l = len(seq) - 2, n = num_unique_responses                                            # Shape: (L, N)

        if self.config["mask"]:
            mask = torch.tensor(mask, dtype=torch.bool, device=device)
            logits[mask == 0] = float('-inf')
        log_probs = F.log_softmax(logits, dim=1)

        if weightsfromarg is not None:
            return log_probs

        targets = torch.tensor([self.unique_response_to_index[r] for r in seq], device=device)
        nll = F.nll_loss(log_probs, targets[2:], reduction='sum')
        return nll

class WeightedHSActivity(Ours, nn.Module):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        nn.Module.__init__(self)
        self.num_weights = self.num_features * 2
        self.weights = nn.Parameter(torch.tensor([self.init_val] * self.num_weights, device=device))
        self.onlyforgroup = True

    def get_nll(self, seq, weightsfromarg=None):
        if weightsfromarg is not None:
            weightstouse = weightsfromarg
        else:
            weightstouse = self.weights

        nll = 0
        prev_feats = torch.stack([torch.tensor(self.features[r], dtype=torch.int8, device=device) for r in seq[1:-1]]).to(device=device)    # Shape: (L, D) where L = len(seq) - 2
        all_feats = torch.stack([torch.tensor(self.features[r], dtype=torch.int8, device=device) for r in self.unique_responses])           # Shape: (N, D) where N = num unique responses
        all_diffs = (all_feats.unsqueeze(0) == prev_feats.unsqueeze(1)).float()                                                             # Output: (L, N, D)
        
        mask = np.ones((len(seq) - 2, len(self.unique_responses)))
        for i in range(2, len(seq)):
            visited_responses = np.array([self.unique_response_to_index[resp] for resp in seq[:i]])
            mask[i - 2, visited_responses] = 0

        logits = torch.einsum("lnd,d->ln", all_diffs, weightstouse[:self.num_weights//2]) + torch.einsum("nd,d->n", all_feats.to(torch.float), weightstouse[self.num_weights//2:]).unsqueeze(0)         # l = len(seq) - 2, n = num_unique_responses                                            # Shape: (L, N)
        
        if self.config["mask"]:
            mask = torch.tensor(mask, dtype=torch.bool, device=device)
            logits[mask == 0] = float('-inf')
        log_probs = F.log_softmax(logits, dim=1)

        if weightsfromarg is not None:
            return log_probs

        targets = torch.tensor([self.unique_response_to_index[r] for r in seq], device=device)
        nll = F.nll_loss(log_probs, targets[2:], reduction='sum')
        return nll

class FreqWeightedHSActivity(Ours, nn.Module):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        nn.Module.__init__(self)
        self.num_weights = self.num_features * 2 + 1
        self.weights = nn.Parameter(torch.tensor([self.init_val] * self.num_weights, device=device))
        self.onlyforgroup = True

    def get_nll(self, seq, weightsfromarg=None, getnll=None):
        if weightsfromarg is not None:
            weightstouse = weightsfromarg
        else:
            weightstouse = self.weights

        nll = 0
        prev_feats = torch.stack([torch.tensor(self.features[r], dtype=torch.int8, device=device) for r in seq[1:-1]]).to(device=device)    # Shape: (L, D) where L = len(seq) - 2
        all_feats = torch.stack([torch.tensor(self.features[r], dtype=torch.int8, device=device) for r in self.unique_responses])           # Shape: (N, D) where N = num unique responses
        all_diffs = (all_feats.unsqueeze(0) == prev_feats.unsqueeze(1)).float()                                                             # Output: (L, N, D)
        
        mask = np.ones((len(seq) - 2, len(self.unique_responses)))
        for i in range(2, len(seq)):
            visited_responses = np.array([self.unique_response_to_index[resp] for resp in seq[:i]])
            mask[i - 2, visited_responses] = 0

        logits = weightstouse[0] * self.d2ts(self.freq).unsqueeze(0) + torch.einsum("lnd,d->ln", all_diffs, weightstouse[1:1 + self.num_weights//2]) + torch.einsum("nd,d->n", all_feats.to(torch.float), weightstouse[1 + self.num_weights//2:]).unsqueeze(0)         # l = len(seq) - 2, n = num_unique_responses                                            # Shape: (L, N)
        
        if self.config["mask"]:
            mask = torch.tensor(mask, dtype=torch.bool, device=device)
            logits[mask == 0] = float('-inf')
        log_probs = F.log_softmax(logits, dim=1)

        if weightsfromarg is not None and getnll is None:
            return log_probs

        targets = torch.tensor([self.unique_response_to_index[r] for r in seq], device=device)
        nll = F.nll_loss(log_probs, targets[2:], reduction='sum')

        # # Add regularization
        # if forloss:
        #     if self.config["reglambda"] > 0:
        #         reg_term = torch.sum(torch.abs(weightstouse))             # L1
        #         # reg_term = torch.sum(weightstouse ** 2)                 # L2
        #         nll = nll + self.config["reglambda"] * reg_term

        return nll
        
    def get_nll_withoutmasking(self, seq, weightsfromarg=None, getnll=None):
        if weightsfromarg is not None:
            weightstouse = weightsfromarg
        else:
            weightstouse = self.weights

        nll = 0
        prev_feats = torch.stack([torch.tensor(self.features[r], dtype=torch.int8, device=device) for r in seq[1:-1]]).to(device=device)    # Shape: (L, D) where L = len(seq) - 2
        all_feats = torch.stack([torch.tensor(self.features[r], dtype=torch.int8, device=device) for r in self.unique_responses])           # Shape: (N, D) where N = num unique responses
        all_diffs = (all_feats.unsqueeze(0) == prev_feats.unsqueeze(1)).float()                                                             # Output: (L, N, D)

        freq_logits = weightstouse[0] * self.d2ts(self.freq).unsqueeze(0)
        HS_logits = torch.einsum("lnd,d->ln", all_diffs, weightstouse[1:1 + self.num_weights//2])
        Activity_logits = torch.einsum("nd,d->n", all_feats.to(torch.float), weightstouse[1 + self.num_weights//2:]).unsqueeze(0)
        logits = freq_logits + HS_logits + Activity_logits         # l = len(seq) - 2, n = num_unique_responses                                            # Shape: (L, N)
        log_probs = F.log_softmax(logits, dim=1)

        targets = torch.tensor([self.unique_response_to_index[r] for r in seq], device=device)

        if weightsfromarg is not None:
            return log_probs, -F.nll_loss(log_probs, targets[2:], reduction='none'), freq_logits[torch.arange(freq_logits.size(0)), targets[2:]], HS_logits[torch.arange(HS_logits.size(0)), targets[2:]], Activity_logits[torch.arange(Activity_logits.size(0)), targets[2:]]

        else:
            nll = F.nll_loss(log_probs, targets[2:], reduction='sum')
            return nll
    
    def get_logits_maxlogits(self, seq, weightsfromarg=None, getnll=None):
        if weightsfromarg is not None:
            weightstouse = weightsfromarg
        else:
            weightstouse = self.weights

        nll = 0
        prev_feats = torch.stack([torch.tensor(self.features[r], dtype=torch.int8, device=device) for r in seq[1:-1]]).to(device=device)    # Shape: (L, D) where L = len(seq) - 2
        all_feats = torch.stack([torch.tensor(self.features[r], dtype=torch.int8, device=device) for r in self.unique_responses])           # Shape: (N, D) where N = num unique responses
        all_diffs = (all_feats.unsqueeze(0) == prev_feats.unsqueeze(1)).float()                                                             # Output: (L, N, D)
        
        mask = np.ones((len(seq) - 2, len(self.unique_responses)))
        for i in range(2, len(seq)):
            visited_responses = np.array([self.unique_response_to_index[resp] for resp in seq[:i]])
            mask[i - 2, visited_responses] = 0

        freq_logits = weightstouse[0] * self.d2ts(self.freq).unsqueeze(0).repeat(len(seq) - 2, 1)
        HS_logits = torch.einsum("lnd,d->ln", all_diffs, weightstouse[1:1 + self.num_weights//2])
        Activity_logits = torch.einsum("nd,d->n", all_feats.to(torch.float), weightstouse[1 + self.num_weights//2:]).unsqueeze(0).repeat(len(seq) - 2, 1)
        global_logits = freq_logits + Activity_logits
        logits = freq_logits + HS_logits + Activity_logits         # l = len(seq) - 2, n = num_unique_responses
        
        if self.config["mask"]:
            mask = torch.tensor(mask, dtype=torch.bool, device=device)
            logits[mask == 0] = float('-inf')
            freq_logits[mask == 0] = float('-inf')
            HS_logits[mask == 0] = float('-inf')
            Activity_logits[mask == 0] = float('-inf')
            global_logits[mask == 0] = float('-inf')

        log_probs = F.log_softmax(logits, dim=1)
        targets = torch.tensor([self.unique_response_to_index[r] for r in seq], device=device)
        nll = F.nll_loss(log_probs, targets[2:], reduction='sum')

        return log_probs, log_probs[torch.arange(log_probs.size(0)), targets[2:]], nll, \
            freq_logits[torch.arange(freq_logits.size(0)), targets[2:]], HS_logits[torch.arange(HS_logits.size(0)), targets[2:]], Activity_logits[torch.arange(Activity_logits.size(0)), targets[2:]], \
            freq_logits[torch.arange(freq_logits.size(0)), freq_logits.argmax(dim=1)], HS_logits[torch.arange(HS_logits.size(0)), HS_logits.argmax(dim=1)], Activity_logits[torch.arange(Activity_logits.size(0)), Activity_logits.argmax(dim=1)], \
            torch.exp(freq_logits[torch.arange(freq_logits.size(0)), targets[2:]]) / torch.exp(freq_logits[torch.arange(freq_logits.size(0)), freq_logits.argmax(dim=1)]), \
            torch.exp(HS_logits[torch.arange(HS_logits.size(0)), targets[2:]]) / torch.exp(HS_logits[torch.arange(HS_logits.size(0)), HS_logits.argmax(dim=1)]), \
            torch.exp(Activity_logits[torch.arange(Activity_logits.size(0)), targets[2:]]) / torch.exp(Activity_logits[torch.arange(Activity_logits.size(0)), Activity_logits.argmax(dim=1)]), \
            torch.exp(global_logits[torch.arange(global_logits.size(0)), targets[2:]]) / torch.exp(global_logits[torch.arange(global_logits.size(0)), global_logits.argmax(dim=1)]), \
            torch.exp(freq_logits[torch.arange(freq_logits.size(0)), targets[2:]]) / torch.exp(freq_logits).sum(dim=1), \
            torch.exp(HS_logits[torch.arange(HS_logits.size(0)), targets[2:]]) / torch.exp(HS_logits).sum(dim=1), \
            torch.exp(Activity_logits[torch.arange(Activity_logits.size(0)), targets[2:]]) / torch.exp(Activity_logits).sum(dim=1), \
            torch.exp(global_logits[torch.arange(global_logits.size(0)), targets[2:]]) / torch.exp(global_logits).sum(dim=1)

class FreqWeightedHSWeightedPers(Ours, nn.Module):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        nn.Module.__init__(self)
        self.num_weights = self.num_features * 2 + 1
        self.weights = nn.Parameter(torch.tensor([self.init_val] * self.num_weights, device=device))
        self.onlyforgroup = True

    def get_nll(self, seq, weightsfromarg=None, getnll=None):
        if weightsfromarg is not None:
            weightstouse = weightsfromarg
        else:
            weightstouse = self.weights

        nll = 0
        prev_prev_feats = torch.stack([torch.tensor(self.features[r], dtype=torch.int8, device=device) for r in seq[:-2]]).to(device=device)
        prev_feats = torch.stack([torch.tensor(self.features[r], dtype=torch.int8, device=device) for r in seq[1:-1]]).to(device=device)             
        prev_diffs = (prev_prev_feats == prev_feats).float()                        
        all_feats = torch.stack([torch.tensor(self.features[r], dtype=torch.int8, device=device) for r in self.unique_responses])           
        all_diffs = (all_feats.unsqueeze(0) == prev_feats.unsqueeze(1)).float()

        mask = np.ones((len(seq) - 2, len(self.unique_responses)))
        for i in range(2, len(seq)):
            visited_responses = np.array([self.unique_response_to_index[resp] for resp in seq[:i]])
            mask[i - 2, visited_responses] = 0

        logits = weightstouse[0] * self.d2ts(self.freq).unsqueeze(0) + torch.einsum("lnd,d->ln", all_diffs, weightstouse[1:1 + self.num_features])         # l = len(seq) - 2, n = num_unique_responses                                            # Shape: (L, N)
        logits_add = torch.einsum('ld,lnd,d->ln', prev_diffs, all_diffs, weightstouse[1 + self.num_features:]) / (prev_diffs.sum(dim=1).unsqueeze(1) + 1e-6)
        # logits_add = torch.einsum('ld,lnd,d->ln', prev_diffs, all_diffs, weightstouse[1 + self.num_features:])
        logits += logits_add

        if self.config["mask"]:
            mask = torch.tensor(mask, dtype=torch.bool, device=device)
            logits[mask == 0] = float('-inf')
        log_probs = F.log_softmax(logits, dim=1)

        if weightsfromarg is not None:
            return log_probs

        targets = torch.tensor([self.unique_response_to_index[r] for r in seq], device=device)
        nll = F.nll_loss(log_probs, targets[2:], reduction='sum')
        return nll

class FreqWeightedHSActivityRecoverable(Ours, nn.Module):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        nn.Module.__init__(self)
        self.num_weights = self.num_features_HS + self.num_features_Act + 1
        self.weights = nn.Parameter(torch.tensor([self.init_val] * self.num_weights, device=device))
        self.onlyforgroup = True

    def get_nll(self, seq, weightsfromarg=None, getnll=None):
        if weightsfromarg is not None:
            weightstouse = weightsfromarg
        else:
            weightstouse = self.weights

        nll = 0
        prev_feats = torch.stack([torch.tensor(self.features_HS[r], dtype=torch.int8, device=device) for r in seq[1:-1]]).to(device=device)    # Shape: (L, D) where L = len(seq) - 2
        all_feats_HS = torch.stack([torch.tensor(self.features_HS[r], dtype=torch.int8, device=device) for r in self.unique_responses])           # Shape: (N, D) where N = num unique responses
        all_diffs = (all_feats_HS.unsqueeze(0) == prev_feats.unsqueeze(1)).float()                                                             # Output: (L, N, D)
        all_feats = torch.stack([torch.tensor(self.features_Act[r], dtype=torch.int8, device=device) for r in self.unique_responses])           # Shape: (N, D) where N = num unique responses

        mask = np.ones((len(seq) - 2, len(self.unique_responses)))
        for i in range(2, len(seq)):
            visited_responses = np.array([self.unique_response_to_index[resp] for resp in seq[:i]])
            mask[i - 2, visited_responses] = 0

        logits = weightstouse[0] * self.d2ts(self.freq).unsqueeze(0) + torch.einsum("lnd,d->ln", all_diffs, weightstouse[1:1 + self.num_features_HS]) + torch.einsum("nd,d->n", all_feats.to(torch.float), weightstouse[1 + self.num_features_HS:]).unsqueeze(0)         # l = len(seq) - 2, n = num_unique_responses                                            # Shape: (L, N)
        
        if self.config["mask"]:
            mask = torch.tensor(mask, dtype=torch.bool, device=device)
            logits[mask == 0] = float('-inf')
        log_probs = F.log_softmax(logits, dim=1)

        if weightsfromarg is not None and getnll is None:
            return log_probs

        targets = torch.tensor([self.unique_response_to_index[r] for r in seq], device=device)
        nll = F.nll_loss(log_probs, targets[2:], reduction='sum')
        return nll

class FreqWeightedHSActivityWeightedPers(Ours, nn.Module):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        nn.Module.__init__(self)
        self.num_weights = self.num_features * 3 + 1
        self.weights = nn.Parameter(torch.tensor([self.init_val] * self.num_weights, device=device))
        self.onlyforgroup = True

    def get_nll(self, seq, weightsfromarg=None, getnll=None):
        if weightsfromarg is not None:
            weightstouse = weightsfromarg
        else:
            weightstouse = self.weights

        nll = 0
        prev_prev_feats = torch.stack([torch.tensor(self.features[r], dtype=torch.int8, device=device) for r in seq[:-2]]).to(device=device)
        prev_feats = torch.stack([torch.tensor(self.features[r], dtype=torch.int8, device=device) for r in seq[1:-1]]).to(device=device)             
        prev_diffs = (prev_prev_feats == prev_feats).float()                        
        all_feats = torch.stack([torch.tensor(self.features[r], dtype=torch.int8, device=device) for r in self.unique_responses])           
        all_diffs = (all_feats.unsqueeze(0) == prev_feats.unsqueeze(1)).float()

        mask = np.ones((len(seq) - 2, len(self.unique_responses)))
        for i in range(2, len(seq)):
            visited_responses = np.array([self.unique_response_to_index[resp] for resp in seq[:i]])
            mask[i - 2, visited_responses] = 0

        logits = weightstouse[0] * self.d2ts(self.freq).unsqueeze(0) + torch.einsum("lnd,d->ln", all_diffs, weightstouse[1:1 + self.num_features]) +   torch.einsum("nd,d->n", all_feats.to(torch.float), weightstouse[1 + self.num_features: 1 + 2*self.num_features]).unsqueeze(0)         # l = len(seq) - 2, n = num_unique_responses                                            # Shape: (L, N)
        logits_add = torch.einsum('ld,lnd,d->ln', prev_diffs, all_diffs, weightstouse[1 + 2*self.num_features:]) / (prev_diffs.sum(dim=1).unsqueeze(1) + 1e-6)
        # logits_add = torch.einsum('ld,lnd,d->ln', prev_diffs, all_diffs, weightstouse[1 + 2*self.num_features:])
        logits += logits_add

        if self.config["mask"]:
            mask = torch.tensor(mask, dtype=torch.bool, device=device)
            logits[mask == 0] = float('-inf')
        log_probs = F.log_softmax(logits, dim=1)

        if weightsfromarg is not None:
            return log_probs

        targets = torch.tensor([self.unique_response_to_index[r] for r in seq], device=device)
        nll = F.nll_loss(log_probs, targets[2:], reduction='sum')
        return nll
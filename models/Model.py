"""Core model utilities, data loading, and shared fitting helpers."""

import os
from sympy import sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import CLIPTextModelWithProjection, AutoTokenizer
from sentence_transformers import SentenceTransformer
import warnings
warnings.simplefilter("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from scipy.optimize import minimize
from tqdm import tqdm
from pybads import BADS
import pymc as pm
import arviz as az
import torch
from torch.optim import LBFGS
from torch.autograd.functional import hessian
import requests
import math
import json
from sklearn.model_selection import train_test_split, KFold
from numba import njit
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))
from utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch
import torch.nn as nn
import requests
from scipy.stats import pearsonr, spearmanr
import time
from collections import Counter
from scipy.stats import ttest_ind
import pickle as pk
from model2vec import StaticModel
from concurrent.futures import ThreadPoolExecutor
import torch.multiprocessing as mp
import copy

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Model:
    def __init__(self, config):
        self.config = config
        with open("../files/modelstorun.json", 'r') as f:
            self.modelstorun = json.load(f)
        self.data = pd.read_csv("../csvs/" + self.config["dataset"] + ".csv")
        
        # Normalize responses and drop invalid/NA entries.
        self.data = self.data[~self.data["response"].isin(["mammal", "bacterium", "unicorn", "woollymammoth"])]     # filtering NA responses
        with open("../files/response_corrections.json", 'r') as f:
            self.corrections = json.load(f)
        self.data["response"] = self.data["response"].map(lambda x: self.corrections.get(x, x))                     # correcting spaces in spelling
        try:
            self.data = self.data[~(self.data["invalid"] == 1)]
        except:
            pass

        if config["usehillsresp"]:
            self.unique_responses = sorted([resp.lower() for resp in self.data["response"].unique()])  # 354 unique animals
        else:                               # useallresp
            self.unique_responses = set()
            csv_dir = "../csvs/"
            for file in os.listdir(csv_dir):
                if file.endswith(".csv"):
                    df = pd.read_csv(os.path.join(csv_dir, file), usecols=["response", "invalid"] if "invalid" in pd.read_csv(os.path.join(csv_dir, file), nrows=1).columns else ["response"])                
                    if "invalid" in df.columns:
                        df = df[df["invalid"] != 1]
                    corrected_responses = (
                        df["response"]
                        .map(lambda x: self.corrections.get(x, x))
                        .str.lower()
                        .dropna()
                        .unique()
                    )
                    self.unique_responses.update(corrected_responses)
            self.unique_responses = list(self.unique_responses)
        
        self.unique_response_to_index = dict(zip(self.unique_responses, np.arange(len(self.unique_responses))))

        if config["useapifreq"]: 
            # Frequency cues from external n-gram data.
            self.freq = self.get_frequencies()       # normalising is bad for performance when log freqs
            for k, v in self.freq.items():
                if pd.isna(v):
                    print(k)
            self.freq2 = self.get_frequencies_hills()
            common_keys = set(self.freq) & set(self.freq2)
            values1 = [self.freq[k] for k in common_keys]
            values2 = [self.freq2[k] for k in common_keys]
            # print(pearsonr(values1, values2))          # 0.37
            # print(spearmanr(values1, values2))         # 0.83
        elif config["dataset"] == "hills":      # ie --usehillsfreq
            self.freq = self.get_frequencies_hills()
        
        # Precompute embeddings and similarity matrices for all responses.
        self.embeddings = self.get_embeddings()
        self.num_embedding_dims = len(next(iter(self.embeddings.values())))
        self.sim_mat = self.get_embedding_sim_mat()

        self.data_unique_responses = sorted([resp.lower() for resp in self.data["response"].unique()])  # 354 unique animals
        if self.config["dataset"] == "hills":
            self.response_to_category, self.num_categories = self.get_categories()

        self.sequences = self.data.groupby("pid").agg(list)["response"].tolist()
        self.data_metrics = {"patchnum": self.data.groupby("pid").agg(list)["fpatchnum"].tolist(), "numwithinpatch": self.data.groupby("pid").agg(list)["fpatchitem"].tolist(), "switchornot": self.data.groupby("pid").agg(list)["flastitem"].tolist()}
        self.num_sequences = len(self.sequences)
        self.sequence_lengths = [len(s) for s in self.sequences]
        try:
            self.RTs = self.data.groupby("pid").agg(list)["RT"].tolist()
        except:
            self.RTs = []
        
        # Precompute CV splits for group fitting.
        self.splits = self.split_sequences(self.sequences.copy())     # perform CV, only used in group fitting

        self.start = 2
        self.init_val = self.config["initval"]
        self.numsubsamples = 3
        self.dynamic = False
        self.dynamic_cat = False
        self.sim_drop = False

        self.suffix = ""
        self.custom_splits = None
         
    def d2ts(self, some_dict):
        """Convert a response-indexed dict into a tensor aligned to unique_responses."""
        return torch.tensor([some_dict[resp] for resp in self.unique_responses], dtype=torch.float32, device=device)

    def np2ts(self, some_np):
        """Convert a numpy array into a torch tensor on the active device."""
        return torch.tensor(some_np, dtype=torch.float32, device=device)

    def d2np(self, some_dict):
        """Convert a response-indexed dict into a numpy array."""
        return np.array([some_dict[resp] for resp in self.unique_responses])

    def get_frequencies(self):
        # https://stackoverflow.com/questions/74951626/python-nlp-google-ngram-api
        if os.path.exists("../files/freq_abs_log.json"):
            with open("../files/freq_abs_log.json", "r") as f:
                freq_abs = json.load(f)
            with open("../files/freq_rel_log.json", "r") as f:
                freq_rel = json.load(f)
            freq_abs = {k: v for k, v in freq_abs.items() if k in self.unique_responses}
            freq_rel = {k: v for k, v in freq_rel.items() if k in self.unique_responses}
        else:
            freq_abs = {}
            freq_rel = {}

        remaining = [resp for resp in self.unique_responses if resp not in freq_abs]

        if not remaining:
            return freq_abs

        chunk_size = 100
        total_chunks = math.ceil(len(remaining) / chunk_size)
        url = 'https://api.ngrams.dev/eng/batch'
        headers = {'Content-Type': 'application/json'}

        for i in range(total_chunks):
            chunk = remaining[i * chunk_size:(i + 1) * chunk_size]
            payload = {
                "flags": "cr",
                "queries": chunk
            }

            # Batch query the n-grams API for frequency counts.
            response = requests.post(url, headers=headers, json=payload)
            print(f"Batch {i + 1}/{total_chunks} | Status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                abs_match_counts = []
                rel_match_counts = []

                for res in data.get("results", []):
                    ngrams = res.get("ngrams", [])
                    count_abs = ngrams[0].get("absTotalMatchCount", 0) if ngrams else 0
                    count_rel = ngrams[0].get("relTotalMatchCount", 0) if ngrams else 0
                    abs_match_counts.append(np.log10(count_abs))
                    rel_match_counts.append(np.log10(count_rel))

                freq_abs.update(dict(zip(chunk, abs_match_counts)))
                freq_rel.update(dict(zip(chunk, rel_match_counts)))

            else:
                print("ERROR!!!!")

        freq_abs = dict(sorted(freq_abs.items(), key=lambda item: item[1], reverse=True))
        freq_rel = dict(sorted(freq_rel.items(), key=lambda item: item[1], reverse=True))
        # Cache results to avoid repeated API calls.
        with open("../files/freq_abs_log.json", "w") as f:
            json.dump(freq_abs, f, indent=2)
        with open("../files/freq_rel_log.json", "w") as f:
            json.dump(freq_rel, f, indent=2)

        return freq_abs

    def get_frequencies_hills(self):
        file_path = '../files/datafreqlistlog.txt'
        frequencies = {}
        with open(file_path, 'r') as file:
            for line in file:
                key, value = line.strip().split(',')
                key = self.corrections.get(key, key)  # Correcting the spelling
                if key in self.unique_responses:
                    frequencies[key] = float(value)
        return frequencies

    def get_embeddings(self): 
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.config["representation"] == "clip":
            model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14", local_files_only=True).to(device)
            tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14", local_files_only=True)
            inputs = tokenizer(self.unique_responses, padding=True, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                embeddings = outputs.text_embeds
            embeddings = embeddings.detach().cpu().numpy()
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)


        if self.config["representation"] == "minilm":
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device, local_files_only=True)
            embeddings = model.encode(self.unique_responses, normalize_embeddings=True)
        

        if self.config["representation"] == "potion_256":
            model = StaticModel.from_pretrained("/home/snath/.cache/huggingface/hub/models--minishlab--potion-base-8M/snapshots/3f12147e8ef0407df4e55de29669f79c11c8e2de/") # "minishlab/potion-base-8M"
            embeddings = model.encode(self.unique_responses, normalize=True)
        
        if self.config["representation"] == "potion_128":
            model = StaticModel.from_pretrained("/home/snath/.cache/huggingface/hub/models--minishlab--potion-base-4M/snapshots/2a69a3f1da737bad10c46a997aab90bb07509d8c/") # "minishlab/potion-base-4M"
            embeddings = model.encode(self.unique_responses, normalize=True)
        
        if self.config["representation"] == "potion_64":
            model = StaticModel.from_pretrained("/home/snath/.cache/huggingface/hub/models--minishlab--potion-base-2M/snapshots/5da9753bbbe6c6e916cffca02770f9e3f2636e56/") # "minishlab/potion-base-2M"
            embeddings = model.encode(self.unique_responses, normalize=True)
        

        if self.config["representation"] == "qwen":     #1024
            model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device=device, local_files_only=True)
            embeddings = model.encode(self.unique_responses, normalize_embeddings=True)
        

        if self.config["representation"] == "bgesmall":  #512
            model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device, local_files_only=True)
            embeddings = model.encode(self.unique_responses, normalize_embeddings=True)
        
        if self.config["representation"] == "bgebase":  #768
            model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device, local_files_only=True)
            embeddings = model.encode(self.unique_responses, normalize_embeddings=True)
        
        if self.config["representation"] == "bgelarge":  #1024
            model = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device, local_files_only=True)
            embeddings = model.encode(self.unique_responses, normalize_embeddings=True)


        if self.config["representation"] == "e5small":  #384
            model = SentenceTransformer("intfloat/e5-small", device=device, local_files_only=True)
            embeddings = model.encode(self.unique_responses, normalize_embeddings=True)
        
        if self.config["representation"] == "e5base":  #768
            model = SentenceTransformer("intfloat/e5-base", device=device, local_files_only=True)
            embeddings = model.encode(self.unique_responses, normalize_embeddings=True)
        
        if self.config["representation"] == "e5large":  #1024
            model = SentenceTransformer("intfloat/e5-large", device=device, local_files_only=True)
            embeddings = model.encode(self.unique_responses, normalize_embeddings=True)
        

        if self.config["representation"] == "infly":  #1536
            model = SentenceTransformer("infly/inf-retriever-v1-1.5b", trust_remote_code=True, device=device, local_files_only=True)
            embeddings = model.encode(self.unique_responses, normalize_embeddings=True)
        
        
        if self.config["representation"] == "rubert":  #312
            model = SentenceTransformer("cointegrated/rubert-tiny2", device=device, local_files_only=True)
            embeddings = model.encode(self.unique_responses, normalize_embeddings=True)
        

        if self.config["representation"] == "gtelarge": #1024
            model = SentenceTransformer("thenlper/gte-large", device=device, local_files_only=True)
            embeddings = model.encode(self.unique_responses, normalize_embeddings=True)

        if self.config["representation"] == "gtebert":  #768
            model = SentenceTransformer("Alibaba-NLP/gte-modernbert-base", device=device, local_files_only=True)
            embeddings = model.encode(self.unique_responses, normalize_embeddings=True)
        
        return dict(zip(self.unique_responses, embeddings))

    def get_embedding_sim_mat(self):
        responses = list(self.unique_responses)
        embeddings_matrix = np.stack([self.embeddings[resp].astype(np.float64) for resp in responses])
        similarity = np.dot(embeddings_matrix, embeddings_matrix.T)
        sim_matrix = {
            responses[i]: {responses[j]: similarity[i, j] for j in range(len(responses))}
            for i in range(len(responses))
        }
        return sim_matrix

    def get_categories(self):
        category_info_path = "../files/Final_Categories_and_Exemplars.xlsx"
        category_name_to_num = (pd.read_excel(category_info_path).reset_index().set_index("Category").to_dict())["index"]

        examples = pd.read_excel(
            category_info_path,
            sheet_name="Exemplars",
        )
        examples["Exemplar"] = examples["Exemplar"].map(lambda x: self.corrections.get(x, x))
        examples["category"] = (
            examples["Category"].map(category_name_to_num).astype("Int64")
        )
        num_categories = examples["category"].nunique()

        examples = (
            examples.groupby("Exemplar")["category"].agg(list).reset_index()
        )  # account for multi-class
        examples_to_category = examples.set_index("Exemplar").to_dict()["category"]

        self.data["categories"] = self.data["response"].map(examples_to_category)

        for item in self.data_unique_responses:
            if item not in examples_to_category:
                print(item)
        assert all(item in examples_to_category for item in self.data_unique_responses)
        return examples_to_category, num_categories

    def split_sequences(self, sequencestosplit):
        np.random.shuffle(sequencestosplit)

        if self.config['cv'] == 1:
            train_seqs, test_seqs = train_test_split(sequencestosplit, test_size=0.2, shuffle=False)
            return [(train_seqs, test_seqs)]
        
        elif self.config['cv'] > 1:
            kf = KFold(n_splits=self.config['cv'], shuffle=False)
            splits = []
            for train_idx, test_idx in kf.split(sequencestosplit):
                train_seqs = [sequencestosplit[i] for i in train_idx]
                test_seqs = [sequencestosplit[i] for i in test_idx]
                splits.append((train_seqs, test_seqs))
            return splits
    
    def fit(self, customsequences=None, folderinfits="model_fits"):
        if customsequences is None:
            splitstofit = self.splits
        else:
            splitstofit = self.custom_splits
        
        # refnll = self.config["refnll"].lower()
        # model = nn.DataParallel(self).to('cuda:0')
        # if model.module.num_weights > 0:
        #     optimizer = torch.optim.LBFGS(model.module.parameters(), lr=self.config["lr"], max_iter=self.config["maxiter"], tolerance_grad=self.config["tol"], tolerance_change=self.config["tol"])
            
        self.results = {}

        weights_list = []
        train_nlls = np.zeros(len(splitstofit))
        test_nlls = np.zeros(len(splitstofit))

        init_state = copy.deepcopy(self.state_dict())       # added

        for split_ind, (train_sequences, test_sequences) in enumerate(splitstofit):
            # Added:
            self.load_state_dict(init_state)
            model = nn.DataParallel(self).to('cuda:0')
            if model.module.num_weights > 0:
                optimizer = torch.optim.LBFGS(model.module.parameters(), lr=self.config["lr"], max_iter=self.config["maxiter"], tolerance_grad=self.config["tol"], tolerance_change=self.config["tol"])
            ####

            self.split_ind = split_ind
            lbfgs_iters = 0
            loss_history = []
            param_history = []
            def closure():
                nonlocal lbfgs_iters
                optimizer.zero_grad()
                loss = torch.stack([model.module.get_nll(seq) for seq in train_sequences]).sum()
                if self.config["reglambda"] > 0:
                    if self.config["regtype"] == "l1":
                        reg_term = self.config["reglambda"] * torch.sum(torch.abs(model.module.weights))
                    elif self.config["regtype"] == "l2":
                        reg_term = self.config["reglambda"] * torch.sum(model.module.weights ** 2)
                    else:
                        reg_term = torch.tensor(0.0, device=loss.device)
                        print("No regularisation added")
                    loss = loss + reg_term
                loss.backward()
                loss_history.append(loss.item())
                param_history.append(model.module.weights.detach().cpu().clone())
                lbfgs_iters += 1
                return loss

            if model.module.num_weights > 0:
                optimizer.step(closure)
                print(f"LBFGS iterations run: {lbfgs_iters}")
                fittedweights = model.module.weights.detach().clone()
                self.results[f"weights_fold{split_ind + 1}{self.suffix}"] = fittedweights
                weights_list.append(fittedweights)

            with torch.no_grad():
                # trainnll = sum([model.module.get_nll(seq) for seq in train_sequences])
                # testnll = sum([model.module.get_nll(seq) for seq in test_sequences])
                trainnll = torch.stack([model.module.get_nll(seq) for seq in train_sequences]).sum().item()
                # testnll = torch.stack([model.module.get_nll(seq) for seq in test_sequences]).sum().item()
                if len(test_sequences) > 0:
                    testnll = torch.stack([model.module.get_nll(seq) for seq in test_sequences]).sum().item()
                else:
                    testnll = 0.0

                # if refnll == model.module.__class__.__name__.lower():
                #     train_ref_nlls.append(trainnll)
                #     test_ref_nlls.append(testnll)
                #     train_nlls[split_ind] = trainnll
                #     test_nlls[split_ind] = testnll
                # else:
                #     try:
                #         train_nlls[split_ind] = trainnll - train_ref_nlls[split_ind]
                #         test_nlls[split_ind] = testnll - test_ref_nlls[split_ind]
                #     except:
                train_nlls[split_ind] = trainnll
                test_nlls[split_ind] = testnll

        self.results[f"trainNLLs{self.suffix}"] = train_nlls
        self.results[f"mean_trainNLL{self.suffix}"] = np.mean(train_nlls)
        self.results[f"std_trainNLL{self.suffix}"] = np.std(train_nlls)
        self.results[f"se_trainNLL{self.suffix}"] = np.std(train_nlls) / np.sqrt(len(splitstofit))

        self.results[f"testNLLs{self.suffix}"] = test_nlls
        self.results[f"mean_testNLL{self.suffix}"] = np.mean(test_nlls)
        self.results[f"std_testNLL{self.suffix}"] = np.std(test_nlls)
        self.results[f"se_testNLL{self.suffix}"] = np.std(test_nlls) / np.sqrt(len(splitstofit))

        if model.module.num_weights > 0:
            self.results[f"weights{self.suffix}"] = weights_list
            self.results[f"mean_weights{self.suffix}"] = torch.mean(torch.stack(weights_list), dim=0)

        if self.config["print"]:
            print(f"Mean +- SE trainNLL over {self.config['cv']} fold(s)", self.results[f"mean_trainNLL{self.suffix}"], "+-", self.results[f"se_trainNLL{self.suffix}"])
            print(f"Sum testNLL over {self.config['cv']} fold(s)", sum(self.results[f"testNLLs{self.suffix}"]))
            if model.module.num_weights > 0:
                print(f"weights for each {self.config['cv']} fold", self.results[f"weights{self.suffix}"])

        if self.config["save"]:
            pk.dump(self.results, open(f"../fits/{folderinfits}/{model.module.__class__.__name__.lower()}_fits_{self.config["featurestouse"]}{self.suffix}.pk", "wb"))

    def simulate(self, customsequences=None, folderinsimulations="model_simulations"):
        if customsequences is None:
            splitstofit = self.splits
        else:
            splitstofit = self.custom_splits

        try:
            results = self.results
        except:
            print("Loading fit from file...")
            results = pk.load(open(f"../fits/model_fits/{self.__class__.__name__.lower()}_fits_{self.config["featurestouse"]}{self.suffix}.pk", "rb"))
        self.simulations = []
        self.bleus = []
        print(self.__class__.__name__)
        for split_ind, (train_seqs, test_seqs) in enumerate(splitstofit):
            for _ in range(self.numsubsamples):
                forbleu = []
                for i in range(len(test_seqs)):
                    simulated_sequence = [test_seqs[i][0], test_seqs[i][1]]
                    for l in range(len(test_seqs[i]) - 2):
                        candidates = list(set(self.unique_responses) - set(simulated_sequence))
                        if self.__class__.__name__ == "Random":
                            prob_dist = torch.ones(len(self.unique_responses))
                        else:
                            ll = self.get_nll(simulated_sequence[-2:] + [""], results[f"weights_fold{split_ind + 1}"]).squeeze(0)
                            prob_dist = torch.exp(ll)
                        inds = [self.unique_response_to_index[c] for c in candidates]
                        prob_dist = prob_dist[inds]
                        prob_dist /= prob_dist.sum()
                        indices = torch.multinomial(prob_dist, 1, replacement=True)
                        next_response = candidates[indices]
                        simulated_sequence.append(next_response)
                    self.simulations.append(simulated_sequence)
                    forbleu.append(simulated_sequence)
                self.bleus.append(calculate_bleu([sim[2:] for sim in forbleu], [seq[2:] for seq in test_seqs]))
        print("SIM BLEUS MEAN:", {k: sum(d[k] for d in self.bleus) / len(self.bleus) for k in self.bleus[0]})

        if self.config["save"]:
            pk.dump(self.simulations, open(f"../simulations/{folderinsimulations}/{self.__class__.__name__.lower()}_simulations_{self.config["featurestouse"]}{self.suffix}.pk", "wb"))

        if self.config["print"]:
            print(self.model_class, "simulations..................")
            print('\n'.join(['\t  '.join(map(str, row)) for row in self.simulations[:3]]))
    
    def simulateweights(self, weights, folderinsimulations="parameter_recovery"):
        self.simulations = []
        self.bleus = []
        print(self.__class__.__name__)
        for _ in range(self.numsubsamples):
            forbleu = []
            for i in range(len(self.sequences)):
                simulated_sequence = [self.sequences[i][0], self.sequences[i][1]]
                for l in range(len(self.sequences[i]) - 2):
                    candidates = list(set(self.unique_responses) - set(simulated_sequence))
                    ll = self.get_nll(simulated_sequence[-2:] + [""], weights).squeeze(0)
                    prob_dist = torch.exp(ll)
                    inds = [self.unique_response_to_index[c] for c in candidates]
                    prob_dist = prob_dist[inds]
                    prob_dist /= prob_dist.sum()
                    indices = torch.multinomial(prob_dist, 1, replacement=True)
                    next_response = candidates[indices]
                    simulated_sequence.append(next_response)
                self.simulations.append(simulated_sequence)
                forbleu.append(simulated_sequence)
            self.bleus.append(calculate_bleu([sim[2:] for sim in forbleu], [seq[2:] for seq in self.sequences]))
        print("SIM BLEUS MEAN:", {k: sum(d[k] for d in self.bleus) / len(self.bleus) for k in self.bleus[0]})

        if self.config["save"]:
            pk.dump(self.simulations, open(f"../simulations/{folderinsimulations}/{self.__class__.__name__.lower()}_simulations_{self.config["featurestouse"]}{self.suffix}.pk", "wb"))

        if self.config["print"]:
            print(self.model_class, "simulations..................")
            print('\n'.join(['\t  '.join(map(str, row)) for row in self.simulations[:3]]))

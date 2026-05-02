"""Shared utilities for modeling, plotting, and preprocessing."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import CLIPTextModelWithProjection, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from collections import defaultdict
import warnings
import time
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
import argparse
import pickle as pk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Functions
def calculate_bleu(generated_sequences, real_sequences):
    """Compute BLEU-1..4 scores averaged across generated sequences."""
    scores = []
    for gen_seq in generated_sequences:
        score1 = sentence_bleu(real_sequences, gen_seq, weights=(1, 0, 0, 0))
        score2 = sentence_bleu(real_sequences, gen_seq, weights=(0, 1, 0, 0))
        score3 = sentence_bleu(real_sequences, gen_seq, weights=(0, 0, 1, 0))
        score4 = sentence_bleu(real_sequences, gen_seq, weights=(0, 0, 0, 1))
        scores.append([score1, score2, score3, score4])
    return dict(zip(["bleu1", "bleu2", "bleu3", "bleu4"], np.round(np.mean(scores, axis=0), 2).tolist()))

def str2bool(v):
    """Parse common string representations into booleans."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def make_TSNE(embeddings, responses, clusters, show_responses=False):
    """Plot a TSNE projection of embeddings with optional labels."""
    tsne = TSNE(n_components=2, perplexity=5, n_iter=3000, random_state=42)
    tsne_embeddings = tsne.fit_transform(embeddings)

    # Visualization
    plt.scatter(
        tsne_embeddings[:, 0],
        tsne_embeddings[:, 1],
        alpha=0.3,
        s=5,
        c=clusters,
        cmap="hsv",
    )
    if show_responses:
        n = 50
        for i, response in enumerate(responses):
            if i % n == 0:
                ax.annotate(
                    response, (tsne_embeddings[i, 0], tsne_embeddings[i, 1])
                )  # plot every nth text on the TSNE
    plt.xlabel("TSNE-1")
    plt.ylabel("TSNE-2")
    plt.grid(False)
    plt.axis("off")
    plt.show()
    # ax.set_xlabel("TSNE-1")
    # ax.set_ylabel("TSNE-2")
    # ax.grid(False)
    # ax.axis("off")

def view_pickle(filepath):
    """Print the contents of a pickled object."""
    with open(filepath, "rb") as f:
        obj = pk.load(f)
    print(obj)
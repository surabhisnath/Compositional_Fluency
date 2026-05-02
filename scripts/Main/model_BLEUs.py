"""Resimulate and plot BLEU scores per model."""
import re
import ast
import matplotlib.pyplot as plt
plt.rcParams.update({
    "axes.facecolor": "white",                      # background stays white
    "axes.edgecolor": "black",                      # keep axis edges
    "patch.facecolor": "lightcoral",
    "text.usetex": False,                           # render text with LaTeX
    "font.family": "sans-serif",                    # use serif fonts
    "axes.spines.top": False,                       # remove top border
    "axes.spines.right": False,                     # remove right border
    "axes.labelsize": 16,                           # bigger axis labels
    "xtick.labelsize": 14,                          # bigger x-tick labels
    "ytick.labelsize": 14,                          # bigger y-tick labels
    "axes.titlesize": 18,                           # bigger title
    "figure.dpi": 100,                              # higher resolution
})
import numpy as np
import json
import os
import pickle as pk
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import *

os.makedirs("../../plots/Figure3/", exist_ok=True)

model_name_to_model_print = json.load(open("../../files/model_name_to_model_print.json", "r"))
model_name_to_color = json.load(open("../../files/model_name_to_color.json", "r"))

numsubsamples = 3
splits = pk.load(open("../../files/splits.pk", "rb"))

sim_bleu_means = {}
references_by_split = [[seq[2:] for seq in test_seqs] for _, test_seqs in splits]
n_test_by_split = [len(test_seqs) for _, test_seqs in splits]

for model_name in model_name_to_model_print.keys():
    sim_path = f"../../simulations/model_simulations/{model_name.lower()}_simulations_gpt41.pk"

    with open(sim_path, "rb") as f:
        simulations = pk.load(f)

    bleu_sums = None
    bleu_count = 0
    sim_ind = 0

    for split_ind, (_, test_seqs) in enumerate(splits):
        n_test = n_test_by_split[split_ind]
        references = references_by_split[split_ind]
        for _ in range(numsubsamples):
            forbleu = simulations[sim_ind : sim_ind + n_test]
            sim_ind += n_test
            candidates = [sim[2:] for sim in forbleu]
            bleu = calculate_bleu(candidates, references)
            if bleu_sums is None:
                bleu_sums = {k: 0.0 for k in bleu}
            for k, v in bleu.items():
                bleu_sums[k] += v
            bleu_count += 1

    sim_bleu_mean = sum(0.25 * (bleu_sums[k] / bleu_count) for k in bleu_sums)
    sim_bleu_means[model_name] = sim_bleu_mean

bleu_values = list(sim_bleu_means.values())
human_bleu = 0.25 * 0.909 + 0.25 * 0.242 + 0.25 * 0.030 + 0.25 * 0.001
modelnames = [model_name_to_model_print[m] for m in sim_bleu_means.keys()]
colors = [model_name_to_color[m] for m in sim_bleu_means.keys()]

plt.figure(figsize=(8, 5))
x = np.arange(len(bleu_values))
plt.bar(x, bleu_values, alpha=0.8, color=colors, edgecolor='black', linewidth=1.2)
plt.xticks(x, modelnames, rotation=90)
plt.ylim(min(bleu_values)-0.01, human_bleu+0.01)
plt.ylabel('Cross-Validated BLEU Score')
plt.axhline(y=human_bleu, color='black', linestyle='--', linewidth=1.2)
plt.text(len(bleu_values) - 0.5, human_bleu + 0.005, f'\nHuman BLEU = {human_bleu:.3f}', color='black', fontsize=10, va='top', ha='right')
plt.tight_layout()
plt.savefig(f"../../plots/Figure3/model_bleus.png", transparent=True, dpi=300)
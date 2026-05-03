"""Plot summed test NLLs for all fitted models."""

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
})
import numpy as np
import json
import pickle as pk

modelNLLs = {
    "FreqWeightedHSActivity": 20922.8115234375,
    "FreqWeightedHSActivity_recoverablefeatures": 20942.60400390625,
    "FreqWeightedHSActivity_recoverableweights": 20993.109130859375,

    "FreqWeightedHSActivity_L110_0": 21006.173828125,
    "FreqWeightedHSActivity_L11_0": 20910.610595703125,
    "FreqWeightedHSActivity_L10_1": 20920.423828125,
    "FreqWeightedHSActivity_L10_01": 20922.6533203125,

    "FreqWeightedHSActivity_L210_0": 20966.877197265625,
    "FreqWeightedHSActivity_L21_0": 20914.54833984375,
    "FreqWeightedHSActivity_L20_1": 20918.958740234375,
    "FreqWeightedHSActivity_L20_01": 20922.555908203125
}

model_name_to_model_print = {
    "FreqWeightedHSActivity": "Conceptome-search",
    "FreqWeightedHSActivity_recoverablefeatures": "Conceptome-search with recoverable features",
    "FreqWeightedHSActivity_recoverableweights": "Conceptome-search with recoverable weights",
   
    "FreqWeightedHSActivity_L110_0": "Conceptome-search with L1 reg. λ = 10.0",
    "FreqWeightedHSActivity_L11_0": "Conceptome-search with L1 reg. λ = 1.0",
    "FreqWeightedHSActivity_L10_1": "Conceptome-search with L1 reg. λ = 0.1",
    "FreqWeightedHSActivity_L10_01": "Conceptome-search with L1 reg. λ = 0.01",

    "FreqWeightedHSActivity_L210_0": "Conceptome-search with L2 reg. λ = 10.0",
    "FreqWeightedHSActivity_L21_0": "Conceptome-search with L2 reg. λ = 1.0",
    "FreqWeightedHSActivity_L20_1": "Conceptome-search with L2 reg. λ = 0.1",
    "FreqWeightedHSActivity_L20_01": "Conceptome-search with L2 reg. λ = 0.01",
}

modelnlls = list(modelNLLs.values())
modelnames = [model_name_to_model_print[m] for m in list(modelNLLs.keys())]
colors = ["#BBB3D8" for m in list(modelNLLs.keys())]

plt.figure(figsize=(9, 8))
x = np.array([0, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13])
plt.bar(x, modelnlls, alpha=0.8, color=colors, edgecolor='black', linewidth=1.2)
plt.xticks(x, modelnames, rotation=45, ha='right', va='top', rotation_mode='anchor')
plt.ylim(min(modelnlls) - 100, max(modelnlls) + 100)
plt.ylabel(f'Cross-validated NLL')
plt.tight_layout()
plt.savefig("../../plots/Supplementary/capacity_constraints.png", dpi=300, transparent=True)
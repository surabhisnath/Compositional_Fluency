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
import os
os.makedirs("../../plots/Figure3/", exist_ok=True)

model_name_to_model_print = json.load(open("../../files/model_name_to_model_print.json", "r"))
model_name_to_color = json.load(open("../../files/model_name_to_color.json", "r"))

modelNLLs = {}

for model_name in model_name_to_model_print.keys():
    path = f"../../fits/model_fits/{model_name.lower()}_fits_gpt41.pk"
    
    results = pk.load(open(path, "rb"))
    modelNLLs[model_name] = sum(results["testNLLs"])

modelnlls = list(modelNLLs.values())
modelnames = [model_name_to_model_print[m] for m in modelNLLs.keys()]
colors = [model_name_to_color[m] for m in modelNLLs.keys()]

plt.figure(figsize=(8, 5))
x = np.arange(len(modelNLLs))
plt.bar(x, modelnlls, alpha=0.8, color=colors, edgecolor='black', linewidth=1.2)
plt.xticks(x, modelnames, rotation=90)
plt.ylim(min(modelnlls) - 100, max(modelnlls) + 100)
plt.ylabel(f'Sum Test NLL (over 5 folds)')
plt.tight_layout()
plt.savefig("../../plots/Figure3/model_nlls.png", transparent=True, dpi=300)

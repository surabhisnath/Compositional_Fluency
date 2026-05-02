"""Parse simulation logs to plot BLEU scores per model."""

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
os.makedirs("../../plots/Figure3/", exist_ok=True)

model_name_to_model_print = json.load(open("../../files/model_name_to_model_print.json", "r"))
model_name_to_color = json.load(open("../../files/model_name_to_color.json", "r"))

labels = ["Random", "Freq", "HS", "Freq_HS", "WeightedHS", "FreqWeightedHS", "Activity", "WeightedHSActivity", "FreqWeightedHSActivity", "OneCueStaticLocal", "CombinedCueStatic", "CombinedCueDynamicCat", "Freq_Sim_Subcategory", "Subcategory", "Freq_Subcategory", "Sim_Subcategory"]
with open("../../models/logfiles/model_fitting/fit_and_simulate_models.log", "r") as f:
    log_data = f.read()
matches = re.findall(r"SIM BLEUS MEAN:\s*(\{.*?\})", log_data)
bleu_dicts = [ast.literal_eval(match) for match in matches]
bleus = [0.25 * d['bleu1'] + 0.25 * d['bleu2'] + 0.25 * d['bleu3'] + 0.25 * d['bleu4'] for d in bleu_dicts]         # Average BLEU-1..4 equally for a single summary score.
human_bleu = 0.25 * 0.909 + 0.25 * 0.242 + 0.25 * 0.030 + 0.25 * 0.001
modelnames = [model_name_to_model_print[m] for m in labels]
colors = [model_name_to_color[m] for m in labels]

plt.figure(figsize=(8, 5))
x = np.arange(len(bleus))
plt.bar(x, bleus, alpha=0.8, color=colors, edgecolor='black', linewidth=1.2)
plt.xticks(x, modelnames, rotation=90)
plt.ylim(min(bleus)-0.01, human_bleu+0.01)
plt.ylabel('Cross-Validated BLEU Score')
plt.axhline(y=human_bleu, color='black', linestyle='--', linewidth=1.2)
plt.text(len(bleus) - 0.5, human_bleu + 0.005, f'\nHuman BLEU = {human_bleu:.3f}', color='black', fontsize=10, va='top', ha='right')
plt.tight_layout()
plt.savefig(f"../../plots/Figure3/model_bleus.png", transparent=True, dpi=300)

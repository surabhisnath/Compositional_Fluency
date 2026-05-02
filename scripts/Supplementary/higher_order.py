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
    "Freq_HS": 23993.01220703125,
    "Freq_HS_Pers": 23991.2451171875,
    "FreqWeightedHS": 22579.26611328125,
    "FreqWeightedHSWeightedPers": 22260.5673828125,
    "FreqWeightedHSActivity": 20922.8115234375,
    "FreqWeightedHSActivityWeightedPers": 20953.670166015625
}

model_name_to_model_print = {
    "Freq_HS": "HS + IF",
    "Freq_HS_Pers": "HS + Pers + IF",
    "FreqWeightedHS": "wHS + IF",
    "FreqWeightedHSWeightedPers": "wHS + wPers + IF",
    "FreqWeightedHSActivity": "wHS + IF + wAct",
    "FreqWeightedHSActivityWeightedPers": "wHS + wPers + IF + wAct"
}

model_name_to_color = {
    "Freq_HS": "#BBB3D8",
    "Freq_HS_Pers": "#BBB3D8",
    "FreqWeightedHS": "#BBB3D8",
    "FreqWeightedHSWeightedPers": "#BBB3D8",
    "FreqWeightedHSActivity": "#BBB3D8",
    "FreqWeightedHSActivityWeightedPers": "#BBB3D8",
}

modelnlls = list(modelNLLs.values())
modelnames = [model_name_to_model_print[m] for m in list(modelNLLs.keys())]
colors = [model_name_to_color[m] for m in list(modelNLLs.keys())]

plt.figure(figsize=(5, 5))
# x = np.arange(len(modelNLLs))
x = np.array([0, 1, 3, 4, 6, 7])
plt.bar(x, modelnlls, alpha=0.8, color=colors, edgecolor='black', linewidth=1.2)
plt.xticks(x, modelnames, rotation=90)
plt.ylim(min(modelnlls) - 100, max(modelnlls) + 100)
plt.ylabel(f'Cross-validated NLL')
# plt.xticks([], [])
# plt.xlabel('')
plt.tight_layout()
plt.savefig("../plots/higher_order_nlls.png", dpi=300, transparent=True)
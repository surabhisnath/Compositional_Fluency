"""Plot test log-likelihood scaling vs. model size."""

from math import comb
import numpy as np
import pandas as pd
import os
import pickle as pk
import re
import matplotlib.pyplot as plt
plt.rcParams.update({
    "axes.facecolor": "white",                      
    "axes.edgecolor": "black",                      
    "patch.facecolor": "lightcoral",
    "text.usetex": False,                           
    "font.family": "sans-serif",                    
    "axes.spines.top": False,                       
    "axes.spines.right": False,                     
    "axes.labelsize": 16,                           
    "xtick.labelsize": 14,                          
    "ytick.labelsize": 14,                          
    "axes.titlesize": 18,                           
    "figure.dpi": 100,                              
})

def extract_test_nll(filepath, keyword):
    """Find the test NLL value located after a matching keyword."""
    pattern = re.compile(rf'\b{re.escape(keyword)}\b')
    with open(filepath, 'r') as file:
        lines = file.readlines()
        
    for i, line in enumerate(lines):
        if pattern.search(line):
            target_index = i + 7
            if target_index < len(lines):
                parts = lines[target_index].strip().split()
                for part in reversed(parts):
                    try:
                        return float(part)
                    except ValueError:
                        continue
    return None


def extract_embedding_dim(filepath, keyword="num embedding dimensions"):
    """Parse embedding dimension from a log file."""
    with open(filepath, 'r') as file:
        for line in file:
            if keyword in line:
                parts = line.strip().split()
                for i, part in enumerate(parts):
                    if part == "dimensions" and i + 1 < len(parts):
                        try:
                            return int(parts[i + 1])
                        except ValueError:
                            pass
    return None

traditional_embedding_models = ["clip", "minilm",
                                "potion256", "potion128", "potion64",
                                "qwen",
                                "bgesmall", "bgebase", "bgelarge",
                                "e5small", "e5base", "e5large",
                                "rubert",
                                "gtelarge", "gtebert",
                                # "infly"
                            ]

# Build x/y pairs for baseline and weighted variants.
traditional_embedding_models_x = []
traditional_embedding_models_y = [] 
for traditional_embedding_model in traditional_embedding_models:
    filepath = f"../models/logfiles/scaling_plot/{traditional_embedding_model}_andweighted.log"
    embedding_dim = extract_embedding_dim(filepath)

    traditional_embedding_models_x.append(2)
    combinedcuestatic = extract_test_nll(filepath, "CombinedCueStatic")
    traditional_embedding_models_y.append(combinedcuestatic)

    traditional_embedding_models_x.append(embedding_dim + 1)
    combinedcuestaticweighted = extract_test_nll(filepath, "CombinedCueStaticWeighted")
    traditional_embedding_models_y.append(combinedcuestaticweighted)

    traditional_embedding_models_x.append(2*embedding_dim + 1)
    combinedcuestaticweightedactivity = extract_test_nll(filepath, "CombinedCueStaticWeightedActivity")
    traditional_embedding_models_y.append(combinedcuestaticweightedactivity)         
    print(traditional_embedding_model, "\t", combinedcuestatic, "\t", combinedcuestaticweighted, "\t", combinedcuestaticweightedactivity)   

our_embedding_models_x = [2, 139, 277]
our_embedding_models_y = [23990, 22509, 20778]

# plt.scatter(traditional_embedding_models_x, traditional_embedding_models_y, label='Traditional', color='blue')
# plt.scatter(our_embedding_models_x, our_embedding_models_y, label='Ours', color='orange')

plt.figure(figsize=(5, 4))
deg = 2

# Traditional fit
coeffs_trad = np.polyfit(traditional_embedding_models_x, traditional_embedding_models_y, deg)
poly_trad = np.poly1d(coeffs_trad)
x_trad_fit = np.linspace(min(traditional_embedding_models_x), max(traditional_embedding_models_x), 100)
y_trad_fit = poly_trad(x_trad_fit)

# Ours fit
coeffs_ours = np.polyfit(our_embedding_models_x, our_embedding_models_y, deg)
poly_ours = np.poly1d(coeffs_ours)
x_ours_fit = np.linspace(min(our_embedding_models_x), max(our_embedding_models_x), 100)
y_ours_fit = poly_ours(x_ours_fit)

# Plot
plt.scatter(traditional_embedding_models_x, traditional_embedding_models_y, label='Pre-trained', color='#FFD470')
plt.plot(x_trad_fit, y_trad_fit, color="#FFD470", linestyle='--')

plt.scatter(our_embedding_models_x, our_embedding_models_y, label='Fauna', color='#8FCF8F')
plt.plot(x_ours_fit, y_ours_fit, color="#8FCF8F", linestyle='--')

plt.xlabel('Number of Weights')
plt.ylabel('Sum test NLL')
plt.title('Scaling of Embedding Models')
plt.legend()
plt.tight_layout()
plt.savefig('../plots/scaling_plot.png', dpi=300)
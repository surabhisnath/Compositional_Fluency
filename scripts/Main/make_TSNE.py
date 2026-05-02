"""Generate TSNE plots for feature embeddings and selected attributes."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pk
from sklearn.manifold import TSNE
import json
import os

def make_TSNE(embeddings, responses, clusters=None, show_responses=False, feature_name=None):
    """Plot and save a TSNE scatter for the provided embeddings."""
    assert len(embeddings) == len(responses), "Embeddings and responses must have the same length"

    tsne = TSNE(n_components=2, perplexity=10, max_iter=3000, random_state=42, metric="hamming")
    tsne_embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(2, 2))
    ax = plt.gca()

    if clusters is not None:
        # Highlight two clusters when binary labels are provided.
        cluster_0_mask = (clusters == 0)
        cluster_1_mask = (clusters == 1)
        ax.scatter(tsne_embeddings[cluster_0_mask, 0], tsne_embeddings[cluster_0_mask, 1], alpha=0.5, s=17, color="#ED7979", edgecolors="#ED3030", linewidths=0.6, label='Cluster 0')
        ax.scatter(tsne_embeddings[cluster_1_mask, 0], tsne_embeddings[cluster_1_mask, 1], alpha=0.5, s=17, color="#75E151", edgecolors="#3C9120", linewidths=0.6, label='Cluster 1')
    else:
        sc = ax.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], alpha=0.7, s=20, c="#C29DC2", edgecolors="#97449C", linewidths=0.5)

    if show_responses:
        n = 5  # Plot every nth label
        for i, response in enumerate(responses):
            if i % n == 0:
                ax.annotate(response, (tsne_embeddings[i, 0], tsne_embeddings[i, 1]), fontsize=4, alpha=0.7)

    ax.set_xlabel("TSNE-1")
    ax.set_ylabel("TSNE-2")
    ax.grid(False)
    ax.axis("off")
    plt.tight_layout()
    os.makedirs("../../plots/Figure1/", exist_ok=True)
    plt.savefig(f"../../plots/Figure1/tsne_plot_{feature_name}.png", dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    featuredict = pk.load(open(f"../../files/features_gpt41.pk", "rb"))
    feature_names = list(next(iter(featuredict.values())).keys())
    with open("../../files/response_corrections.json", 'r') as f:
        corrections = json.load(f)

    features = {corrections.get(k, k): np.array([1 if values.get(f, "").lower()[:4] == "true" else 0 for f in feature_names]) for k, values in featuredict.items()}
    responses = list(features.keys())
    embeddings = np.array(list(features.values()))
    feature_name_to_ind = dict(zip(feature_names, np.arange(len(feature_names))))

    make_TSNE(embeddings, responses, None, False, "all_features")
    make_TSNE(embeddings, responses, embeddings[:,feature_name_to_ind['feature_Is mammal']], False, "Is mammal")
    make_TSNE(embeddings, responses, embeddings[:,feature_name_to_ind['feature_Is domesticated']], False, "Is domesticated")
    make_TSNE(embeddings, responses, embeddings[:,feature_name_to_ind['feature_Is bird']], False, "Is bird")
    make_TSNE(embeddings, responses, embeddings[:,feature_name_to_ind['feature_Is migratory']], False, "Is migratory")
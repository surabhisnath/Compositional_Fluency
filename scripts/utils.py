"""Shared utilities for modeling, plotting, and preprocessing."""

import warnings

import numpy as np

warnings.simplefilter("ignore")
from nltk.translate.bleu_score import sentence_bleu


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
    return dict(
        zip(
            ["bleu1", "bleu2", "bleu3", "bleu4"],
            np.round(np.mean(scores, axis=0), 2).tolist(),
        )
    )

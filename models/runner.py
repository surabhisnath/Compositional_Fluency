"""Main entry point for fitting, simulation, recovery and analyses."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os 
from Model import Model
from Ours import Ours
from Hills import Hills
from Heineman import Heineman
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))
from utils import *
import time
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
import random
random.seed(42)
import pickle as pk
np.random.seed(42)
import os
torch.manual_seed(42)
from tqdm import tqdm
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import ttest_ind
from itertools import combinations
from brokenaxes import brokenaxes
from scipy.stats import spearmanr
from scipy.stats import norm


best_model_class = "ours"
best_model_name = "FreqWeightedHSActivity"

def changeweights(weights, i):
    """Generate perturbed weight vectors for parameter recovery."""
    device = weights.device
    shape = weights.shape
    if i == 0:
        return weights
    if i == 1:
        # Variation 1: Add small Gaussian noise (std=0.1)
        noise = torch.randn(shape, device=device) * 0.1
        fakeweights = weights + noise
    elif i == 2:
        # Variation 2: Add larger Gaussian noise (std=0.3)
        noise = torch.randn(shape, device=device) * 0.3
        fakeweights = weights + noise
    elif i == 3:
        # Variation 3: Scale weights by small multiplicative noise (mean=1.0, std=0.05)
        scale = torch.normal(mean=1.0, std=0.05, size=shape).to(device)
        fakeweights = weights * scale
    elif i == 4:
        # Variation 4: Add constant upward shift (+0.2)
        fakeweights = weights + 0.2
    elif i == 5:
        # Variation 5: Add constant downward shift (−0.2)
        fakeweights = weights - 0.2
    elif i == 6:
        # Variation 6: Resample new weights from original distribution (mean=0.15, std=0.63)
        fakeweights = torch.normal(mean=0.15, std=0.63, size=shape).to(device)
    elif i == 7:
        # Variation 7: Flip the sign of 10% of the weights
        fakeweights = weights.clone()
        idx = torch.randperm(len(weights))[:int(0.1 * len(weights))].to(device)
        fakeweights[idx] *= -1
    elif i == 8:
        # Variation 8: Zero out 15% of the weights to introduce sparsity
        fakeweights = weights.clone()
        idx = torch.randperm(len(weights))[:int(0.15 * len(weights))].to(device)
        fakeweights[idx] = 0
    elif i == 9:    
        # Variation 9: Truncate weights to be within 1 std deviation around the mean
        fakeweights = torch.clamp(weights, 0.15 - 0.63, 0.15 + 0.63)
    elif i == 10:
        # Variation 10: Reverse weights and add small Gaussian noise (std=0.1)
        noise = torch.randn(shape, device=device) * 0.1
        fakeweights = weights.flip(0) + noise

    pk.dump(fakeweights, open(f"../fits/parameter_recovery/fakeweights{i}.pk", "wb"))
    return fakeweights

def run(config):
    models = {}
    
    modelobj = Model(config)                # Initialize shared data and configuration.

    ours = Ours(modelobj)
    ours.create_models()
    models["ours"] = ours
    
    hills = Hills(modelobj)
    hills.create_models()
    models["hills"] = hills
    
    if config["dataset"] == "hills":        # category cue only defined on animals from Hills dataset
        heineman = Heineman(modelobj)
        heineman.create_models()
        models["heineman"] = heineman

    if config["printBLEU"]:
        BLEUs = []
        for (train_sequences, test_sequences) in modelobj.splits:
            for i in range(modelobj.numsubsamples):
                train_sample = random.sample(train_sequences, k=len(test_sequences))
                BLEUs.append(calculate_bleu([trseq[2:] for trseq in train_sample], [teseq[2:] for teseq in test_sequences]))
        human_scores = []
        for d in BLEUs:
            combined = 0.25 * (d["bleu1"] + d["bleu2"] + d["bleu3"] + d["bleu4"])
            human_scores.append(combined)
        human_scores = np.array(human_scores)
        mean_human_bleu = np.mean(human_scores)
        sd_human_bleu = np.std(human_scores, ddof=1)
        se_human_bleu = sd_human_bleu / np.sqrt(len(human_scores))
        print("HUMAN BLEU SCORE =", mean_human_bleu)
        print("HUMAN BLEU SE =", se_human_bleu)
        print("HUMAN BLEU SD =", sd_human_bleu)
 
    if config["fit"]:
        print("--------------------------------FITTING MODELS--------------------------------")
        foldername = "model_fits"
        os.makedirs(f"../fits/{foldername}", exist_ok=True)
        labels = []
        modelnlls = []
        for model_class in models:
            for model_name in models[model_class].models:
                if config["save"]:
                    try:
                        results = pk.load(open(f"../fits/model_fits/{model_name.lower()}_fits_{config["featurestouse"]}.pk", "rb"))
                    except:
                        print(model_class, model_name)
                        start_time = time.time()
                        models[model_class].models[model_name].fit(folderinfits=foldername)
                        results = models[model_class].models[model_name].results
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print(f"{model_name} completed in {elapsed_time:.2f} seconds")
                else:
                    print(model_class, model_name)
                    start_time = time.time()
                    models[model_class].models[model_name].fit(folderinfits=foldername)
                    results = models[model_class].models[model_name].results
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print(f"{model_name} completed in {elapsed_time:.2f} seconds")

                modelnlls.append(sum(results["testNLLs"]))
                labels.append(model_name)
        
        if config["save"]:
            pk.dump(dict(zip(labels, modelnlls)), open("../files/modelNLLs.pk", "wb"))

    if config["simulate"]:
        print("--------------------------------SIMULATING MODELS--------------------------------")
        foldername = "model_simulations"
        os.makedirs(f"../simulations/{foldername}", exist_ok=True)
        labels = []
        modelbleus = []
        for model_class in models:
            for model_name in models[model_class].models:
                if config["save"]:
                    try:
                        simseqs = pk.load(open(f"../simulations/model_simulations/{model_name.lower()}_simulations_{config["featurestouse"]}.pk", "rb"))
                    except:
                        if models[model_class].models[model_name].dynamic:
                            if not models[model_class].models[model_name].dynamic_cat:
                                continue
                        print(model_class, model_name)
                        models[model_class].models[model_name].simulate(folderinsimulations=foldername)
                        simseqs = models[model_class].models[model_name].simulations
                else:
                    if models[model_class].models[model_name].dynamic:
                        if not models[model_class].models[model_name].dynamic_cat:
                            continue
                    print(model_class, model_name)
                    models[model_class].models[model_name].simulate(folderinsimulations=foldername)
                    simseqs = models[model_class].models[model_name].simulations
         
    def get_results():
        suffix = "_fulldata"
        try:
            results = pk.load(open(f"../fits/model_fits/{best_model_name.lower()}_fits_{config["featurestouse"]}{suffix}.pk", "rb"))
        except:
            models[best_model_class].models[best_model_name].suffix = suffix
            models[best_model_class].models[best_model_name].custom_splits = [(models[best_model_class].models[best_model_name].sequences, [])]
            models[best_model_class].models[best_model_name].fit(customsequences=True)
            results = pk.load(open(f"../fits/model_fits/{best_model_name.lower()}_fits_{config["featurestouse"]}{suffix}.pk", "rb"))
        return results

    def get_weights():
        results = get_results()
        learned_weights = results[f"weights_fold1_fulldata"].detach()
        features = models[best_model_class].models[best_model_name].feature_names
        learned_weights_freq = learned_weights[0]
        learned_weights_HS = learned_weights[1:1+len(features)]
        learned_weights_Act = learned_weights[1+len(features):]
        return learned_weights, learned_weights_HS, learned_weights_freq, learned_weights_Act

    if config["printweightsfulldata"]:
        all_learned_weights, learned_weights_HS, learned_weights_freq, learned_weights_Act = get_weights()
        print("learned_weights", all_learned_weights)
        print("learned_weights_HS", learned_weights_HS)
        print("learned_weights_freq", learned_weights_freq)
        print("learned_weights_Act", learned_weights_Act)
    
    if config["ablation"]:
        print("--------------------------------ABLATION STUDY--------------------------------")
        results = get_results()
        original_weights = results[f"weights_fold1_fulldata"].detach()
        best_model_nll = sum(results[f"trainNLLs_fulldata"])
        sequences = models[best_model_class].models[best_model_name].sequences
        num_features = models[best_model_class].models[best_model_name].num_features

        try:
            barplot_HS = pk.load(open(f"../fits/ablations/ablations_HS.pk", "rb"))
            barplot_Activity = pk.load(open(f"../fits/ablations/ablations_Activity.pk", "rb"))
        except:
            totalnlls = []
            for i in tqdm(range(len(original_weights) + 1)):
                weights = original_weights.clone()
                if i != 0:
                    weights[i-1] = 0
                totalnll = sum([models[best_model_class].models[best_model_name].get_nll(seq, weights, True) for seq in sequences])
                totalnlls.append(totalnll)
    
            barplot_HS = [totalnlls[i].detach().cpu().item() for i in np.arange(0, 2 + num_features)]
            barplot_Activity = [totalnlls[i].detach().cpu().item() for i in [0, 1] + list(np.arange(2 + num_features, 2 + 2*num_features))]
            pk.dump(barplot_HS, open(f"../fits/ablations/ablations_HS.pk", "wb"))
            pk.dump(barplot_Activity, open(f"../fits/ablations/ablations_Activity.pk", "wb"))

        plt.figure(figsize=(15, 8))
        labels = ["with all weights", "no freq"] + [f"no HS_{feat}" for feat in models[best_model_class].models[best_model_name].feature_names]
        barplot_HS, labels = zip(*sorted(zip(barplot_HS[2:], labels[2:])))
        barplot_HS = np.array(barplot_HS)
        x = np.arange(len(barplot_HS))
        plt.bar(x, barplot_HS - best_model_nll, alpha=0.8, color='#9370DB')
        plt.xticks(x, labels, rotation=60, fontsize=6, ha='right')
        plt.ylim(min(barplot_HS - best_model_nll), max(barplot_HS - best_model_nll))
        plt.ylabel(f'Increase in NLL')
        plt.title(f'Ablation for HS')
        plt.grid(axis='y', linestyle=':', alpha=0.5)
        plt.tight_layout()
        os.makedirs("../plots/Figure4/", exist_ok=True)
        plt.savefig(f"../plots/Figure4/ablation_HS.png", dpi=300, bbox_inches='tight')
        print(f"Saved ../plots/Figure4/ablation_HS.png")
        print("Top 10 important features for HS:")
        print(labels[-1:-9:-1])

        plt.figure(figsize=(15, 8))
        labels = ["with all weights", "no freq"] + [f"no Activity_{feat}" for feat in models[best_model_class].models[best_model_name].feature_names]
        barplot_Activity, labels = zip(*sorted(zip(barplot_Activity[2:], labels[2:])))
        barplot_Activity = np.array(barplot_Activity)
        x = np.arange(len(barplot_Activity))
        plt.bar(x, barplot_Activity - best_model_nll, alpha=0.8, color='#9370DB')
        plt.xticks(x, labels, rotation=60, fontsize=6, ha='right')
        plt.ylim(min(barplot_Activity - best_model_nll), max(barplot_Activity - best_model_nll))
        plt.ylabel(f'Increase in NLL')
        plt.title(f'Ablation for Activity')
        plt.grid(axis='y', linestyle=':', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"../plots/Figure4/ablation_Activity.png", dpi=300, bbox_inches='tight')
        print(f"Saved ../plots/Figure4/ablation_Activity.png")
        print("Top 10 important features for Activity:")
        print(labels[-1:-9:-1])
    
    if config["visweights"]:
        print("--------------------------------Visualize weights--------------------------------")
        results = get_results()
        weights = results[f"weights_fold1_fulldata"].detach()
        train_nll = sum(results[f"trainNLLs_fulldata"])

        features = models[best_model_class].models[best_model_name].feature_names
        
        weights_HS = weights[1:1+len(features)] 
        weights_Act = weights[1+len(features):]

        delta_ablations_HS = np.array(pk.load(open(f"../fits/ablations/ablations_HS.pk", "rb"))[2:]) - train_nll
        delta_ablations_Act = np.array(pk.load(open(f"../fits/ablations/ablations_Activity.pk", "rb"))[2:]) - train_nll

        top10_HS_idx = np.argsort(delta_ablations_HS)[-10:]
        top10_Act_idx = np.argsort(delta_ablations_Act)[-10:]
        top10_idx = set(top10_HS_idx) | set(top10_Act_idx)

        bax = brokenaxes(xlims=((-2, 105), (max(delta_ablations_HS)-5, max(delta_ablations_HS)+5)), ylims=((-15, 610), (max(delta_ablations_Act)-25, max(delta_ablations_Act)+25)), hspace=.075, wspace=.05)

        for d in bax.diag_handles:
            d.set_visible(False)

        bax.scatter(delta_ablations_HS, delta_ablations_Act,
                    color="slateblue", alpha=0.6, s=50)

        # print(np.mean(np.array(delta_ablations_HS) <= 20618 - train_nll))
        # print(np.mean(np.array(delta_ablations_Act) <= 20700 - train_nll))

        highlight_idx = [i for i in range(len(features)) if (delta_ablations_HS[i] > 20618 - train_nll and delta_ablations_Act[i] > 20700 - train_nll) or (i in top10_idx)]

        colours = []
        for ind in highlight_idx:
            if weights_Act[ind] > 0 and weights_HS[ind] > 0:
                colours.append("crimson")
            if weights_Act[ind] < 0 and weights_HS[ind] < 0:
                colours.append("lawngreen")
            if weights_Act[ind] > 0 and weights_HS[ind] < 0:
                colours.append("gold")
            if weights_Act[ind] < 0 and weights_HS[ind] > 0:
                colours.append("dodgerblue")
        
        bax.scatter(np.array(delta_ablations_HS)[highlight_idx], np.array(delta_ablations_Act)[highlight_idx], color=colours, alpha=0.8, s=50, linewidth=0.4)
        for ax in bax.axs:
            ax.tick_params(axis="both", labelsize=15)

        bax.set_xlabel("HS Ablation Effect", labelpad=35, fontsize=16)
        bax.set_ylabel("Activity Ablation Effect", labelpad=50, fontsize=16)
        
        plt.tight_layout()
        os.makedirs("../plots/Figure4/", exist_ok=True)
        plt.savefig("../plots/Figure4/visweights.png", dpi=600)
    
    if config["RT_analysis"]:
        print("--------------------------------RT ANALYSIS--------------------------------")
        results = get_results()
        weights = results[f"weights_fold1_fulldata"].detach()
        RTs = models[best_model_class].models[best_model_name].RTs
        sequences = models[best_model_class].models[best_model_name].sequences

        data_metrics = models[best_model_class].models[best_model_name].data_metrics
        
        RTs_forreg, logPrej_forreg, freq_forreg, HS_forreg, activity_forreg, pid, responses, trials, cue_transitions_forreg, patchnum2_forreg, numwithinpatch2_forreg  = [], [], [], [], [], [], [], [], [], [], []
        patchnum, numwithinpatch, switchornot = [], [], []
        patchnum2, numwithinpatch2, cue_transitions = pk.load(open("../files/patchnum2.pk", "rb")), pk.load(open("../files/numwithinpatch2.pk", "rb")), pk.load(open("../files/cue_transitions.pk", "rb"))
        
        for sid, seq in enumerate(sequences):
            logprobs_withoutmasking, nll, freq, HS, activity = models[best_model_class].models[best_model_name].get_nll_withoutmasking(seq, weights)
            freq_forreg.extend(freq.cpu().numpy())
            HS_forreg.extend(HS.cpu().numpy())
            activity_forreg.extend(activity.cpu().numpy())
            cue_transitions_forreg.extend(cue_transitions[sid])
            patchnum2_forreg.extend(patchnum2[sid])
            numwithinpatch2_forreg.extend(numwithinpatch2[sid])

            den = torch.logsumexp(logprobs_withoutmasking, dim=1)      # shape len(seq) - 2

            mask = np.ones((len(seq) - 2, len(models[best_model_class].models[best_model_name].unique_responses)))
            for i in range(2, len(seq)):
                visited_responses = np.array([models[best_model_class].models[best_model_name].unique_response_to_index[resp] for resp in seq[:i-1]])
                mask[i - 2, visited_responses] = 0
            mask = torch.tensor(mask, dtype=torch.bool, device=device)
            visited = logprobs_withoutmasking.masked_fill(mask, -np.inf)
            num = torch.logsumexp(visited, dim=1)       # should be shape len(seq) - 2

            logPrej = num
            # logPrej = num - den
            logPrej_forreg.extend(logPrej.cpu().numpy())
            # logPrej_forreg.extend(logPrej.cpu().numpy() - np.log(np.arange(1, 1 + len(seq) - 2)))

            responses.extend(seq[2:])
            RT = np.log(np.array(RTs[sid][2:]) + 0.001)
            RTs_forreg.extend(RT)
            pid.extend([sid] * (len(seq) - 2))
            trials.extend(np.arange(2, 2 + len(seq) - 2))
            patchnum.extend(data_metrics["patchnum"][sid][2:])
            numwithinpatch.extend(data_metrics["numwithinpatch"][sid][2:])
            switchornot.extend(data_metrics["switchornot"][sid][2:])

        df = pd.DataFrame({"response": responses, "IF": freq_forreg, "wHS": HS_forreg, "wFA": activity_forreg, "logRT": RTs_forreg, "logPrej": logPrej_forreg, "pid": pid, "trial": trials, "patchnum": patchnum, "numwithinpatch": numwithinpatch, "switchornot": switchornot, "cue_transitions": cue_transitions_forreg, "patchnum2": patchnum2_forreg, "numwithinpatch2": numwithinpatch2_forreg})
        df = df[df["logRT"] > -1.6]   # removes RT < 200 ms
        df.to_csv("../csvs/RT_analysis.csv", index=False)
        non_continuous_cols = ["response", "cue_transitions", "switchornot"]
        continuous_cols = df.columns.difference(non_continuous_cols)
        df[continuous_cols] = (df[continuous_cols] - df[continuous_cols].mean()) / df[continuous_cols].std(ddof=0)

        df["prev_IF"] = df.groupby("pid")["IF"].shift(1)
        df["prev_wHS"] = df.groupby("pid")["wHS"].shift(1)
        df["prev_wFA"] = df.groupby("pid")["wFA"].shift(1)
        df["prev_prev_IF"] = df.groupby("pid")["prev_IF"].shift(1)
        df["prev_prev_wHS"] = df.groupby("pid")["prev_wHS"].shift(1)
        df["prev_prev_wFA"] = df.groupby("pid")["prev_wFA"].shift(1)
        df = df.dropna()

        # ------------------------ RT Regression Models ---------------------------
        def RT_model(modelname):
            print(modelname + " + 1|pid")
            model = smf.mixedlm(modelname, df, groups=df["pid"]).fit(reml=False)
            print(model.summary())
            var_random = model.cov_re.iloc[0, 0]
            var_resid = model.scale
            X = model.model.exog
            beta = model.fe_params.values
            fitted_fixed = X @ beta
            var_fixed = np.var(fitted_fixed, ddof=1)
            R2_marginal = var_fixed / (var_fixed + var_random + var_resid)
            R2_conditional = (var_fixed + var_random) / (var_fixed + var_random + var_resid)
            print(f"Marginal R^2 (fixed effects): {R2_marginal:.3f}")
            print(f"Conditional R^2 (fixed + random): {R2_conditional:.3f}")
            print(f"Log-Likelihood: {model.llf:.2f}")
            print(f"AIC: {model.aic:.2f}")
            print(f"BIC: {model.bic:.2f}")
            print()

        RT_model("logRT ~ IF + wHS + wFA")
        RT_model("logRT ~ IF + wHS + wFA + logPrej")
        RT_model("logRT ~ IF + wHS + wFA + logPrej + C(cue_transitions)")

        # supplementary
        RT_model("logRT ~ IF + wHS + wFA + logPrej + trial + C(cue_transitions)")
        RT_model("logRT ~ IF + wHS + wFA + logPrej + C(switchornot)")
        RT_model("logRT ~ IF + wHS + wFA + prev_IF + prev_wHS + prev_wFA + prev_prev_IF + prev_prev_wHS + prev_prev_wFA")
    
    if config["ARS"]:
        print("--------------------------------ARS--------------------------------")
        results = get_results()
        weights = results[f"weights_fold1_fulldata"].detach()
        sequences = models[best_model_class].models[best_model_name].sequences
        RTs = models[best_model_class].models[best_model_name].RTs

        def map_type(t):
            """Map 'HS' → 0, 'freq'/'activity'/'global' → 1."""
            return 1 if t == "HS" else 0

        def classify_transition(prev_type, curr_type):
            if (prev_type == "global" or prev_type == "freq" or prev_type == "activity") and (curr_type == "global" or curr_type == "freq" or curr_type == "activity"):
                return 0
            elif (prev_type == "global" or prev_type == "freq" or prev_type == "activity") and curr_type == "HS":
                return 1
            elif prev_type == "HS" and (curr_type == "global" or curr_type == "freq" or curr_type == "activity"):
                return 2
            elif prev_type == "HS" and curr_type == "HS":
                return 3
            return np.nan

        per_seq_logrt = []
        per_seq_probs = []

        cue_transitions = []
        patchnum2 = []
        numwithinpatch2 = []

        ######
        example_seq = ["owl", "crocodile", "dog", "cat", "tiger", "lion", "ant", "beetle", "wasp", "shark", "whale", "dolphin", "duck", "swan", "goose", "frog"]
        (_, log_probs, nll, freqlogit, HSlogit, Actlogit, _, _, _, freqeratiomax, HSeratiomax, activityeratiomax, globaleratiomax, freqeratiosum, HSeratiosum, activityeratiosum, globaleratiosum) = models[best_model_class].models[best_model_name].get_logits_maxlogits(example_seq, weights)
        # temp = list(zip(HSeratiomax, freqeratiomax, activityeratiomax))
        # for a in temp:
        #     print([float(i) for i in a])
        transition_labels = []
        for hs, freq, act in zip(HSeratiomax, freqeratiomax, activityeratiomax):
            values = [float(hs), float(freq), float(act)]
            if np.argmax(values) == 0:
                transition_labels.append("local")
            else:
                transition_labels.append("global")
        output = ""
        for i, label in enumerate(transition_labels):
            output += f" --{label}--> {example_seq[i + 2]}"
        print(output)
        #####

        for sid, seq in enumerate(sequences):
            rt_seq = RTs[sid]
            (_, log_probs, _, _, _, _, _, _, _, freqeratiomax, HSeratiomax, activityeratiomax, globaleratiomax, freqeratiosum, HSeratiosum, activityeratiosum, globaleratiosum) = models[best_model_class].models[best_model_name].get_logits_maxlogits(seq, weights)
            model_probs = np.exp(log_probs.detach().cpu().numpy())
            
            cue_transitions_seq = [np.nan]
            max_list = []
            for i in range(len(seq) - 2):
                if config["ARS_normalisation_type"] == "max":
                    f1, h1, a1, g1 = (freqeratiomax[i].item(), HSeratiomax[i].item(), activityeratiomax[i].item(), globaleratiomax[i].item())
                    if config["ARS_segmentation_type"] == 3:
                        max_type1 = ["HS", "freq", "activity"][torch.tensor([h1, f1, a1]).argmax().item()]
                    elif config["ARS_segmentation_type"] == 2:
                        max_type1 = ["HS", "global"][torch.tensor([h1, g1]).argmax().item()]
                    max_list.append(max_type1)
                    if i > 0:
                        cue_transitions_seq.append(classify_transition(max_list[-2], max_list[-1]))

                if config["ARS_normalisation_type"] == "mean":
                    f2, h2, a2, g2 = (freqeratiosum[i].item(), HSeratiosum[i].item(), activityeratiosum[i].item(), globaleratiosum[i].item())
                    if config["ARS_segmentation_type"] == 3:
                        max_type2 = ["HS", "freq", "activity"][torch.tensor([h2, f2, a2]).argmax().item()]
                    elif config["ARS_segmentation_type"] == 2:
                        max_type2 = ["HS", "global"][torch.tensor([h2, g2]).argmax().item()]
                    max_list.append(max_type2)
                    if i > 0:
                        cue_transitions_seq.append(classify_transition(max_list[-2], max_list[-1]))

            # calculate hills metrics for our segmentations:
            patchnum2_seq = []
            numwithinpatch2_seq = []
            current_patch = 0
            within_patch = 0
            for i, ct in enumerate(cue_transitions_seq):                
                if i == 0 or np.isnan(ct):
                    # First response always starts patch 0
                    current_patch = 0
                    within_patch = 0
                elif ct in (0, 1, 2):
                    # Boundary → new patch
                    current_patch += 1
                    within_patch = 0
                else:  # ct == 3 (HS → HS)
                    within_patch += 1
                patchnum2_seq.append(current_patch)
                numwithinpatch2_seq.append(within_patch)
            cue_transitions.append(cue_transitions_seq)
            patchnum2.append(patchnum2_seq)
            numwithinpatch2.append(numwithinpatch2_seq)

            logrt = [[[] for _ in range(2)] for _ in range(2)]
            probs = [[[] for _ in range(2)] for _ in range(2)]

            for i in range(1, len(max_list)):
                t_from, t_to = map_type(max_list[i-1]), map_type(max_list[i])

                rt_val = np.log(rt_seq[i + 2] + 0.001)
                logrt[t_from][t_to].append(rt_val)
                probs[t_from][t_to].append(model_probs[i])
            per_seq_logrt.append(logrt)
            per_seq_probs.append(probs)

        pk.dump(cue_transitions, open("../files/cue_transitions.pk", "wb"))
        pk.dump(patchnum2, open("../files/patchnum2.pk", "wb"))
        pk.dump(numwithinpatch2, open("../files/numwithinpatch2.pk", "wb"))

        mean_logrt = np.zeros((2, 2))
        se_logrt = np.zeros((2, 2))
        mean_probs = np.zeros((2, 2))
        se_probs = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                # I've checked this logic to be true
                vals = [np.mean(logrt[i][j]) for logrt in per_seq_logrt if logrt[i][j]]       # not everyone may have all 4 types of transitions in their seq therefore if logrt[i][j] ie if it is not empty
                mean_logrt[i, j] = np.mean(vals)
                se_logrt[i, j] = np.std(vals, ddof=1) / np.sqrt(len(vals))

                vals = [np.mean(probs[i][j]) for probs in per_seq_probs if probs[i][j]]       # not everyone may have all 4 types of transitions in their seq therefore if probs[i][j] ie if it is not empty
                mean_probs[i, j] = np.mean(vals)
                se_probs[i, j] = np.std(vals, ddof=1) / np.sqrt(len(vals))

        labels = ["global", "local"]
        print("\n=== Mean ± SEM log(RT) (max_type) ===")
        df_mean = pd.DataFrame(mean_logrt, index=labels, columns=labels)
        df_se   = pd.DataFrame(se_logrt, index=labels, columns=labels)
        print(df_mean.round(3).astype(str) + " ± " + df_se.round(3).astype(str))

        print("\n=== Mean ± SEM probs (max_type) ===")
        df_mean = pd.DataFrame(mean_probs, index=labels, columns=labels)
        df_se   = pd.DataFrame(se_probs, index=labels, columns=labels)
        print(df_mean.round(3).astype(str) + " ± " + df_se.round(3).astype(str))
        
        # ---------------------------------

        # t-tests:
        transition_labels = ["local→local", "local→global", "global→local", "global→global"]
        idx = [(0,0), (0,1), (1,0), (1,1)]
        all_vals = {}
        for name, (r, c) in zip(transition_labels, idx):
            all_vals[name] = [np.mean(lr[r][c]) for lr in per_seq_logrt if lr[r][c]]
        for (name1, vals1), (name2, vals2) in combinations(all_vals.items(), 2):
            if len(vals1) > 1 and len(vals2) > 1:
                t_stat, p_val = ttest_ind(vals1, vals2, equal_var=False)
                print(f"{name1} vs {name2}: t = {t_stat:.3f}, p = {p_val:.5f}")
        
        # ---------------------------------

        def plot_transition_heatmap(data, errors, title, cbar_label, save_path):
            fig, ax = plt.subplots(figsize=(5, 5))
            im = ax.imshow(data, cmap="Reds", alpha=0.5)

            for i in range(2):
                for j in range(2):
                    if np.isnan(data[i, j]):
                        text = "NaN"
                    else:
                        text = f"{data[i, j]:.3f} ±\n{errors[i, j]:.3f}"
                    ax.text(j, i, text, ha='center', va='center', fontsize=13, color='black')

            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(labels, fontsize=15)
            ax.set_yticklabels(labels, fontsize=15)
            ax.set_xlabel("To", fontsize=18, labelpad=10)
            ax.set_ylabel("From", fontsize=18, labelpad=10)
            ax.set_title(title, fontsize=15, pad=12)

            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(cbar_label, fontsize=15)
            cbar.ax.tick_params(labelsize=15)

            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close()
        
        os.makedirs("../plots/Figure6/", exist_ok=True)
        plot_transition_heatmap(mean_logrt, se_logrt, title="Mean log(RT) by transition type", cbar_label="Mean log(RT)", save_path="../plots/Figure6/meanlogRT_transitions.png")
        plot_transition_heatmap(mean_probs, se_probs, title="Mean probability by transition type", cbar_label="Mean probability", save_path="../plots/Figure6/meanprob_transitions.png")
    
    if config["recovery"]:
        print("--------------------------------MODEL RECOVERY--------------------------------")
        foldername = "model_recovery"
        os.makedirs(f"../fits/{foldername}", exist_ok=True)
        # for model_class_sim in reversed(list(models)):
        #     for model_name_sim in reversed(list(models[model_class_sim].models)):
        for model_class_sim in models:
            for model_name_sim in models[model_class_sim].models:
                try:
                    simseqs = models[model_class_sim].models[model_name_sim].simulations
                except:
                    print(f"Loading simulations for {model_name_sim}")
                    simseqs = pk.load(open(f"../simulations/model_simulations/{model_name_sim.lower()}_simulations_{config["featurestouse"]}.pk", "rb"))
                
                for model_class in models:
                    for model_name in models[model_class].models:
                        for ssid, ss in enumerate([simseqs[::3], simseqs[1::3], simseqs[2::3]]):
                            print(model_name_sim, model_name, ssid)
                            suffix = f"_recovery_{model_name_sim.lower()}_{ssid + 1}"
                            if not os.path.exists(f"../fits/{foldername}/{model_name.lower()}_fits_{config["featurestouse"]}{suffix}.pk"):
                                print("Fitting...", model_name_sim, model_name, ssid)
                                models[model_class].models[model_name].suffix = suffix
                                models[model_class].models[model_name].custom_splits = models[model_class].models[model_name].split_sequences(ss)
                                start_time = time.time()
                                models[model_class].models[model_name].fit(customsequences=True, folderinfits=foldername)
                                end_time = time.time()
                                elapsed_time = end_time - start_time
                                print(f"{model_name} completed in {elapsed_time:.2f} seconds")

    if config["parameterrecovery"]:
        print("--------------------------------PARAMETER RECOVERY--------------------------------")
        # fit on full data, get weights, simulate, recover
        # modulate original weights, simulate, recover (repeat 10 times)
        foldername = "parameter_recovery"
        os.makedirs(f"../simulations/{foldername}", exist_ok=True)
        os.makedirs(f"../fits/{foldername}", exist_ok=True)
        
        results = get_results()
        original_weights = results[f"weights_fold1_fulldata"].detach()

        for i in range(11):
            try:
                simseqs = pk.load(open(f"../simulations/{foldername}/{best_model_name.lower()}_simulations_gpt41_fakeweights_{i}.pk", "rb"))
            except:
                print(f"Modifying weights... {i}")
                weights = changeweights(original_weights, i)
                models[best_model_class].models[best_model_name].suffix = f"_fakeweights_{i}"
                models[best_model_class].models[best_model_name].simulateweights(weights)
                simseqs = models[best_model_class].models[best_model_name].simulations

            for ssid, ss in enumerate([simseqs[::3], simseqs[1::3], simseqs[2::3]]):
                suffix2 = f"_paramrecovery_{i}_{ssid + 1}"
                if not os.path.exists(f"../fits/{foldername}/{best_model_name.lower()}_fits_gpt41{suffix2}.pk"):
                    print(best_model_class, best_model_name, i, ssid)
                    models[best_model_class].models[best_model_name].suffix = suffix2
                    models[best_model_class].models[best_model_name].custom_splits = [(ss, [])]
                    start_time = time.time()
                    models[best_model_class].models[best_model_name].fit(customsequences=True, folderinfits=foldername)
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print(f"{best_model_name} completed in {elapsed_time:.2f} seconds")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="process_modelling", description="Implements various models of semantic exploration")

    parser.add_argument("--dataset", type=str, default="hills", help="hills or claire or data_LLMs_VF")
    parser.add_argument("--representation", type=str, default="clip", help="representation to use for embedding responses: clip (768), gtelarge (1024), minilm (348), potion_256 (256), potion_128 (128), potion_64 (64)")
    
    parser.add_argument("--fit", action="store_true", help="fit all models (default: False)")
    parser.add_argument("--simulate", action="store_true", help="simulate all models (default: False)")
    
    parser.add_argument("--save", action="store_true", default=True, help="save pk files (default: True)")
    parser.add_argument("--nosave", action="store_false", dest="save", help="don't save pk files")

    parser.add_argument("--print", action="store_true", default=True, help="print fits (default: True)")
    parser.add_argument("--noprint", action="store_false", dest="print", help="don't print fits")

    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--initval", type=float, default=1.0, help="initial parameter value")
    parser.add_argument("--tol", type=float, default=1e-6, help="gradient and function/param tolerance")
    parser.add_argument("--maxiter", type=int, default=5000, help="maximum number of training iterations")

    parser.add_argument("--fitting", type=str, default="group", help="how to fit betas: individual or group")
    parser.add_argument("--cv", type=int, default=5, help="cross-validation folds for group fitting. 1 = train-test:80-20. >1 = cv folds")

    parser.add_argument("--featurestouse", type=str, default="gpt41", help="features to use: gpt41, llama or random")
    parser.add_argument("--mask", action="store_true", default=True, help="use mask over previous responses (default: True)")
    parser.add_argument("--nomask", action="store_false", dest="mask", help="don't use mask")

    parser.add_argument("--usehillsresp", action="store_true", default=True, help="use hills responses (default: True)")
    parser.add_argument("--useallresp", action="store_false", dest="usehillsresp", help="use all responses, across datasets")

    parser.add_argument("--useapifreq", action="store_true", default=True, help="use API frequency (default: True)")
    parser.add_argument("--usehillsfreq", action="store_false", dest="useapifreq", help="use hills frequency")

    parser.add_argument("--reglambda", type=float, default=0, help="regularisation lambda")
    parser.add_argument("--regtype", type=str, default="none", help="regularisation type - l1 or l2 (default: none)")

    parser.add_argument("--printweightsfulldata", action="store_true", help="print weights on fulldata (default: False)")
    parser.add_argument("--printBLEU", action="store_true", help="print human-human BLEU (default: False)")
    
    parser.add_argument("--recovery", action="store_true", help="recover all models (default: False)")
    parser.add_argument("--parameterrecovery", action="store_true", help="simulate fake weights (default: False)")
    
    parser.add_argument("--ablation", action="store_true", help="ablate weights (default: False)")
    parser.add_argument("--visweights", action="store_true", help="visualise weights (default: False)")

    parser.add_argument("--RT_analysis", action="store_true", help="analyse RTs (default: False)")
    
    parser.add_argument("--ARS", action="store_true", help="analyse RTs (default: False)")
    parser.add_argument("--ARS_normalisation_type", type=str, default="max", help="normalisation type for ARS: max or mean")
    parser.add_argument("--ARS_segmentation_type", type=int, default=3, help="segmentation type for ARS: 3 (wHS, IF, wFA), or 2 (local-global)")
    
    parser.add_argument("--remove_features_that_donot_recover", action="store_true", help="remove features that do not recover (default: False)")
    parser.add_argument("--remove_weights_that_donot_recover", action="store_true", help="remove features that do not recover (default: False)")
    

    args = parser.parse_args()
    config = vars(args)
    
    print("CONFIG:", config)
    
    run(config)
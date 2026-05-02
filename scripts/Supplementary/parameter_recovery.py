"""Generate parameter recovery plots and summary statistics."""

import pickle as pk
import random
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import os

def identity_r2(true, recovered):
    """R^2 against the identity line (y=x)."""
    ss_res = np.sum((recovered - true)**2)
    ss_tot = np.sum((recovered - np.mean(recovered))**2)
    return 1 - ss_res / ss_tot

def identity_rmsd(true, recovered):
    """Root-mean-square deviation between true and recovered values."""
    return np.sqrt(np.mean((recovered - true)**2))

weight_names = ['freq', 'Is mammal', 'Is bird', 'Is insect', 'Is reptile', 'Is amphibian', 'Is fish', 'Is rodent', 'Is primate', 'Is jungle animal', 'Is non-jungle animal', 'Is feline', 'Is canine', 'Is subspecies of an animal', 'Is carnivore', 'Is herbivore', 'Is omnivore', 'Is larger in size compared to other animals', 'Is smaller in size compared to other animals', 'Is average size compared to other animals', 'Is warm-blooded', 'Is cold-blooded', 'Is a predator', 'Is prey for larger animals', 'Is a parasite', 'Is a host for parasites', 'Is nocturnal', 'Is diurnal', 'Has fur', 'Has feathers', 'Has scales', 'Has exoskeleton', 'Has beak', 'Has claws', 'Has whiskers', 'Has horns', 'Has antlers', 'Has tusks', 'Has wings', 'Has tail', 'Has less than four limbs', 'Has exactly four limbs', 'Has more than four limbs', 'Has stripes', 'Has spots', 'Has mane', 'Has crest', 'Has gills', 'Has flippers', 'Has compound eyes', 'Has segmented body', 'Has a long neck', 'Can fly', 'Can swim', 'Can climb', 'Can dig', 'Can jump', 'Can camouflage', 'Can hibernate', 'Can be trained or tamed by humans', 'Is found in zoos', 'Lives in water', 'Lives in trees', 'Lives underground', 'Lives on land', 'Is native to Africa', 'Is native to Asia', 'Is native to North America', 'Is native to South America', 'Is native to Australia', 'Is native to Europe', 'Lives in Arctic/far North', 'Is found in deserts', 'Is found in forests', 'Is found in oceans', 'Is found in grasslands', 'Is found in mountains', 'Lives in burrows', 'Lays eggs', 'Gives birth', 'Is venomous', 'Is domesticated', 'Lives in groups', 'Is solitary', 'Builds nests', 'Is migratory', 'Has social hierarchy', 'Uses tools', 'Shows intelligence', 'Communicates vocally', 'Can change color', 'Is capable of mimicry', 'Has echolocation', 'Is known for speed', 'Is known for strength', 'Is kept as a pet', 'Is used in farming', 'Is hunted by humans', 'Is used for food by humans', 'Is used for transportation', 'Is used in scientific research', 'Has a long lifespan', 'Has regenerative ability', 'Is vertebrate', 'Is invertebrate', 'Is marsupial', 'Is placental', 'Is monotreme', 'Is flightless', 'Has webbed feet', 'Is known for intelligence', 'Is a scavenger', 'Is territorial', 'Is endangered', 'Is bioluminescent', 'Is capable of parental care', 'Is a pollinator', 'Can tolerate extreme temperatures', 'Exhibits seasonal color changes', 'Is active during dawn or dusk (crepuscular)', 'Produces pheromones for communication', 'Lives symbiotically with other species', 'Is bi-parental (both parents care for offspring)', 'Displays mating rituals', 'Has specialized courtship behavior', 'Exhibits territorial marking', 'Is associated with mythology or folklore', 'Exhibits altruistic behavior', 'Is a keystone species in its ecosystem', 'Can regenerate body parts', 'Is raised in captivity or farms', 'Has unique reproductive strategies (e.g., asexual reproduction)', 'Hibernates during winter', 'Has a role in biological pest control', 'Has distinct seasonal breeding cycles', 'Forms a symbiotic relationship with plants (e.g., pollination)', 'Uses specific vocalizations to communicate', 'Is a flagship species (conservation symbol)', 'Displays warning coloration']

true_weights = []

# Load ground-truth weights (full data + synthetic perturbations).
results = pk.load(open(f"../fits/model_fits/freqweightedhsactivity_fits_gpt41_fulldata.pk", "rb"))
original_weights = results[f"weights_fold1_fulldata"]
true_weights.append(original_weights.cpu().numpy())

for i in range(1, 11):
    weights = pk.load(open(f"../fits/parameter_recovery/fakeweights{i}.pk", "rb")).cpu().numpy()
    true_weights.append(weights)

true_weights = np.array(true_weights)

recovered_weights_means = []
recovered_weights_se = []
for i in range(0,11):
    # if i == 0:      # not sure about this (unchanged weights recovery)
    #     print(pk.load(open(f"../fits/parameter_recovery/freqweightedhsactivity_fits_gpt41_paramrecovery_{i}_1.pk", "rb")).keys())
    #     weights1 = pk.load(open(f"../fits/parameter_recovery/freqweightedhsactivity_fits_gpt41_paramrecovery_{i}_1.pk", "rb"))[f"weights_fold1_paramrecovery_1_{i+1}"].cpu().numpy()
    #     weights2 = pk.load(open(f"../fits/parameter_recovery/freqweightedhsactivity_fits_gpt41_paramrecovery_{i}_2.pk", "rb"))[f"weights_fold1_paramrecovery_2_{i+1}"].cpu().numpy()
    #     weights3 = pk.load(open(f"../fits/parameter_recovery/freqweightedhsactivity_fits_gpt41_paramrecovery_{i}_3.pk", "rb"))[f"weights_fold1_paramrecovery_3_{i+1}"].cpu().numpy()
    # else:
    weights1 = pk.load(open(f"../fits/parameter_recovery/freqweightedhsactivity_fits_gpt41_paramrecovery_{i}_1.pk", "rb"))[f"weights_fold1_paramrecovery_{i}_1"].cpu().numpy()
    weights2 = pk.load(open(f"../fits/parameter_recovery/freqweightedhsactivity_fits_gpt41_paramrecovery_{i}_2.pk", "rb"))[f"weights_fold1_paramrecovery_{i}_2"].cpu().numpy()
    weights3 = pk.load(open(f"../fits/parameter_recovery/freqweightedhsactivity_fits_gpt41_paramrecovery_{i}_3.pk", "rb"))[f"weights_fold1_paramrecovery_{i}_3"].cpu().numpy()

    recovered_weights_means.append((weights1 + weights2 + weights3) / 3)
    recovered_weights_se.append(np.std([weights1, weights2, weights3], axis=0) / np.sqrt(3))

recovered_weights_means = np.array(recovered_weights_means)
recovered_weights_se = np.array(recovered_weights_se)

results = []

os.makedirs("../plots/parameter_recovery/", exist_ok=True)
cmap = plt.get_cmap('tab20b')
plt.figure(figsize=(4, 4))
r2_freq = identity_r2(true_weights[:,0], recovered_weights_means[:,0])
rmsd_freq = identity_rmsd(true_weights[:,0], recovered_weights_means[:,0])
plt.scatter(true_weights[0, 0], recovered_weights_means[0, 0], color='red', label='True', edgecolor='k', s=60, alpha=0.9)
for i in range(1, 11):
    color = cmap(i / 10)
    plt.scatter(true_weights[i, 0], recovered_weights_means[i, 0],
                color=color, label=f'Var {i}', edgecolor='k', s=60, alpha=0.9)
all_vals = list(true_weights[:, 0]) + list(recovered_weights_means[:, 0])
min_val, max_val = min(all_vals) - 0.1, max(all_vals) + 0.1
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='gray')
plt.gca().set_aspect('equal', adjustable='box')
plt.title(f"{weight_names[0]}\n$R^2$={r2_freq:.2f}, RMSD={rmsd_freq:.2f}")
plt.xlabel("True Weights")
plt.ylabel("Recovered Weights")
plt.legend(title="Points", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig(f"../plots/parameter_recovery/0_{weight_names[0].replace('/', ' or ')}_recovery.png", bbox_inches='tight', dpi=300)
plt.close()

results.append({'Feature': weight_names[0],
        'R2_freq': r2_freq,
        'RMSD_freq': rmsd_freq})

for w in np.arange(1, 139):
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    # --- HS plot ---
    r2_hs = identity_r2(true_weights[:, w], recovered_weights_means[:, w])
    rmsd_hs = identity_rmsd(true_weights[:, w], recovered_weights_means[:, w])
    ax[0].scatter(true_weights[0, w], recovered_weights_means[0, w], color='red', label='True', edgecolor='k', s=60, alpha=0.9)
    for i in range(1, 11):
        color = cmap(i / 10)
        ax[0].scatter(true_weights[i, w], recovered_weights_means[i, w],
                      color=color, label=f'Var {i}', edgecolor='k', s=60, alpha=0.9)
    ax[0].set_xlim(min(true_weights[:, w].min(), recovered_weights_means[:, w].min()) - 0.1,
                   max(true_weights[:, w].max(), recovered_weights_means[:, w].max()) + 0.1)
    ax[0].set_ylim(ax[0].get_xlim())  # Make aspect square-like
    ax[0].plot(ax[0].get_xlim(), ax[0].get_xlim(), linestyle='--', color='gray')
    ax[0].set_aspect('equal', adjustable='box')
    ax[0].set_title(f"HS\n$R^2$={r2_hs:.2f}, RMSD={rmsd_hs:.2f}")
    ax[0].set_xlabel("True Weights")
    ax[0].set_ylabel("Recovered Weights")
    ax[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    # --- Activity plot ---
    offset = 138
    r2_act = identity_r2(true_weights[:, w + offset], recovered_weights_means[:, w + offset])
    rmsd_act = identity_rmsd(true_weights[:, w + offset], recovered_weights_means[:, w + offset])
    ax[1].scatter(true_weights[0, w + offset], recovered_weights_means[0, w + offset], color='red', label='True', edgecolor='k', s=60, alpha=0.9)
    for i in range(1, 11):
        color = cmap(i / 10)
        ax[1].scatter(true_weights[i, w + offset], recovered_weights_means[i, w + offset],
                      color=color, label=f'Var {i}', edgecolor='k', s=60, alpha=0.9)
    ax[1].set_xlim(min(true_weights[:, w + offset].min(), recovered_weights_means[:, w + offset].min()) - 0.1,
                   max(true_weights[:, w + offset].max(), recovered_weights_means[:, w + offset].max()) + 0.1)
    ax[1].set_ylim(ax[1].get_xlim())
    ax[1].plot(ax[1].get_xlim(), ax[1].get_xlim(), linestyle='--', color='gray')
    ax[1].set_aspect('equal', adjustable='box')
    ax[1].set_title(f"Activity\n$R^2$={r2_act:.2f}, RMSD={rmsd_act:.2f}")
    ax[1].set_xlabel("True Weights")
    ax[1].set_ylabel("Recovered Weights")
    ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    fig.suptitle(weight_names[w])
    plt.tight_layout()
    plt.savefig(f"../plots/{w}_{weight_names[w].replace('/', ' or ')}_recovery.png", bbox_inches='tight', dpi=300)
    plt.close()

    results.append({
        'Feature': weight_names[w],
        'R2_HS': r2_hs,
        'RMSD_HS': rmsd_hs,
        'R2_Activity': r2_act,
        'RMSD_Activity': rmsd_act
    })

df_results = pd.DataFrame(results)
df_results.to_csv('../csvs/parameter_recovery.csv', index=False)

plt.figure(figsize=(8, 4))
plt.hist(df_results["R2_HS"])
plt.savefig('../plots/r2_hs_histogram.png', bbox_inches='tight', dpi=300)
plt.figure(figsize=(8, 4))
plt.hist(df_results["R2_Activity"])
plt.savefig('../plots/r2_activity_histogram.png', bbox_inches='tight', dpi=300)
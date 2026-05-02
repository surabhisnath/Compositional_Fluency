"""Plot counts of features per semantic category."""

import matplotlib.pyplot as plt
import numpy as np

# Category	Count
# 1. "Taxonomy / Species Type	13
# 2. Ecological Role	12
# 3. Physiological and Behavioral Traits	19
# 4. Body Characteristics	25
# 5. Abilities / Behavior	18
# 6. Social Behavior	11
# 7. Habitat / Distribution	18
# 8. Reproduction / Lifecycle	11
# 9. Human Interaction & Cultural Use	11

# Mapping of category name to included feature keys.
feature_categories = {
    "Taxonomy / Species Type": [
        "feature_Is mammal", "feature_Is bird", "feature_Is insect", "feature_Is reptile",
        "feature_Is amphibian", "feature_Is fish", "feature_Is rodent", "feature_Is primate",
        "feature_Is jungle animal", "feature_Is non-jungle animal", "feature_Is feline",
        "feature_Is canine", "feature_Is subspecies of an animal", "feature_Is vertebrate",
        "feature_Is invertebrate"
    ],

    "Ecological Role": [
        "feature_Is a predator", "feature_Is prey for larger animals", "feature_Is a parasite",
        "feature_Is a host for parasites", "feature_Is a scavenger", "feature_Is a keystone species in its ecosystem",
        "feature_Is a pollinator", "feature_Has a role in biological pest control",
        "feature_Is a flagship species (conservation symbol)", "feature_Is carnivore",
        "feature_Is herbivore", "feature_Is omnivore"
    ],

    "Physiological Traits": [
        "feature_Is larger in size compared to other animals", "feature_Is smaller in size compared to other animals",
        "feature_Is average size compared to other animals", "feature_Is warm-blooded", "feature_Is cold-blooded",
        "feature_Is nocturnal", "feature_Is diurnal", "feature_Is active during dawn or dusk (crepuscular)",
        "feature_Can hibernate", "feature_Hibernates during winter", "feature_Is venomous",
        "feature_Is flightless", "feature_Has webbed feet", "feature_Is bioluminescent",
        "feature_Has regenerative ability", "feature_Has a long lifespan", "feature_Is endangered",
        "feature_Can tolerate extreme temperatures", "feature_Exhibits seasonal color changes"
    ],

    "Body Characteristics": [
        "feature_Has fur", "feature_Has feathers", "feature_Has scales", "feature_Has exoskeleton",
        "feature_Has beak", "feature_Has claws", "feature_Has whiskers", "feature_Has horns",
        "feature_Has antlers", "feature_Has tusks", "feature_Has wings", "feature_Has tail",
        "feature_Has less than four limbs", "feature_Has exactly four limbs", "feature_Has more than four limbs",
        "feature_Has stripes", "feature_Has spots", "feature_Has mane", "feature_Has crest",
        "feature_Has gills", "feature_Has flippers", "feature_Has compound eyes", "feature_Has segmented body",
        "feature_Has a long neck", "feature_Displays warning coloration"
    ],

    "Abilities": [
        "feature_Can fly", "feature_Can swim", "feature_Can climb", "feature_Can dig", "feature_Can jump",
        "feature_Can camouflage", "feature_Can be trained or tamed by humans", "feature_Is migratory",
        "feature_Uses tools", "feature_Shows intelligence", "feature_Can change color",
        "feature_Is capable of mimicry", "feature_Has echolocation", "feature_Is known for speed",
        "feature_Is known for strength", "feature_Can regenerate body parts", "feature_Uses specific vocalizations to communicate",
        "feature_Communicates vocally"
    ],

    "Social Behaviour": [
        "feature_Lives in groups", "feature_Is solitary", "feature_Has social hierarchy",
        "feature_Produces pheromones for communication", "feature_Lives symbiotically with other species",
        "feature_Is bi-parental (both parents care for offspring)", "feature_Exhibits altruistic behavior",
        "feature_Exhibits territorial marking", "feature_Displays mating rituals",
        "feature_Has specialized courtship behavior"
    ],

    "Habitat": [
        "feature_Lives in water", "feature_Lives in trees", "feature_Lives underground", "feature_Lives on land",
        "feature_Lives in burrows", "feature_Is native to Africa", "feature_Is native to Asia",
        "feature_Is native to North America", "feature_Is native to South America", "feature_Is native to Australia",
        "feature_Is native to Europe", "feature_Lives in Arctic/far North", "feature_Is found in deserts",
        "feature_Is found in forests", "feature_Is found in oceans", "feature_Is found in grasslands",
        "feature_Is found in mountains", "feature_Forms a symbiotic relationship with plants (e.g., pollination)"
    ],

    "Reproduction / Lifecycle": [
        "feature_Lays eggs", "feature_Gives birth", "feature_Is marsupial", "feature_Is placental",
        "feature_Is monotreme", "feature_Builds nests", "feature_Is capable of parental care",
        "feature_Has distinct seasonal breeding cycles", "feature_Has unique reproductive strategies (e.g., asexual reproduction)",
        "feature_Displays mating rituals", "feature_Has specialized courtship behavior"
    ],

    "Human Interaction": [
        "feature_Is domesticated", "feature_Is kept as a pet", "feature_Is used in farming",
        "feature_Is hunted by humans", "feature_Is used for food by humans", "feature_Is used for transportation",
        "feature_Is used in scientific research", "feature_Is raised in captivity or farms",
        "feature_Is associated with mythology or folklore", "feature_Is a flagship species (conservation symbol)"
    ]
}

plt.figure(figsize=(4, 5))

# Step 1: Sort by length of feature list
sorted_items = sorted(feature_categories.items(), key=lambda x: len(x[1]))
categories = [item[0] for item in sorted_items]
counts = [len(item[1]) for item in sorted_items]

# Step 2: Generate tighter-spaced x positions
x = np.arange(len(categories)) * 0.12   # reduce spacing factor < 1.0 to make bars closer

# Step 3: Plot
bars = plt.bar(x, counts, width=0.1, color="#D8BFD8")  # keep width fixed

# Step 4: Add count labels above bars
for xi, count in zip(x, counts):
    plt.text(xi, count + 0.1, str(count), ha='center', va='bottom', fontsize=8)

# Step 5: Adjust x-ticks to match shifted bars
plt.xticks(x, categories, rotation=90)
plt.ylabel("Count")
plt.title("Feature Categories & Counts")

for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.gca().axes.yaxis.set_visible(False)

plt.tight_layout()
plt.savefig("../../plots/Figure1/feature_categories.png", dpi=300, bbox_inches="tight")
plt.close()

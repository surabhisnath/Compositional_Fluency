# A Compositional Model of Semantic Fluency

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-enabled-ee4c2c?logo=pytorch&logoColor=white)

This repository contains the data and scripts for the paper: A Compositional Model of Semantic Fluency.

Authors: [Surabhi S Nath](https://surabhisnath.github.io), [Alireza Modirshanechi](https://sites.google.com/view/modirsha), [Peter Dayan](https://www.mpg.de/12309370/biological-cybernetics-dayan)

## Abstract

The ability to recall semantically connected concepts---be it animals, summer fruits, or cities in Italy---is a remarkable capacity of the human mind. Such semantic fluency is thought to rely on traversing a mental space in which concepts are represented in terms of their meanings. However, the structure, properties, and navigability of this representational space remain enigmatic and highly debated. Existing approaches rely either on complex, uninterpretable distributional word-embeddings or on rigid, hand-crafted category norms. Here, we exploit the strengths of both, introducing Conceptome: a version of a compositional, interpretable, feature-based representation of semantic concepts, constructed by leveraging large language models. We use Conceptome to develop Conceptome-search, an auto-regressive model of how humans explore semantic spaces. We validate Conceptome and Conceptome-search using an animal fluency task, showing that they outperform state-of-the-art models in predicting human choices and capture key behavioral patterns such as interference. Our work, hence, offers new insights into the mechanisms underlying semantic fluency and memory retrieval. More broadly, our approach provides a general framework for constructing high-quality representations, with potential applications across cognition, including exploration, navigation, and creative thinking.

## Repository Description

1. `csvs` contain the csv data files
2. `figures` contain the final figures used in the paper
3. `files` contain all config and auxillary files used in the code 
4. `fits` contain the fit pickle files
5. `models` contain the model classes scripts and the main runner script
6. `plots` contain all plots for each figure, which were used to make the figures
7. `scripts` contain analysis scripts
8. `simulations` contain the simulation pickle files

## Setup

We recommend setting up a python virtual environment and installing all the requirements. Please follow these steps:

```bash
git clone https://github.com/surabhisnath/process_modelling.git
cd process_modelling

python3 -m venv .env

# On macOS/Linux
source .env/bin/activate
# On Windows
.env\Scripts\activate

pip install -r requirements.txt
```

## Running the code

### NOTE: GPU required

`models/runner.py` is the main runner script for the following analyses. All settings can be set using arguments of `runner.py`.
Before you being, ensure all models you wish to run are set to 1 in `files/modeltorun.json`.

1. To replicate Figure 1C: run `scripts/Main/make_TSNE.py`. To plot the histogram of feature categories in Figure 1B: run `scripts/main/plot_feature_categories.py`. All plots will be saved in `plots/Figure1/`

2. To analyse features and replicate Figure 2: run `scripts/Main/Model-free_Analysis.ipynb`. All plots will be saved in `plots/Figure2/`

3. To replicate Figure 3: run `scripts/Main/model_NLLs.py` to plot Figure 3B, and run `python scripts/Main/model_BLEUs.py` to plot Figure 3C. `model_nlls.png` and `model_bleus.png` will be saved in `plots/Figure3/`

    To re-run model fitting and simulation: run `python models/runner.py --fit --simulate` (run overnight)

4. To replicate Figure 4: 
First perform feature ablation by running: `python runner.py --ablation`. Ablations saved as `fits/ablations/ablations_Activity.pk` and `fits/ablations/ablations_HS.pk`
Then plot the figure: `python runner.py --visweights`. Plot saved as `plots/Figure4/visweights.png`

5.  To replicate RT modelling in Figure 5 and Figure 6:
Run: `python runner.py --RT_analysis`.

6. To replicate Figure 6 plots:
Run: `python runner.py --ARS`. Plots saved as `plots/Figure6/meanlogRT_transitions.png` and `plots/Figure6/meanprob_transitions.png`

## Citation

If you found this work useful, please consider citing us:

```
@article{nathcompositional,
  title={A Compositional Model of Semantic Fluency},
  author={Nath, Surabhi S and Modirshanechi, Alireza and Dayan, Peter},
  year={2026},
  publisher={OSF},
  url={https://osf.io/preprints/psyarxiv/adwzp_v1}, 
}
```
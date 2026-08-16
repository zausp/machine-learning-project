# Predicting Social Tenancy Misuse Risk in Subsidised Housing

Supplementary code and materials for the research paper *Predicting Unusual Household and Housing Profiles Related to Social Tenancy
Risk Using Machine Learning* (2026).

This repository contains the full analysis pipeline behind the paper: a machine-learning approach for flagging households in subsidised social housing whose profiles are unusual in ways associated with tenancy misuse (illegal subletting and eligibility fraud). Because no ground-truth fraud labels exist in public data, the project constructs **proxy labels** via weak supervision and trains a leakage-controlled classifier to produce a graded risk score for human review — not a definitive fraud determination.

## Repository contents

| File | Description |
|------|-------------|
| `group_project.ipynb` | The complete analysis notebook: data cleaning, proxy-label construction, model training, and evaluation. Runs top to bottom. |
| `group_project.pdf` | A rendered PDF of the executed notebook, with all outputs and figures, for reading without a Jupyter environment. |
| `ahs_variable_reference.[md/csv/pdf]` | Reference table decoding the AHS variables used, their meanings, and their role in the pipeline (feature vs. label component). Linked from the paper's Data section. |

## Dataset

The analysis uses the **American Housing Survey (AHS) 2023 National Public Use File**, published by the U.S. Census Bureau. The data is **not redistributed** in this repository; it is publicly available and must be obtained directly:

- Source: <https://www.census.gov/programs-surveys/ahs>
- File used: `household.csv` from the 2023 National PUF (one row per housing unit)
- Codebook: the AHS Codebook Interactive Tool and the 2023 mini-codebook decode all variable values

Place `household.csv` in the repository root (or update the path in the notebook's load cell) before running.

## Running the notebook

**Requirements:** Python 3.x with the following packages:

```
pandas
scikit-learn
imbalanced-learn
matplotlib
```

Install with:

```bash
pip install pandas scikit-learn imbalanced-learn matplotlib
```

**Steps:**

1. Obtain `household.csv` from the AHS source above and place it in the repository root.
2. Open `group_project.ipynb` in Jupyter.
3. Run all cells top to bottom (Kernel → Restart & Run All).

A fixed random seed (42) is applied to all stochastic operations, so results are deterministic across runs.

## Approach in brief

- **Scope.** Filtered to renter-occupied units, then to subsidised units (`HUDSUB` in {1, 2}) — the only population that can be subject to social tenancy misuse.
- **Proxy labels (weak supervision).** With no ground-truth labels available, a transparent weighted-additive ruleset scores each household across five documented misuse indicators (eligibility anomaly, unrelated adults, secondary family, overcrowding, compounding recent move) and assigns a risk tier.
- **Circularity control.** The variables used to build the labels are **excluded** from the model's feature set, so the model cannot re-learn the labelling rule. It is trained only on independent cost, structural, and demographic features. A contrast experiment quantifies the circularity avoided.
- **Imbalance handling.** SMOTE is applied *inside* the cross-validation pipeline (training folds only), preventing synthetic samples from leaking into evaluation.
- **Models.** Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and SVM are compared.
- **Evaluation.** Macro-F1 (primary), weighted-F1 (secondary), and quadratic weighted kappa (for the ordinal tiers), with confusion matrices, permutation importance, and worked example cases.

## Key limitation

All results depend on a proxy label; the model learns the *definition* of risk encoded in the ruleset, not confirmed fraud. The AHS public file also cannot observe non-occupation (a tenant living elsewhere), the strongest real subletting signal. Findings transfer as **methodology**, not as calibrated local rates. See the paper's Limitations section for full discussion.

## Citation

If referencing this work, please cite the accompanying paper:

> Ahmed Zaus Zahid, Afnan Ali, Aminath Iuzaaz Ismail (2026) *Predicting Unusual Household and Housing Profiles Related to Social Tenancy
Risk Using Machine Learning*. UWE Bristol.

## Authors

Ahmed Zaus Zahid, Afnan Ali, Aminath Iuzaaz Ismail

UWE Bristol — Machine Learning (2026)

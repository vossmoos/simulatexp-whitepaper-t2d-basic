# Metabolomic Digital Twin Demo for T2D

This repository is a compact demonstration of how virtual metabolomic digital twin data can be validated against an external real-world Type 2 Diabetes dataset.

The focus of this project is not to present a full medical product, but to show a practical validation workflow:

- take a model already built from digital twin data
- align overlapping biomarkers with an external dataset
- test whether the learned signal transfers to real samples the model has never seen

In this repo, the digital twin idea means that each synthetic row represents a plausible virtual patient built by scaled AI according to multiple domain-informed metabolomic patterns rather than arbitrary random data.

## Project Goal

The code demonstrates that focused medical digital twins can produce signals that remain biologically meaningful when checked against a real external validation dataset.

This example uses a 12-metabolite panel relevant to T2D and validates a Random Forest, trained only on synthetic twins, on an external dataset with matched overlapping markers.

## Main Artifacts

- `03_external_validate_random_forest.ipynb` for external validation and result inspection
- `utils/plasma_training_common.py` for shared loading, preprocessing, matching, and validation helpers
- `models/rf_plasma_bundle.joblib` for the trained Random Forest bundle
- `models/rf_training_overlap.json` for the synthetic-to-external metabolite overlap map
- `models/external_validation_rf_showfile.json` for saved validation metrics
- `real/showfile_t.txt` for the external source file used in validation alignment

## Biomarker Panel

The model uses 12 overlapping metabolites:

- `2-Hydroxybutyric acid`
- `3-Hydroxybutyric acid`
- `Alanine`
- `Glucose`
- `Glutamine`
- `Glycine`
- `Isoleucine`
- `Leucine`
- `Palmitic acid`
- `Phenylalanine`
- `Tyrosine`
- `Valine`

The overlap file shows 12 matched metabolites out of 106 detected entries in the validation source.

## Validation Summary

The included validation artifact reports:

- model trained on synthetic twins only
- model family: Random Forest
- synthetic training cohort size: `N=600`
- external validation cohort size: `N=56`
- accuracy: `80.36%`
- ROC-AUC: `0.726`
- recall for T2D: `90.9%`

This is the key point of the demo: the model never saw this real dataset, or any real dataset directly, yet it still captures enough disease-relevant structure to identify many real T2D cases.

## External Dataset

Validation data reference:

- [GSE280402](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE280402)

The associated study examines Type 2 Diabetes through bulk RNA sequencing and single-cell analysis. In this repository, it is used only as an external subject-domain validation source after feature overlap alignment.

## Why This Repo Exists

This repository is meant as a small white-paper companion and technical proof of concept for:

- privacy-preserving model development with synthetic cohorts
- testing whether digital twin-derived signals survive the gap between synthetic and physical data
- showing that external validation is possible even when the model was not exposed to real patient datasets during development

## Notes

- This is a focused demonstration, not a clinical diagnostic product.
- The model is intentionally simple to keep the pipeline transparent and reproducible.
- The main value of the repo is the workflow: digital twin-trained model -> feature overlap -> external validation.
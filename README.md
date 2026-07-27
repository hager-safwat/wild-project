# Shill Bidding Detection

Analysis of the Shill Bidding dataset — identifying fraudulent ("shill")
bids in online auctions, where bidders inflate prices artificially.

## What it does
- Explores the data: class distribution, correlations, missing values.
- Detects outliers using z-scores.
- Reduces dimensionality with **PCA** and **t-SNE** to visualise structure.
- Scales features and trains an **SVM** classifier, tuned with grid search.

## File
- `wild_project.py` — full pipeline (exported from a Colab notebook).

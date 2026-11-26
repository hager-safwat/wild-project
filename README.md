# Shill Bidding Detection (SVM)

Team coursework project with **Nicolas Raymond** and **Nassim Matouk**:
detecting fraudulent ("shill") bids in online auctions, where accomplices of
the seller bid to inflate the price artificially.

## What it does

- **Exploration:** class distribution, feature distributions, bidding ratio
  per class, correlation matrix, missing values.
- **Outliers:** z-scores on bidding ratio and successive outbidding.
- **Visualisation:** PCA and t-SNE projections of the scaled features.
- **Model:** an RBF SVM tuned with grid search on C and gamma, evaluated with
  a classification report, confusion matrix and the precision-recall AUC.
- **Extensions ("go wild" part):** SMOTE for class imbalance, Isolation
  Forest outlier removal, Random Forest feature selection, a class-weighted
  SVM and its ROC curve.

## Results

As reported in the team report (`Shill_Bidding_SVM_Report.pdf`):

| Model | Best parameters | Cross-validation score |
|---|---|---|
| SVM | C = 10, gamma = 0.001 | 0.956 |
| Class-weighted SVM | | 0.957 |

The report also gives a precision-recall AUC of 0.859 for the first model.

Notes for anyone reusing the code:

- In the script, SMOTE, outlier removal and feature selection are computed,
  but the final class-weighted SVM is trained on the original training split,
  so those steps do not feed into its score.
- SMOTE is applied before the train / test split; for an unbiased estimate it
  should be applied to the training data only.
- `Bidder_ID` is one-hot encoded, so the same bidder can appear in training
  and test data; a split grouped by bidder would test generalisation to new
  bidders.

## Dataset

[Shill Bidding Dataset](https://archive.ics.uci.edu/dataset/562/shill+bidding+dataset)
(UCI Machine Learning Repository): 6321 bidding records scraped from eBay
auctions, with 9 behavioural features and a label (0 = normal, 1 = shill).
Download `Shill Bidding Dataset.csv` and place it next to the script. It is
not included in this repository.

## Run it

```bash
pip install -r requirements.txt
python wild_project.py "Shill Bidding Dataset.csv"
```

Plots open in windows; use `MPLBACKEND=Agg` to run without a display. The
script was originally a Colab notebook; it has been checked end to end with
current versions of pandas and scikit-learn (the correlation step now uses
numeric columns only, which pandas 2 requires).

## Files

- `wild_project.py`: the full pipeline.
- `Shill_Bidding_SVM_Report.pdf`: team report.
- `requirements.txt`: Python packages.

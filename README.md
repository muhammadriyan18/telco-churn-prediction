# Telco Customer Churn Prediction

I built this project to practice an end-to-end machine learning workflow: take a messy real-world dataset, clean it, compare a few models, tune the best one, and then put it behind a small web app so someone who doesn't write Python can actually use it.

The question is simple: **will a telecom customer leave (churn) or stay?** Losing a customer costs more than keeping one, so if you can spot the ones likely to leave, you get a chance to do something about it.

## What's in this repo

```
├── Telco_Churn_Prediction_Commented.ipynb   # full analysis, every cell has a comment explaining its purpose
├── app.py                                   # Streamlit app (single customer + CSV batch predictions)
├── telco_churn_pipeline.joblib              # saved model (created by running the notebook)
├── requirements.txt
└── README.md
```

The dataset isn't included. It's the IBM Telco Customer Churn data from Kaggle (`WA_Fn-UseC_-Telco-Customer-Churn.csv`, about 7,000 customers and 20 columns). Download it and point the notebook at it.

## How the notebook goes

1. **Loading and cleaning.** `TotalCharges` was stored as text and a few rows were blank, so I converted it to numbers and filled the gaps with the median.
2. **Exploring the data.** About 26% of customers churned, so the classes are imbalanced. Churn is much higher on month-to-month contracts, and it drops steadily as tenure goes up (roughly 47% churn in the first year versus about 7% for customers with 5+ years).
3. **Preprocessing.** A scikit-learn pipeline handles everything: median imputation and scaling for numeric columns, and most-frequent imputation plus one-hot encoding for categorical ones. Putting it all in a pipeline means the same steps run on new data without any manual work.
4. **Comparing models.** I tried a dummy baseline, Logistic Regression, a Decision Tree and a Random Forest on a stratified 80/20 split.
5. **Tuning.** I tuned the Random Forest with `RandomizedSearchCV` (3-fold CV), optimising F1 for the churn class rather than accuracy.
6. **Evaluation.** Confusion matrix, ROC curve and permutation importance on the test set.
7. **Saving.** The final pipeline is saved with `joblib`.

## Results

Test set of 1,409 customers. Precision, recall and F1 are for the churn ("Yes") class.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Dummy baseline | 0.735 | 0.000 | 0.000 | 0.000 |
| Decision Tree | 0.732 | 0.495 | 0.487 | 0.491 |
| Random Forest (default) | 0.764 | 0.548 | 0.620 | 0.582 |
| Logistic Regression | 0.738 | 0.504 | 0.783 | 0.614 |
| **Random Forest (tuned)** | 0.752 | 0.523 | 0.762 | 0.620 |

The tuned Random Forest has a ROC-AUC of **0.839**.

A few honest notes on these numbers:

- Accuracy looks decent but is misleading here. The dummy model that always says "no churn" scores 73.5% just because most customers stay. That's why I looked at F1 and recall for the churn class instead.
- The tuned Random Forest gives up a little accuracy to catch more churners (recall went from 0.62 to 0.76). If the goal is finding customers at risk, that's usually a trade worth making, but it also means more false alarms.
- Plain Logistic Regression ended up very close to the tuned forest. A fancier model isn't automatically a better one on this dataset.

**What drives churn the most:** contract type was by far the most important feature, followed by internet service type and tenure.

## The Streamlit app

The app loads the saved pipeline and has two tabs:

- **Single customer:** fill in a form with the customer's details and get the churn probability, a Will churn / Will stay prediction, and a Low / Medium / High risk label (under 35%, 35-60%, above 60%).
- **Batch (CSV):** upload a file with the same columns as the original dataset, get predictions for every row, and download the results as a CSV.

To run it:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Make sure `telco_churn_pipeline.joblib` is in the same folder as `app.py`. If it isn't there, the app will tell you.

## Running the notebook yourself

1. Download the dataset from Kaggle and unzip it.
2. Open the notebook and update `zip_path` in the extraction cell to wherever your file is (or just load the CSV directly).
3. Run all cells. The last few cells save the model file that the app needs.

Use the same scikit-learn version for training and for the app. Saved models can break or give warnings if the versions don't match.

## Limitations

- This is one dataset from one company, so I wouldn't expect the model to carry over to a different telecom without retraining.
- The risk thresholds (35% and 60%) are just reasonable-looking cut-offs I picked, not something I tuned against a business cost.
- I didn't test other models like gradient boosting or SVM in depth, so there's probably room to improve.

## Tools used

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, joblib, Streamlit.

## Dataset

IBM Telco Customer Churn, available on Kaggle.

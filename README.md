# Customer Churn Prediction

> **College project.** Built as coursework to practise an end-to-end machine learning
> workflow: exploratory analysis, feature selection, model comparison, and deployment
> behind a small web app. It is a learning exercise, not a production system.
>
> Forked from [codebrain001/customer-churn-prediction](https://github.com/codebrain001/customer-churn-prediction).

A Streamlit app that predicts whether a telecom customer is likely to churn, using a
gradient boosting model trained on the Telco customer churn dataset (7,043 customers).
It is tuned to catch churners rather than to maximise raw accuracy — see
[Results](#results).

![Demo of the Streamlit app](streamlit-app.gif)

---

## What it does

Churn rate is a key indicator for subscription businesses. Identifying customers who
are unhappy with the service points at weaknesses in the product or pricing plan and
allows a company to act before the customer leaves.

The app supports the two ways such a model normally gets used:

- **Online prediction** — fill in one customer's details in a form and get a single
  prediction back, with the estimated probability of churn.
- **Batch prediction** — upload a CSV of customers and get a prediction and probability
  for every row.

---

## Running it

Requires Python 3.9 or newer.

**1. Clone the repository and enter it**

```bash
git clone https://github.com/happyc0der/customer-churn-prediction.git && cd customer-churn-prediction
```

**2. Create a virtual environment and install the dependencies**

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` instead.

**3. Start the app**

```bash
streamlit run app.py
```

Streamlit prints a local URL (usually <http://localhost:8501>) and opens it in your
browser. Run the command from the repository root so the app can find the saved model.

### Trying a prediction

- **Online** — leave the sidebar on `Online`, fill in the form, and press **Predict**.
- **Batch** — switch the sidebar to `Batch`, upload a CSV, and press **Predict**.
  `data/batch_churn.csv` is a small sample you can use; `data/churn.csv` (the full
  dataset) also works. The uploaded file must contain these columns:

  `SeniorCitizen`, `Dependents`, `tenure`, `PhoneService`, `MultipleLines`,
  `InternetService`, `OnlineSecurity`, `OnlineBackup`, `TechSupport`, `StreamingTV`,
  `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`,
  `MonthlyCharges`, `TotalCharges`

  `SeniorCitizen` may be either `Yes`/`No` or `1`/`0`.

### Retraining the model

```bash
python train.py
```

This rebuilds `notebook/model.sav`. See [How the model was built](#how-the-model-was-built).

### Re-running the analysis notebook

The notebook holds the exploratory analysis and the original baseline model. It needs a
few extra packages:

```bash
pip install -r requirements-notebook.txt && jupyter lab notebook/EDA.ipynb
```

Run the cells in order — later cells depend on transformations applied by earlier ones.
The hyperparameter search cell fits 500 candidates across 30 cross-validation folds and
takes a while. The notebook does not overwrite the deployed model; `train.py` produces
that.

---

## Project structure

| Path | What it is |
| --- | --- |
| `app.py` | Streamlit app — the online and batch prediction screens |
| `train.py` | Trains the model and writes `notebook/model.sav` |
| `preprocessing.py` | Prepares raw feature columns; shared by `train.py` and the app |
| `notebook/EDA.ipynb` | Exploratory analysis and the original baseline model |
| `notebook/model.sav` | The trained pipeline (preprocessing + model), saved with `joblib` |
| `data/churn.csv` | The Telco churn dataset used for training (7,043 rows) |
| `data/batch_churn.csv` | Small sample file for trying out batch prediction |
| `requirements.txt` | Dependencies for the app and for retraining |
| `requirements-notebook.txt` | Extra dependencies for re-running the notebook |

---

## How the model was built

`train.py` is the training pipeline. Run it from the repository root to
reproduce `notebook/model.sav`:

```bash
python train.py
```

It takes a few minutes. Add `--quick` for a reduced search, or `--no-save` to
print the results without overwriting the model.

What it does:

1. **Holds out a stratified 20% test set** up front, and touches it exactly once
   at the end.
2. **Puts every preprocessing step inside the pipeline** — median imputation for
   the blank `TotalCharges` values, standard scaling, and one-hot encoding — so
   each step is fitted on the training fold only.
3. **Searches nine classical models** with randomised hyperparameter search and
   5-fold cross-validation: logistic regression, linear SVM, RBF SVM, random
   forest, extra trees, gradient boosting, histogram gradient boosting, k-nearest
   neighbours, and naive Bayes.
4. **Selects on average precision**, not accuracy. Only ~26.5% of customers churn,
   so a model that never predicts churn already scores 73.5% accuracy.
5. **Tunes the decision threshold** for F1 by cross-validation on the training
   set. The threshold is saved with the model, so `model.predict()` uses it.

Gradient boosting won, with a chosen threshold of 0.33 rather than the default 0.5.

### Results

On the held-out test set (1,409 customers, 374 of whom churned):

| | Old model | **Current model** |
| --- | --- | --- |
| | Logistic regression | **Gradient boosting** |
| Accuracy | 0.805 | 0.772 |
| Balanced accuracy | 0.732 | **0.759** |
| Precision (churn) | 0.649 | 0.554 |
| Recall (churn) | 0.578 | **0.730** |
| F1 (churn) | 0.611 | **0.630** |
| ROC AUC | 0.844 | 0.845 |
| PR AUC | 0.640 | **0.656** |

In plain terms, out of 374 customers who actually left:

- the old model caught **216** and missed 158, raising 117 false alarms
- the current model catches **273** and misses 101, raising 220 false alarms

That is the trade this model makes deliberately. Accuracy *falls*, because
predicting churn more readily costs some precision, but the model finds 57 more
of the customers who were about to leave. For a retention campaign that is
usually the better error to make: contacting a happy customer is cheap, losing
one silently is not.

Two honest caveats:

- **Most of the gain comes from the threshold, not the algorithm.** ROC AUC barely
  moved (0.844 → 0.845), which means the models rank customers by risk about
  equally well. On this dataset the classical algorithms are within noise of each
  other — gradient boosting only edges logistic regression on PR AUC
  (0.675 vs 0.661 in cross-validation). The real gain was moving off the 0.5
  cut-off.
- **The old model's numbers here are, if anything, flattering.** It was trained on
  a 70/30 split of the full dataset, so most of this test set was in its training
  data. The comparison still favours the new model.

For reference, all nine models by cross-validated average precision:

| Model | Avg. precision |
| --- | --- |
| **Gradient boosting** | **0.675** |
| Histogram gradient boosting | 0.667 |
| Random forest | 0.666 |
| Logistic regression | 0.661 |
| RBF SVM | 0.660 |
| Linear SVM | 0.659 |
| Extra trees | 0.656 |
| K-nearest neighbours | 0.633 |
| Naive Bayes | 0.617 |

---

## Notes

`notebook/model.sav` is a complete scikit-learn pipeline — it takes the raw
feature columns and does its own imputation, scaling and encoding. Keeping
preprocessing inside the saved model is deliberate: when the encoding lives in
two places it drifts, and the model ends up scoring features that do not mean
what it was trained on.

`preprocessing.py` is the single entry point that prepares raw columns for the
model, used by both `train.py` and the app, for the same reason.

The notebook holds the exploratory analysis and the original baseline. Its
scaling is fitted before the train/test split, so its reported scores are mildly
optimistic; `train.py` supersedes it for training the deployed model.

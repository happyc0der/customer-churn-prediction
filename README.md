# Customer Churn Prediction

> **College project.** Built as coursework to practise an end-to-end machine learning
> workflow: exploratory analysis, feature selection, model comparison, and deployment
> behind a small web app. It is a learning exercise, not a production system.
>
> Forked from [codebrain001/customer-churn-prediction](https://github.com/codebrain001/customer-churn-prediction).

A Streamlit app that predicts whether a telecom customer is likely to churn, using a
logistic regression model trained on the Telco customer churn dataset (7,043 customers).

![Demo of the Streamlit app](streamlit-app.gif)

---

## What it does

Churn rate is a key indicator for subscription businesses. Identifying customers who
are unhappy with the service points at weaknesses in the product or pricing plan and
allows a company to act before the customer leaves.

The app supports the two ways such a model normally gets used:

- **Online prediction** — fill in one customer's details in a form and get a single
  prediction back.
- **Batch prediction** — upload a CSV of customers and get a prediction for every row.

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

### Re-running the analysis notebook

The notebook contains the exploratory analysis, feature selection, and model training.
It needs a few extra packages:

```bash
pip install -r requirements-notebook.txt && jupyter lab notebook/EDA.ipynb
```

Run the cells in order — later cells depend on transformations applied by earlier ones.
The hyperparameter search cell fits 500 candidates across 30 cross-validation folds and
takes a while. Re-running the final cell overwrites `notebook/model.sav`.

---

## Project structure

| Path | What it is |
| --- | --- |
| `app.py` | Streamlit app — the online and batch prediction screens |
| `preprocessing.py` | Feature encoding and scaling shared by the app and the notebook |
| `notebook/EDA.ipynb` | Exploratory analysis, feature selection, model training |
| `notebook/model.sav` | The trained logistic regression model, saved with `joblib` |
| `data/churn.csv` | The Telco churn dataset used for training (7,043 rows) |
| `data/batch_churn.csv` | Small sample file for trying out batch prediction |
| `requirements.txt` | Dependencies for running the app |
| `requirements-notebook.txt` | Extra dependencies for re-running the notebook |

---

## How the model was built

1. **Exploratory analysis** of the target and each feature group (demographics,
   services signed up for, payment details) to see which ones separate churners from
   non-churners.
2. **Preprocessing** — dropped `customerID`, coerced `TotalCharges` to numeric and
   filled its blanks with the column median, encoded the categorical features, and
   min-max scaled `tenure`, `MonthlyCharges`, and `TotalCharges`.
3. **Feature selection** with recursive feature elimination (`RFECV`), which narrowed
   30 encoded features down to the 23 used by the model.
4. **Model comparison** across logistic regression, SVC, random forest, decision tree,
   and naive Bayes.
5. **Hyperparameter tuning** of the best performer with a randomised search.

Logistic regression came out ahead:

| Model | Accuracy |
| --- | --- |
| **Logistic regression (tuned)** | **0.803** |
| Logistic regression (baseline) | 0.802 |
| SVC | 0.799 |
| Random forest | 0.780 |
| Decision tree | 0.726 |
| Naive Bayes | 0.654 |

Accuracy alone flatters the model here, because only ~26.5% of customers actually
churned — always predicting "no churn" would already score about 73.5%. Recall on the
churn class is the weaker number at roughly 0.58, so the model misses a fair share of
the customers who do leave.

---

## Notes

`notebook/model.sav` was pickled with scikit-learn 0.24.1. Loading it with a newer
scikit-learn works but prints an `InconsistentVersionWarning`; re-run the notebook's
final cell to regenerate the file with your installed version.

The scaling ranges in `preprocessing.py` are the min/max of the training data, so the
app scales new inputs exactly the way the model was trained. They need to be updated if
the model is ever retrained on a different dataset.

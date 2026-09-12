"""Raw-feature preparation shared by `train.py` and the Streamlit app.

The saved model (`notebook/model.sav`) is a scikit-learn pipeline that does
its own imputation, scaling and one-hot encoding. So the only job left here is
to hand it the raw columns in a consistent shape, whether they came from the
app's form or from an uploaded CSV.

Keeping this the single entry point for both training and serving is
deliberate: when encoding lives in two places it drifts, and the model then
silently scores features that do not mean what it was trained on.
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / 'data' / 'churn.csv'
MODEL_PATH = BASE_DIR / 'notebook' / 'model.sav'

# The raw features the model is trained on. These are exactly the fields the
# app's online form collects, so both entry points agree.
RAW_COLUMNS = [
    'SeniorCitizen', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
    'PaymentMethod', 'MonthlyCharges', 'TotalCharges',
]

NUMERIC_COLUMNS = ['tenure', 'MonthlyCharges', 'TotalCharges']


def prepare_features(df, option):
    """Return the raw feature columns, cleaned and consistently typed.

    `option` is "Online" (a single row from the form) or "Batch" (an uploaded
    CSV). The pipeline inside the saved model handles everything else.
    """
    if option not in ('Online', 'Batch'):
        raise ValueError(f"Unknown option {option!r}, expected 'Online' or 'Batch'")

    missing = [column for column in RAW_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f'Input is missing required columns: {missing}')

    df = df[RAW_COLUMNS].copy()

    # churn.csv stores SeniorCitizen as 0/1 while the app's form uses Yes/No.
    # Normalise to Yes/No so the encoder sees one vocabulary.
    df['SeniorCitizen'] = df['SeniorCitizen'].replace({0: 'No', 1: 'Yes', '0': 'No', '1': 'Yes'})

    # TotalCharges is blank for customers in their first month. Leave the gap
    # as NaN -- the imputer in the pipeline fills it using the training median.
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    return df

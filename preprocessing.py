"""Raw-feature preparation shared by `train.py` and the Streamlit app.

Each saved model in `models/` is a scikit-learn pipeline that does its own
imputation, scaling and one-hot encoding. So the only job left here is
to hand it the raw columns in a consistent shape, whether they came from the
app's form or from an uploaded CSV.

Keeping this the single entry point for both training and serving is
deliberate: when encoding lives in two places it drifts, and the model then
silently scores features that do not mean what it was trained on.
"""

import json
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / 'data' / 'churn.csv'

# `train.py` writes one fitted pipeline per algorithm into models/, plus an
# index describing them. The app reads the index to populate its model picker.
MODELS_DIR = BASE_DIR / 'models'
INDEX_PATH = MODELS_DIR / 'index.json'
PLOT_PATH = BASE_DIR / 'docs' / 'model_comparison.png'

# The raw features the model is trained on. These are exactly the fields the
# app's online form collects, so both entry points agree.
RAW_COLUMNS = [
    'SeniorCitizen', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
    'PaymentMethod', 'MonthlyCharges', 'TotalCharges',
]

NUMERIC_COLUMNS = ['tenure', 'MonthlyCharges', 'TotalCharges']


# Spellings of SeniorCitizen seen in the wild. Series.replace does not cross
# the bool/int boundary on a bool-dtype column, so True/False reached the
# encoder untouched and raised "y contains previously unseen labels"; mapping
# element-wise handles every case, including "yes" in the wrong case.
_YES_NO = {
    # True and 1 are the same dict key (True == 1), as are False and 0.
    True: 'Yes', False: 'No',
    '1': 'Yes', '0': 'No',
    'yes': 'Yes', 'no': 'No',
    'true': 'Yes', 'false': 'No',
    'y': 'Yes', 'n': 'No',
}


def _to_yes_no(value):
    """Map one SeniorCitizen value to Yes/No, leaving anything odd untouched.

    Unrecognised values are passed through rather than guessed at, so
    `unknown_values` can report them instead of silently changing a
    prediction.
    """
    if pd.isna(value):
        return value
    key = value.strip().lower() if isinstance(value, str) else value
    try:
        return _YES_NO.get(key, value)
    except TypeError:      # unhashable, e.g. a list in a malformed CSV
        return value


def _fitted_pipeline(model):
    """The pipeline inside a threshold-tuned model."""
    return getattr(model, 'estimator_', model)


def known_categories(model):
    """The category vocabulary the model's encoder learned during training."""
    encoder = _fitted_pipeline(model).named_steps['prep'].named_transformers_['cat']
    return {name: encoder.categories_[i]
            for i, name in enumerate(encoder.feature_names_in_)}


def unknown_values(df, model):
    """Values in `df` that the model never saw, keyed by column.

    The encoder is built with handle_unknown='ignore', which means an
    unrecognised category is quietly encoded as all-zeros and the row is
    scored as if it held the reference category. That is a wrong answer with
    no error attached -- a "Month to month" typo moves a churn probability by
    ten points -- so callers should surface whatever this reports.
    """
    found = {}
    for column, categories in known_categories(model).items():
        if column not in df.columns:
            continue
        known = set(categories)
        extra = sorted({str(v) for v in df[column].dropna().unique()
                        if v not in known})
        if df[column].isna().any():
            extra.append('(blank)')
        if extra:
            found[column] = extra
    return found


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
    df['SeniorCitizen'] = df['SeniorCitizen'].map(_to_yes_no)

    # TotalCharges is blank for customers in their first month. Leave the gap
    # as NaN -- the imputer in the pipeline fills it using the training median.
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    return df


def load_index():
    """Read the model gallery index written by `train.py`."""
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f'No model gallery found at {INDEX_PATH}. Run `python train.py` first.')
    return json.loads(INDEX_PATH.read_text())


def load_model(slug):
    """Load one fitted pipeline from the gallery by its slug."""
    path = MODELS_DIR / f'{slug}.joblib'
    if not path.exists():
        raise FileNotFoundError(
            f'No model file at {path}. Run `python train.py` first.')
    return joblib.load(path)

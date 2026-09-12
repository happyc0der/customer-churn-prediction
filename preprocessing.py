"""Preprocessing shared by the Streamlit app and the training notebook.

The steps here mirror the ones performed in `notebook/EDA.ipynb` before the
model was trained: select the features chosen by RFE, encode the categorical
columns, and min-max scale the three numeric columns.
"""

import pandas as pd

# The 23 features selected by recursive feature elimination in the notebook,
# in the exact order the saved model expects them.
FEATURE_COLUMNS = [
    'SeniorCitizen', 'Dependents', 'tenure', 'PhoneService', 'PaperlessBilling',
    'MonthlyCharges', 'TotalCharges', 'MultipleLines_No_phone_service',
    'MultipleLines_Yes', 'InternetService_Fiber_optic', 'InternetService_No',
    'OnlineSecurity_No_internet_service', 'OnlineSecurity_Yes',
    'OnlineBackup_No_internet_service', 'TechSupport_No_internet_service',
    'TechSupport_Yes', 'StreamingTV_No_internet_service', 'StreamingTV_Yes',
    'StreamingMovies_No_internet_service', 'StreamingMovies_Yes',
    'Contract_One_year', 'Contract_Two_year', 'PaymentMethod_Electronic_check',
]

# Raw columns kept from an uploaded batch file before encoding.
RAW_COLUMNS = [
    'SeniorCitizen', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
    'PaymentMethod', 'MonthlyCharges', 'TotalCharges',
]

# Min/max of the numeric columns in the full training set (data/churn.csv).
# The scaler in the notebook was fitted on these values, so scaling at
# prediction time has to reuse them rather than refitting on the input.
TRAINING_RANGES = {
    'tenure': (0.0, 72.0),
    'MonthlyCharges': (18.25, 118.75),
    'TotalCharges': (18.8, 8684.8),
}

BINARY_COLUMNS = ['SeniorCitizen', 'Dependents', 'PhoneService', 'PaperlessBilling']


def binary_map(feature):
    """Map a yes/no column to 1/0.

    `data/churn.csv` stores SeniorCitizen as 0/1 while the sample batch file
    stores it as Yes/No, so both spellings are accepted.
    """
    return feature.map({'Yes': 1, 'No': 0, 1: 1, 0: 0})


def normalize_column(column):
    """Replace the separators the notebook replaced when it renamed columns."""
    for character in (' ', '(', ')', '-'):
        column = column.replace(character, '_')
    return column


def preprocess(df, option):
    """Turn a raw churn dataframe into the feature matrix the model expects.

    `option` is either "Online" (a single row built from the form inputs) or
    "Batch" (a full uploaded CSV).
    """
    if option not in ('Online', 'Batch'):
        raise ValueError(f"Unknown option {option!r}, expected 'Online' or 'Batch'")

    df = df.copy()

    if option == 'Batch':
        missing = [c for c in RAW_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Uploaded file is missing required columns: {missing}")
        df = df[RAW_COLUMNS]
        # TotalCharges has blank entries in the raw dataset; the notebook fills
        # them with the column median before training, so do the same here.
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # Encode the binary categorical features, then one-hot encode the rest.
    df[BINARY_COLUMNS] = df[BINARY_COLUMNS].apply(binary_map)
    df = pd.get_dummies(df)

    # get_dummies names columns after the category values, so a category like
    # "Fiber optic" becomes "InternetService_Fiber optic". The notebook renamed
    # those separators to underscores before training, so apply the same rule
    # here -- otherwise reindex silently fills the mismatched columns with 0.
    df.columns = [normalize_column(column) for column in df.columns]
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    # Scale using the training ranges. Refitting a scaler on the input instead
    # would map every value of a single-row prediction to 0.
    for column, (minimum, maximum) in TRAINING_RANGES.items():
        df[column] = (df[column] - minimum) / (maximum - minimum)

    return df

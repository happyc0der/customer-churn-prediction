"""Tests for the saved model gallery.

These are the regression guards for the bugs that made the app's predictions
disagree with the model that was trained.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score

from preprocessing import MODELS_DIR, prepare_features

# A customer the model should read as high risk: new, on fibre, month to month.
RISKY = {
    'SeniorCitizen': 'No', 'Dependents': 'No', 'tenure': 1,
    'PhoneService': 'Yes', 'MultipleLines': 'No', 'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No', 'OnlineBackup': 'No', 'TechSupport': 'No',
    'StreamingTV': 'No', 'StreamingMovies': 'No', 'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes', 'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 80, 'TotalCharges': 80,
}


def one(**overrides):
    return pd.DataFrame.from_dict([{**RISKY, **overrides}])


def test_index_and_files_agree(index):
    on_disk = {path.stem for path in MODELS_DIR.glob('*.joblib')}
    listed = {entry['slug'] for entry in index['models']}
    assert on_disk == listed


def test_best_is_in_the_gallery(index):
    assert index['best'] in {entry['slug'] for entry in index['models']}


def test_models_are_ranked_best_first(index):
    scores = [entry['cv_score'] for entry in index['models']]
    assert scores == sorted(scores, reverse=True)
    assert index['models'][0]['slug'] == index['best']


def test_display_names_are_unique(index):
    """The app's picker looks models up by name."""
    names = [entry['name'] for entry in index['models']]
    assert len(names) == len(set(names))


# One search pick once scored f1 0.000 while still reporting ROC AUC 0.831: it
# had collapsed to predicting a single class.
def test_no_model_collapses_to_one_class(models, training_frame):
    prepared = prepare_features(training_frame, 'Batch')
    for slug, model in models.items():
        assert len(np.unique(model.predict(prepared))) == 2, f'{slug} predicts one class'


# Every model carries its own tuned threshold. If predict() stopped honouring
# it, the app's "flags a customer above X%" caption would become a lie.
def test_verdict_follows_each_models_threshold(models, index, sample):
    prepared = prepare_features(sample, 'Batch')
    for entry in index['models']:
        model = models[entry['slug']]
        probability = model.predict_proba(prepared)[:, 1]
        expected = (probability >= entry['threshold']).astype(int)
        np.testing.assert_array_equal(model.predict(prepared), expected,
                                      err_msg=f"{entry['slug']} ignores its threshold")


# The scaler used to be refitted on the input. For a single row that maps every
# numeric feature to 0, so tenure made no difference at all.
def test_tenure_changes_a_single_row_prediction(best_model):
    new = best_model.predict_proba(prepare_features(one(tenure=1), 'Online'))[0][1]
    old = best_model.predict_proba(
        prepare_features(one(tenure=60, TotalCharges=5000, Contract='Two year'),
                         'Online'))[0][1]
    assert new > old + 0.2, 'tenure and contract barely moved the prediction'


def test_numeric_features_are_not_flattened(best_model):
    """Two different customers must not encode to the same numbers."""
    a = prepare_features(one(tenure=1, MonthlyCharges=20, TotalCharges=20), 'Online')
    b = prepare_features(one(tenure=70, MonthlyCharges=110, TotalCharges=7700), 'Online')
    prep = best_model.estimator_.named_steps['prep']
    assert not np.allclose(prep.transform(a), prep.transform(b))


# 10 of 23 features were once silently all-zero, because the encoder emitted
# "InternetService_Fiber optic" while the model wanted underscores.
def test_no_encoded_feature_is_always_zero(best_model, training_frame):
    prepared = prepare_features(training_frame, 'Batch')
    encoded = best_model.estimator_.named_steps['prep'].transform(prepared)
    encoded = np.asarray(encoded)
    dead = [i for i in range(encoded.shape[1]) if not encoded[:, i].any()]
    assert dead == [], f'columns {dead} are zero for every customer'


# Scoring the labelled data end to end used to crash outright on SeniorCitizen
# being 0/1 and TotalCharges being blank.
def test_scores_the_whole_training_set(best_model, training_frame):
    prepared = prepare_features(training_frame, 'Batch')
    predicted = best_model.predict(prepared)
    actual = training_frame['Churn'].eq('Yes').astype(int)

    assert len(predicted) == len(training_frame)
    # Comfortably better than always guessing the majority class (0.735).
    assert accuracy_score(actual, predicted) > 0.75


@pytest.mark.parametrize('rows', [1, 3])
def test_small_batches_work(best_model, sample, rows):
    prepared = prepare_features(sample.head(rows), 'Batch')
    assert len(best_model.predict(prepared)) == rows

"""Tests for the shaping step that both training and the app go through.

Most of these pin a bug that actually shipped in this repository.
"""

import numpy as np
import pandas as pd
import pytest

from preprocessing import (
    RAW_COLUMNS,
    implausible_charges,
    prepare_features,
    unknown_values,
)


def test_missing_columns_are_named(training_frame):
    with pytest.raises(ValueError, match='missing required columns'):
        prepare_features(training_frame[['tenure']], 'Batch')


@pytest.mark.parametrize('option', ['Streaming', 'online', '', None])
def test_unknown_option_is_rejected(sample, option):
    with pytest.raises(ValueError, match='Unknown option'):
        prepare_features(sample, option)


# SeniorCitizen is spelled 0/1 in churn.csv and Yes/No in the app's own form.
# Booleans used to crash with "y contains previously unseen labels" because
# Series.replace does not cross the bool/int boundary on a bool column.
@pytest.mark.parametrize('mapping', [
    None,                               # Yes / No, as shipped
    {'Yes': 1, 'No': 0},                # ints, as in churn.csv
    {'Yes': 1.0, 'No': 0.0},            # floats, from a column with blanks
    {'Yes': True, 'No': False},         # booleans
    {'Yes': 'yes', 'No': 'no'},         # lower case
    {'Yes': '1', 'No': '0'},            # digits as text
    {'Yes': ' Yes ', 'No': ' No '},     # padded
])
def test_senior_citizen_spellings_all_agree(sample, best_model, mapping):
    expected = best_model.predict_proba(prepare_features(sample, 'Batch'))[:, 1]

    frame = sample.copy()
    if mapping:
        frame['SeniorCitizen'] = frame['SeniorCitizen'].map(mapping)
    actual = best_model.predict_proba(prepare_features(frame, 'Batch'))[:, 1]

    np.testing.assert_allclose(actual, expected)


def test_blank_total_charges_is_left_for_the_imputer(training_frame):
    """The pipeline's imputer fills these using the training median."""
    prepared = prepare_features(training_frame, 'Batch')
    assert prepared['TotalCharges'].isna().sum() == 11
    assert pd.api.types.is_numeric_dtype(prepared['TotalCharges'])


def test_extra_columns_are_dropped(sample):
    frame = sample.copy()
    frame['some_unrelated_column'] = 'ignore me'
    assert list(prepare_features(frame, 'Batch').columns) == RAW_COLUMNS


def test_input_frame_is_not_mutated(sample):
    before = sample.copy()
    prepare_features(sample, 'Batch')
    pd.testing.assert_frame_equal(sample, before)


# The encoder ignores categories it never saw, which silently scores the row as
# the reference category. A "Month to month" typo moved a churn probability by
# ten points with no error attached.
def test_unrecognised_categories_are_reported(sample, best_model):
    frame = sample.copy()
    frame.loc[0, 'Contract'] = 'Month to month'
    frame.loc[1, 'InternetService'] = 'fibre optic'

    reported = unknown_values(prepare_features(frame, 'Batch'), best_model)

    assert reported == {'Contract': ['Month to month'],
                        'InternetService': ['fibre optic']}


@pytest.mark.parametrize('value', ['', '   ', 'Yess'])
def test_blank_and_typo_values_are_reported(sample, best_model, value):
    frame = sample.copy()
    frame['SeniorCitizen'] = frame['SeniorCitizen'].astype(object)
    frame.at[0, 'SeniorCitizen'] = value

    reported = unknown_values(prepare_features(frame, 'Batch'), best_model)

    assert reported == {'SeniorCitizen': [value]}


def test_missing_value_is_reported_as_blank(sample, best_model):
    frame = sample.copy()
    frame['SeniorCitizen'] = frame['SeniorCitizen'].astype(object)
    frame.at[0, 'SeniorCitizen'] = np.nan

    assert unknown_values(prepare_features(frame, 'Batch'), best_model) == {
        'SeniorCitizen': ['(blank)']}


def test_clean_files_report_nothing(sample, training_frame, best_model):
    assert unknown_values(prepare_features(sample, 'Batch'), best_model) == {}
    assert unknown_values(prepare_features(training_frame, 'Batch'), best_model) == {}


def test_contradictory_charges_are_flagged(sample):
    frame = sample.copy()
    frame['TotalCharges'] = frame['TotalCharges'].astype(float)
    frame.loc[0, 'tenure'] = 0
    frame.loc[0, 'TotalCharges'] = 10_000

    flagged = dict(implausible_charges(prepare_features(frame, 'Batch')))

    assert 0 in flagged
    assert 'tenure is 0 months' in flagged[0]


def test_charge_check_is_quiet_on_real_data(training_frame, sample):
    """The bounds come from the training data, so they must not fire on it.

    They are worth nothing if they cry wolf on the very customers the model
    was fitted to.
    """
    assert implausible_charges(prepare_features(training_frame, 'Batch')) == []
    assert implausible_charges(prepare_features(sample, 'Batch')) == []

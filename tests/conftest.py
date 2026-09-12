"""Shared fixtures.

Loading the model gallery is the slow part, so anything that touches disk is
session-scoped.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from preprocessing import load_index, load_model  # noqa: E402


@pytest.fixture(scope='session')
def root():
    return ROOT


@pytest.fixture(scope='session')
def index():
    return load_index()


@pytest.fixture(scope='session')
def models(index):
    """Every model in the gallery, keyed by slug."""
    return {entry['slug']: load_model(entry['slug']) for entry in index['models']}


@pytest.fixture(scope='session')
def best_model(models, index):
    return models[index['best']]


@pytest.fixture(scope='session')
def training_frame():
    """The full labelled dataset."""
    return pd.read_csv(ROOT / 'data' / 'churn.csv')


@pytest.fixture
def sample():
    """The small batch file shipped for trying the app out."""
    return pd.read_csv(ROOT / 'data' / 'batch_churn.csv')

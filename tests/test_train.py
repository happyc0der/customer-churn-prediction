"""Tests for the training script.

`--only` once called shutil.rmtree on models/ and then wrote back just the
model it had trained, so `--only naive_bayes` deleted the other eight and
crowned naive Bayes "best" -- leaving the app defaulting to the worst
algorithm in the study.
"""

import json
import shutil
import subprocess
import sys

import pytest


@pytest.fixture
def workspace(root, tmp_path):
    """A throwaway copy of everything train.py touches."""
    for name in ['preprocessing.py', 'train.py']:
        shutil.copy(root / name, tmp_path / name)
    shutil.copytree(root / 'data', tmp_path / 'data')
    shutil.copytree(root / 'models', tmp_path / 'models')
    return tmp_path


def run_train(workspace, *args):
    result = subprocess.run(
        [sys.executable, 'train.py', *args],
        cwd=workspace, capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def read_index(workspace):
    return json.loads((workspace / 'models' / 'index.json').read_text())


def test_index_records_the_library_versions(index):
    """Pickles are tied to the library that wrote them."""
    assert index['sklearn_version']
    assert index['python_version']


@pytest.mark.slow
def test_only_keeps_the_rest_of_the_gallery(workspace):
    before = {path.name: path.read_bytes()
              for path in (workspace / 'models').glob('*.joblib')}
    best_before = read_index(workspace)['best']
    assert len(before) > 1 and best_before != 'naive_bayes'

    run_train(workspace, '--only', 'naive_bayes', '--quick')

    after = {path.name: path.read_bytes()
             for path in (workspace / 'models').glob('*.joblib')}
    assert set(after) == set(before), 'a partial run changed which models exist'

    # Everything except the retrained model must be untouched, and the ranking
    # must still be computed over the whole gallery.
    for name, payload in before.items():
        if name != 'naive_bayes.joblib':
            assert after[name] == payload, f'{name} was rewritten'
    assert read_index(workspace)['best'] == best_before
    assert len(read_index(workspace)['models']) == len(before)


@pytest.mark.slow
def test_no_save_leaves_the_gallery_alone(workspace):
    before = {path.name: path.read_bytes()
              for path in (workspace / 'models').glob('*.joblib')}
    index_before = (workspace / 'models' / 'index.json').read_bytes()

    run_train(workspace, '--only', 'naive_bayes', '--quick', '--no-save')

    after = {path.name: path.read_bytes()
             for path in (workspace / 'models').glob('*.joblib')}
    assert after == before
    assert (workspace / 'models' / 'index.json').read_bytes() == index_before


def test_unknown_slug_is_rejected(workspace):
    result = subprocess.run(
        [sys.executable, 'train.py', '--only', 'no_such_model'],
        cwd=workspace, capture_output=True, text=True, timeout=300)
    assert result.returncode != 0
    assert 'unknown slug' in (result.stdout + result.stderr).lower()

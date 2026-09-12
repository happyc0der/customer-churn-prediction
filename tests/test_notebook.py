"""Checks on the analysis notebook.

It shipped with a cell that could not run at all: a bare `summarize result`
line, left over from a comment, made it a SyntaxError.
"""

import ast
import json

import pytest


@pytest.fixture(scope='module')
def notebook(root):
    return json.loads((root / 'notebook' / 'EDA.ipynb').read_text())


def code_cells(notebook):
    return [(i, ''.join(cell['source']))
            for i, cell in enumerate(notebook['cells'])
            if cell['cell_type'] == 'code']


def test_every_code_cell_parses(notebook):
    broken = []
    for index, source in code_cells(notebook):
        try:
            ast.parse(source)
        except SyntaxError as error:
            broken.append(f'cell {index}: {error}')
    assert broken == []


def test_no_empty_cells(notebook):
    empty = [index for index, source in code_cells(notebook) if not source.strip()]
    assert empty == []


def test_notebook_does_not_overwrite_the_deployed_models(notebook):
    """Its final cell saves under its own name; train.py owns models/."""
    sources = [source for _, source in code_cells(notebook)]
    saves = [s for s in sources if 'joblib.dump' in s]
    assert saves, 'expected the notebook to still save its baseline'
    for source in saves:
        assert 'model_notebook_baseline.sav' in source


def test_figures_survive_for_github(notebook):
    """Charts are kept as PNG, which is what GitHub renders.

    The plotly HTML copies were stripped to cut 4 MB; the PNGs are what is
    left, so losing them would silently empty the notebook on GitHub.
    """
    pngs = sum(1 for cell in notebook['cells']
               for output in cell.get('outputs', [])
               if 'image/png' in (output.get('data') or {}))
    assert pngs >= 20


def test_the_plotly_bundle_stays_out(root):
    """3.5 MB of inlined plotly.js does not belong in version control."""
    size_mb = (root / 'notebook' / 'EDA.ipynb').stat().st_size / 1e6
    assert size_mb < 3.5, f'notebook has grown back to {size_mb:.1f} MB'

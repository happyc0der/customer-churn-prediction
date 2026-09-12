"""Train every candidate model and write the model gallery in `models/`.

Run from the repository root:

    python train.py

This trains each classical algorithm the project covers, tunes a decision
threshold for each one, scores them all on the same held-out test set, and
saves every fitted pipeline so the app can predict with any of them.

Each saved artifact is a complete scikit-learn pipeline: it takes the raw
feature columns (see preprocessing.RAW_COLUMNS) and handles imputation,
scaling and one-hot encoding internally. Keeping preprocessing inside the
pipeline is what stops the training and serving paths from drifting apart.
"""

import argparse
import json
import shutil
import time
import warnings
from datetime import date

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    TunedThresholdClassifierCV,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC, LinearSVC
from scipy.stats import loguniform, randint, uniform

from preprocessing import (
    DATA_PATH,
    INDEX_PATH,
    MODELS_DIR,
    PLOT_PATH,
    NUMERIC_COLUMNS,
    RAW_COLUMNS,
    prepare_features,
)

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Model selection optimises average precision (area under the precision/recall
# curve). Only about 26% of customers churn, so plain accuracy is dominated by
# the majority class -- a model that never predicts churn already scores 0.735.
SELECTION_METRIC = 'average_precision'

# Saved artifacts are compressed. Tree ensembles are the reason: an unpruned
# 800-tree forest pickles to ~160 MB, which has no business in a git
# repository. The search spaces below keep the ensembles pruned, which also
# happens to be better for generalisation on a dataset this size.
COMPRESS_LEVEL = 3


def build_preprocessor():
    """Impute, scale and one-hot encode the raw columns."""
    categorical = [c for c in RAW_COLUMNS if c not in NUMERIC_COLUMNS]
    return ColumnTransformer([
        ('num', Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
        ]), NUMERIC_COLUMNS),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='if_binary'), categorical),
    ])


def candidates():
    """The classical models to search over, with their parameter spaces.

    Keys are the slugs used for filenames; `name` is what the app displays.
    """
    return {
        'logistic_regression': {
            'name': 'Logistic regression',
            'family': 'Linear',
            'blurb': 'Linear decision boundary on the encoded features. Fast, and the '
                     'coefficients are readable as per-feature effects.',
            'estimator': LogisticRegression(max_iter=5000, class_weight='balanced',
                                            random_state=RANDOM_STATE),
            'space': {'clf__C': loguniform(1e-3, 1e3),
                      'clf__solver': ['liblinear', 'lbfgs']},
            'n_iter': 30,
        },
        'linear_svm': {
            'name': 'Linear SVM',
            'family': 'Support vector machine',
            'blurb': 'Maximum-margin linear separator (liblinear), wrapped in Platt '
                     'scaling so it reports probabilities.',
            'estimator': CalibratedClassifierCV(
                LinearSVC(class_weight='balanced', dual='auto', max_iter=5000,
                          random_state=RANDOM_STATE),
                method='sigmoid', cv=3),
            'space': {'clf__estimator__C': loguniform(1e-3, 1e1)},
            'n_iter': 8,
        },
        'rbf_svm': {
            'name': 'RBF SVM',
            'family': 'Support vector machine',
            'blurb': 'Support vector machine with a radial basis kernel, so it can fit '
                     'curved boundaries. Calibrated to report probabilities.',
            'estimator': CalibratedClassifierCV(
                SVC(kernel='rbf', class_weight='balanced', cache_size=500,
                    random_state=RANDOM_STATE),
                method='sigmoid', cv=3),
            # gamma above ~1e-1 makes the kernel so peaked it memorises the
            # training set; 'scale' would pick roughly 0.025 for these features.
            'space': {'clf__estimator__C': loguniform(1e-2, 3e1),
                      'clf__estimator__gamma': loguniform(1e-4, 1e-1)},
            'n_iter': 12,
        },
        'random_forest': {
            'name': 'Random forest',
            'family': 'Bagged trees',
            'blurb': 'Many decorrelated decision trees averaged together. Robust, and '
                     'needs little tuning.',
            'estimator': RandomForestClassifier(class_weight='balanced',
                                                random_state=RANDOM_STATE),
            # Depth and leaf size are bounded to keep the pickled forest small
            # enough to live in a git repository. Loosening them buys about
            # 0.002 average precision and costs several megabytes.
            'space': {'clf__n_estimators': randint(150, 300),
                      'clf__max_depth': randint(3, 12),
                      'clf__min_samples_leaf': randint(10, 40),
                      'clf__max_features': ['sqrt', 'log2']},
            'n_iter': 20,
        },
        'extra_trees': {
            'name': 'Extra trees',
            'family': 'Bagged trees',
            'blurb': 'Like a random forest, but split points are chosen at random, '
                     'trading a little bias for lower variance.',
            'estimator': ExtraTreesClassifier(class_weight='balanced',
                                              random_state=RANDOM_STATE),
            # Depth and leaf size are bounded to keep the pickled forest small
            # enough to live in a git repository. Loosening them buys about
            # 0.002 average precision and costs several megabytes.
            'space': {'clf__n_estimators': randint(150, 300),
                      'clf__max_depth': randint(3, 12),
                      'clf__min_samples_leaf': randint(10, 40),
                      'clf__max_features': ['sqrt', 'log2']},
            'n_iter': 20,
        },
        'gradient_boosting': {
            'name': 'Gradient boosting',
            'family': 'Boosted trees',
            'blurb': 'Shallow trees fitted one after another, each correcting the '
                     'errors of the ones before it.',
            'estimator': GradientBoostingClassifier(random_state=RANDOM_STATE),
            'space': {'clf__n_estimators': randint(100, 400),
                      'clf__learning_rate': loguniform(1e-2, 3e-1),
                      'clf__max_depth': randint(2, 5),
                      'clf__subsample': uniform(0.6, 0.4)},
            'n_iter': 20,
        },
        'hist_gradient_boosting': {
            'name': 'Histogram gradient boosting',
            'family': 'Boosted trees',
            'blurb': 'Gradient boosting on binned features. Much faster to fit, and '
                     'usually scores about the same.',
            'estimator': HistGradientBoostingClassifier(class_weight='balanced',
                                                        random_state=RANDOM_STATE),
            'space': {'clf__learning_rate': loguniform(1e-2, 3e-1),
                      'clf__max_leaf_nodes': randint(5, 40),
                      'clf__min_samples_leaf': randint(10, 100),
                      'clf__l2_regularization': loguniform(1e-3, 1e1)},
            'n_iter': 25,
        },
        'knn': {
            'name': 'K-nearest neighbours',
            'family': 'Instance based',
            'blurb': 'Classifies a customer by the labels of the most similar customers '
                     'in the training set. No model is really fitted.',
            'estimator': KNeighborsClassifier(),
            'space': {'clf__n_neighbors': randint(5, 80),
                      'clf__weights': ['uniform', 'distance'],
                      'clf__p': [1, 2]},
            'n_iter': 15,
        },
        'naive_bayes': {
            'name': 'Naive Bayes',
            'family': 'Probabilistic',
            'blurb': 'Assumes every feature is independent given the outcome. That is '
                     'plainly false here, which is why it trails the rest.',
            'estimator': GaussianNB(),
            'space': {'clf__var_smoothing': loguniform(1e-11, 1e-3)},
            'n_iter': 10,
        },
    }


def load_data():
    raw = pd.read_csv(DATA_PATH)
    y = raw['Churn'].eq('Yes').astype(int)
    X = prepare_features(raw, 'Batch')
    return X, y


def metrics_for(model, X_test, y_test):
    predicted = model.predict(X_test)
    scores = model.predict_proba(X_test)[:, 1]
    return {
        'accuracy': accuracy_score(y_test, predicted),
        'balanced_accuracy': balanced_accuracy_score(y_test, predicted),
        'precision': precision_score(y_test, predicted, zero_division=0),
        'recall': recall_score(y_test, predicted, zero_division=0),
        'f1': f1_score(y_test, predicted, zero_division=0),
        'roc_auc': roc_auc_score(y_test, scores),
        'pr_auc': average_precision_score(y_test, scores),
    }


def jsonable(value):
    """numpy scalars are not JSON serialisable; unwrap them."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    return value


def write_comparison_plot(entries, y_test, curves, path):
    """ROC and precision/recall curves for every model, side by side."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve, roc_curve
    except ImportError:
        print('  matplotlib not installed, skipping the comparison plot')
        print('  (pip install -r requirements-notebook.txt)')
        return False

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 5.5))
    for entry in entries:
        scores = curves[entry['slug']]
        fpr, tpr, _ = roc_curve(y_test, scores)
        ax_roc.plot(fpr, tpr, lw=1.6,
                    label=f"{entry['name']} ({entry['test']['roc_auc']:.3f})")
        precision, recall, _ = precision_recall_curve(y_test, scores)
        ax_pr.plot(recall, precision, lw=1.6,
                   label=f"{entry['name']} ({entry['test']['pr_auc']:.3f})")

    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Chance')
    ax_roc.set(xlabel='False positive rate', ylabel='True positive rate',
               title='ROC curves (AUC in legend)')
    ax_roc.legend(fontsize=8, loc='lower right')
    ax_roc.grid(alpha=0.3)

    base_rate = float(np.mean(y_test))
    ax_pr.axhline(base_rate, color='k', ls='--', lw=1, alpha=0.5,
                  label=f'Chance ({base_rate:.3f})')
    ax_pr.set(xlabel='Recall', ylabel='Precision',
              title='Precision/recall curves (avg. precision in legend)')
    ax_pr.legend(fontsize=8, loc='upper right')
    ax_pr.grid(alpha=0.3)

    fig.suptitle('Classical models on the held-out churn test set', fontsize=13)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f'  wrote {path}')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true',
                        help='Run a reduced search (useful for a smoke test).')
    parser.add_argument('--no-save', action='store_true',
                        help='Report results without writing the model gallery.')
    parser.add_argument('--only', metavar='SLUG', nargs='+',
                        help='Train only these models (by slug).')
    args = parser.parse_args()

    warnings.filterwarnings('ignore')
    X, y = load_data()
    print(f'Loaded {len(X)} rows, churn rate {y.mean():.3f}')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    print(f'Train {len(X_train)} rows / test {len(X_test)} rows (stratified)\n')

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    chosen = candidates()
    if args.only:
        unknown = [s for s in args.only if s not in chosen]
        if unknown:
            parser.error(f'unknown slug(s): {unknown}. Known: {list(chosen)}')
        chosen = {k: v for k, v in chosen.items() if k in args.only}

    entries, curves, fitted = [], {}, {}
    print(f'Searching {len(chosen)} models, selecting on {SELECTION_METRIC}, 5-fold CV')
    print('=' * 78)
    for slug, spec in chosen.items():
        started = time.time()
        pipeline = Pipeline([('prep', build_preprocessor()), ('clf', spec['estimator'])])
        search = RandomizedSearchCV(
            pipeline, spec['space'], n_iter=3 if args.quick else spec['n_iter'],
            scoring=SELECTION_METRIC, cv=cv, random_state=RANDOM_STATE, n_jobs=-1)
        search.fit(X_train, y_train)

        best_pipeline = search.best_estimator_

        # The default 0.5 cut-off is a poor operating point on imbalanced data.
        # Every model gets its own threshold, tuned by cross-validation on the
        # training set, so the comparison is between models at their best
        # operating point rather than at an arbitrary shared one.
        tuned = TunedThresholdClassifierCV(
            estimator=best_pipeline, scoring='f1', cv=cv, refit=True,
            random_state=RANDOM_STATE)
        tuned.fit(X_train, y_train)

        test_metrics = metrics_for(tuned, X_test, y_test)
        if len(np.unique(tuned.predict(X_test))) == 1:
            print(f"  !! {spec['name']} predicts a single class on the test set; "
                  f'its hyperparameters are degenerate.')
        curves[slug] = tuned.predict_proba(X_test)[:, 1]
        fitted[slug] = tuned
        entries.append({
            'slug': slug,
            'name': spec['name'],
            'family': spec['family'],
            'blurb': spec['blurb'],
            'file': f'{slug}.joblib',
            'cv_score': float(search.best_score_),
            'threshold': float(tuned.best_threshold_),
            'params': jsonable({k.removeprefix('clf__'): v
                                for k, v in search.best_params_.items()}),
            'test': jsonable(test_metrics),
            'fit_seconds': round(time.time() - started, 1),
        })
        print(f"  {spec['name']:<28} cv {search.best_score_:.4f}  "
              f"thr {tuned.best_threshold_:.3f}  "
              f"f1 {test_metrics['f1']:.3f}  ({time.time() - started:.0f}s)")

    entries.sort(key=lambda e: e['cv_score'], reverse=True)
    best = entries[0]
    print('=' * 78)

    header = (f'\n{"model":<28}{"acc":>7}{"bal-acc":>9}{"prec":>7}'
              f'{"recall":>8}{"f1":>7}{"ROC":>7}{"PR":>7}{"thr":>7}')
    print('Held-out test set, each model at its own tuned threshold')
    print(header)
    print('-' * len(header.strip()))
    for entry in entries:
        m = entry['test']
        print(f"{entry['name']:<28}{m['accuracy']:>7.3f}{m['balanced_accuracy']:>9.3f}"
              f"{m['precision']:>7.3f}{m['recall']:>8.3f}{m['f1']:>7.3f}"
              f"{m['roc_auc']:>7.3f}{m['pr_auc']:>7.3f}{entry['threshold']:>7.3f}")

    print(f"\nBest by cross-validated {SELECTION_METRIC}: {best['name']}")
    for key, value in best['params'].items():
        print(f'    {key} = {value}')

    best_predictions = fitted[best['slug']].predict(X_test)
    print('\nConfusion matrix for the best model (rows: actual, cols: predicted)')
    print(confusion_matrix(y_test, best_predictions))
    print()
    print(classification_report(y_test, best_predictions,
                                target_names=['stays', 'churns'], digits=3))

    if args.no_save:
        print('--no-save given, nothing written.')
        return

    print(f'Writing the gallery to {MODELS_DIR.name}/')
    if MODELS_DIR.exists():
        shutil.rmtree(MODELS_DIR)
    MODELS_DIR.mkdir(parents=True)
    for entry in entries:
        destination = MODELS_DIR / entry['file']
        joblib.dump(fitted[entry['slug']], destination, compress=COMPRESS_LEVEL)
        entry['size_kb'] = round(destination.stat().st_size / 1024, 1)
        print(f"  {entry['file']:<34} {entry['size_kb']:>8.1f} KB")

    index = {
        'generated': date.today().isoformat(),
        'dataset': DATA_PATH.name,
        'rows': int(len(X)),
        'test_size': TEST_SIZE,
        'random_state': RANDOM_STATE,
        'selection_metric': SELECTION_METRIC,
        'best': best['slug'],
        'models': entries,
    }
    INDEX_PATH.write_text(json.dumps(index, indent=2) + '\n')
    total_kb = sum(e['size_kb'] for e in entries)
    print(f"  {'index.json':<34} {INDEX_PATH.stat().st_size / 1024:>8.1f} KB")
    print(f'  gallery total: {total_kb / 1024:.1f} MB')

    print('\nComparison plot')
    write_comparison_plot(entries, y_test, curves, PLOT_PATH)

    print(f"\nDone. The app defaults to {best['name']}; "
          f'all {len(entries)} models are selectable in the sidebar.')


if __name__ == '__main__':
    main()

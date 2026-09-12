"""Train the churn model and write notebook/model.sav.

Run from the repository root:

    python train.py

The saved artifact is a complete scikit-learn pipeline: it takes the raw
feature columns (see preprocessing.RAW_COLUMNS) and handles imputation,
scaling and one-hot encoding internally. Keeping preprocessing inside the
pipeline is what stops the training and serving paths from drifting apart.
"""

import argparse
import warnings

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
from sklearn.svm import SVC
from scipy.stats import loguniform, randint, uniform

from preprocessing import DATA_PATH, MODEL_PATH, NUMERIC_COLUMNS, RAW_COLUMNS, prepare_features

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Model selection optimises average precision (area under the precision/recall
# curve). Only about 26% of customers churn, so plain accuracy is dominated by
# the majority class -- a model that never predicts churn already scores 0.735.
SELECTION_METRIC = 'average_precision'


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
    """The classical models to search over, with their parameter spaces."""
    return {
        'Logistic regression': (
            LogisticRegression(max_iter=5000, class_weight='balanced', random_state=RANDOM_STATE),
            {'clf__C': loguniform(1e-3, 1e3),
             'clf__solver': ['liblinear', 'lbfgs']},
            30,
        ),
        'Linear SVM': (
            SVC(kernel='linear', class_weight='balanced', random_state=RANDOM_STATE),
            {'clf__C': loguniform(1e-3, 1e2)},
            6,
        ),
        'RBF SVM': (
            SVC(kernel='rbf', class_weight='balanced', random_state=RANDOM_STATE),
            {'clf__C': loguniform(1e-2, 1e2),
             'clf__gamma': loguniform(1e-4, 1e0)},
            10,
        ),
        'Random forest': (
            RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1),
            {'clf__n_estimators': randint(200, 800),
             'clf__max_depth': randint(3, 20),
             'clf__min_samples_leaf': randint(1, 40),
             'clf__max_features': ['sqrt', 'log2', None]},
            20,
        ),
        'Extra trees': (
            ExtraTreesClassifier(class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1),
            {'clf__n_estimators': randint(200, 800),
             'clf__max_depth': randint(3, 20),
             'clf__min_samples_leaf': randint(1, 40),
             'clf__max_features': ['sqrt', 'log2', None]},
            20,
        ),
        'Gradient boosting': (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {'clf__n_estimators': randint(100, 500),
             'clf__learning_rate': loguniform(1e-2, 3e-1),
             'clf__max_depth': randint(2, 6),
             'clf__subsample': uniform(0.6, 0.4)},
            20,
        ),
        'Hist gradient boosting': (
            HistGradientBoostingClassifier(class_weight='balanced', random_state=RANDOM_STATE),
            {'clf__learning_rate': loguniform(1e-2, 3e-1),
             'clf__max_leaf_nodes': randint(5, 40),
             'clf__min_samples_leaf': randint(10, 100),
             'clf__l2_regularization': loguniform(1e-3, 1e1)},
            30,
        ),
        'K-nearest neighbours': (
            KNeighborsClassifier(n_jobs=-1),
            {'clf__n_neighbors': randint(5, 80),
             'clf__weights': ['uniform', 'distance'],
             'clf__p': [1, 2]},
            15,
        ),
        'Naive Bayes': (
            GaussianNB(),
            {'clf__var_smoothing': loguniform(1e-11, 1e-3)},
            10,
        ),
    }


def load_data():
    raw = pd.read_csv(DATA_PATH)
    y = raw['Churn'].eq('Yes').astype(int)
    X = prepare_features(raw, 'Batch')
    return X, y


def decision_scores(model, X):
    """Continuous scores for ranking metrics, however the model exposes them."""
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def evaluate(name, model, X_test, y_test):
    predicted = model.predict(X_test)
    scores = decision_scores(model, X_test)
    return {
        'model': name,
        'accuracy': accuracy_score(y_test, predicted),
        'balanced_acc': balanced_accuracy_score(y_test, predicted),
        'precision': precision_score(y_test, predicted, zero_division=0),
        'recall': recall_score(y_test, predicted, zero_division=0),
        'f1': f1_score(y_test, predicted, zero_division=0),
        'roc_auc': roc_auc_score(y_test, scores),
        'pr_auc': average_precision_score(y_test, scores),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true',
                        help='Run a reduced search (useful for a smoke test).')
    parser.add_argument('--no-save', action='store_true',
                        help='Report results without overwriting the saved model.')
    args = parser.parse_args()

    warnings.filterwarnings('ignore')
    X, y = load_data()
    print(f'Loaded {len(X)} rows, churn rate {y.mean():.3f}\n')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    print(f'Train {len(X_train)} rows / test {len(X_test)} rows (stratified)\n')

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results, fitted = [], {}

    print(f'Model selection by {SELECTION_METRIC}, 5-fold CV on the training set')
    print('-' * 72)
    for name, (estimator, space, n_iter) in candidates().items():
        pipeline = Pipeline([('prep', build_preprocessor()), ('clf', estimator)])
        search = RandomizedSearchCV(
            pipeline, space, n_iter=3 if args.quick else n_iter, scoring=SELECTION_METRIC,
            cv=cv, random_state=RANDOM_STATE, n_jobs=-1, refit=True)
        search.fit(X_train, y_train)
        fitted[name] = search.best_estimator_
        results.append({'model': name, 'cv_score': search.best_score_,
                        'params': search.best_params_})
        print(f'  {name:<24} {SELECTION_METRIC} {search.best_score_:.4f}')

    results.sort(key=lambda r: r['cv_score'], reverse=True)
    best = results[0]
    print('-' * 72)
    print(f'\nBest by CV: {best["model"]} ({SELECTION_METRIC} {best["cv_score"]:.4f})')
    for key, value in best['params'].items():
        print(f'    {key} = {value}')

    best_estimator = fitted[best['model']]

    # SVMs were searched without probability estimates because calibrating them
    # on every fit is slow. If one wins, refit it so the saved artifact can
    # still report a churn probability.
    final_clf = best_estimator.named_steps['clf']
    if isinstance(final_clf, SVC) and not final_clf.probability:
        print('\nRefitting the winning SVM with probability estimates enabled')
        best_estimator.set_params(clf__probability=True)
        best_estimator.fit(X_train, y_train)

    # The default 0.5 cut-off is a poor operating point on imbalanced data, so
    # pick the threshold that maximises F1, using cross-validation on the
    # training set only.
    print('\nTuning the decision threshold for F1 (cross-validated on train)')
    tuned = TunedThresholdClassifierCV(
        estimator=best_estimator, scoring='f1', cv=cv, refit=True,
        random_state=RANDOM_STATE)
    tuned.fit(X_train, y_train)
    print(f'  chosen threshold: {tuned.best_threshold_:.4f}')

    print('\nHeld-out test set results')
    print('-' * 72)
    table = [evaluate(f'{best["model"]} (0.5 cut-off)', best_estimator, X_test, y_test),
             evaluate(f'{best["model"]} (tuned threshold)', tuned, X_test, y_test)]
    header = f'{"model":<40}{"acc":>7}{"bal-acc":>9}{"prec":>7}{"recall":>8}{"f1":>7}{"ROC":>7}{"PR":>7}'
    print(header)
    for row in table:
        print(f'{row["model"]:<40}{row["accuracy"]:>7.3f}{row["balanced_acc"]:>9.3f}'
              f'{row["precision"]:>7.3f}{row["recall"]:>8.3f}{row["f1"]:>7.3f}'
              f'{row["roc_auc"]:>7.3f}{row["pr_auc"]:>7.3f}')

    print('\nConfusion matrix, tuned threshold (rows: actual, cols: predicted)')
    print(confusion_matrix(y_test, tuned.predict(X_test)))
    print()
    print(classification_report(y_test, tuned.predict(X_test),
                                target_names=['stays', 'churns'], digits=3))

    if args.no_save:
        print('--no-save given, model not written.')
        return

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(tuned, MODEL_PATH)
    print(f'Saved {best["model"]} with tuned threshold to {MODEL_PATH}')


if __name__ == '__main__':
    main()

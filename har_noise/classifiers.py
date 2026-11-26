"""Classifiers and hyperparameter selection.

Hyperparameters are chosen by 5-fold cross-validation on the TRAINING set
only. Folds are split by subject (GroupKFold), for the same reason we keep the
official split: a subject must never be in both the fitting and the checking
part. The projection is re-fitted inside every fold, so the validation fold
never leaks into the projection (this matters most for LDA, which uses labels).
"""

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .metrics import classification_scores
from .projection import make_projection

GRIDS = {
    "knn": [{"n_neighbors": k} for k in (1, 3, 5, 11)],
    "svm": [{"C": c} for c in (0.01, 0.1, 1.0)],
    "gnb": [{"var_smoothing": v} for v in (1e-9, 1e-5, 1e-2)],
}


def make_classifier(name, params):
    if name == "knn":
        return KNeighborsClassifier(**params)
    if name == "svm":
        # scaling only inside the SVM pipeline; fitted on clean training features
        return make_pipeline(
            StandardScaler(), LinearSVC(dual="auto", max_iter=10000, random_state=0, **params)
        )
    if name == "gnb":
        return GaussianNB(**params)
    raise ValueError(name)


def fit_quiet(model, X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return model.fit(X, y)


def tune(proj_name, X, y, subjects, n_splits=5):
    """Return (best params per classifier, list of all CV scores)."""
    cv_rows = []
    for fold, (tr, va) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, subjects)):
        proj = make_projection(proj_name).fit(X[tr], y[tr])
        Ztr, Zva = proj.transform(X[tr]), proj.transform(X[va])
        for clf_name, grid in GRIDS.items():
            for params in grid:
                model = fit_quiet(make_classifier(clf_name, params), Ztr, y[tr])
                s = classification_scores(y[va], model.predict(Zva))
                cv_rows.append(
                    {
                        "projection": proj_name,
                        "classifier": clf_name,
                        "params": str(params),
                        "fold": fold,
                        **s,
                    }
                )
    best = {}
    for clf_name, grid in GRIDS.items():
        means = [
            np.mean([r["macro_f1"] for r in cv_rows if r["classifier"] == clf_name and r["params"] == str(p)])
            for p in grid
        ]
        best[clf_name] = grid[int(np.argmax(means))]  # first best wins ties
    return best, cv_rows


def fit_final(proj_name, X, y, best):
    """Refit the projection on the whole training set, then each tuned classifier."""
    proj = make_projection(proj_name).fit(X, y)
    Z = proj.transform(X)
    models = {c: fit_quiet(make_classifier(c, p), Z, y) for c, p in best.items()}
    return proj, models
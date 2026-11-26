"""Three corruption functions. Applied to TEST data only; the trained
pipeline never sees corrupted data. Every function returns a new array and
never changes its input. `level` is in [0, 0.5]."""

import numpy as np


def train_stats(X_train, groups):
    """Statistics the corruptions need, taken from the clean training set."""
    return {"mean": X_train.mean(0), "std": X_train.std(0), "groups": groups}


def gaussian_noise(X, level, rng, stats):
    """Add zero-mean noise with std = level * (training std of that feature)."""
    if level == 0:
        return X.copy()
    return X + rng.normal(0.0, 1.0, X.shape) * (level * stats["std"])


def n_groups_to_drop(level, n_groups):
    """Fraction of the signal groups to drop, rounded half up (0.5*17 -> 9)."""
    return int(np.floor(level * n_groups + 0.5))


def group_dropout(X, level, rng, stats):
    """Pick a random set of signal groups (e.g. all 79 fBodyGyro features) and
    set them to 0, the value a missing sensor channel usually gets filled with.
    The same groups are dropped for every test sample (a dead channel)."""
    X = X.copy()
    names = list(stats["groups"])
    k = n_groups_to_drop(level, len(names))
    if k == 0:
        return X
    for g in rng.choice(names, size=k, replace=False):
        X[:, stats["groups"][g]] = 0.0
    return X


def ood_injection(X, level, rng, stats):
    """Replace a fraction of test rows with out-of-distribution vectors.
    Each feature is drawn independently around the training mean with twice
    the training spread, then clipped to the valid range [-1, 1]. These rows
    look like sensor readings but carry no activity pattern. Labels are kept,
    so the best possible accuracy is about (1 - level) + level / 6."""
    X = X.copy()
    n = round(level * len(X))
    if n == 0:
        return X
    rows = rng.choice(len(X), size=n, replace=False)
    fake = stats["mean"] + 2.0 * stats["std"] * rng.normal(0.0, 1.0, (n, X.shape[1]))
    X[rows] = np.clip(fake, -1.0, 1.0)
    return X


CORRUPTIONS = {"gaussian": gaussian_noise, "dropout": group_dropout, "ood": ood_injection}
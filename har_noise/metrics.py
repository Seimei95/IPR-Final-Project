"""Metrics: Fisher's criterion, classification scores, robustness summaries."""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def fisher_criterion(Z, y):
    """Scalar Fisher criterion J = trace(Sb) / trace(Sw) in the space Z.
    Higher = classes further apart relative to their own spread."""
    mu = Z.mean(0)
    sw = sb = 0.0
    for c in np.unique(y):
        Zc = Z[y == c]
        mc = Zc.mean(0)
        sw += ((Zc - mc) ** 2).sum()
        sb += len(Zc) * ((mc - mu) ** 2).sum()
    return sb / sw


def classification_scores(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"acc": accuracy_score(y_true, y_pred), "macro_p": p, "macro_r": r, "macro_f1": f}


def confusion(y_true, y_pred, n_classes=6):
    return confusion_matrix(y_true, y_pred, labels=range(n_classes))


def degradation_slope(levels, accs):
    """Accuracy points lost per +10% corruption (least-squares line). Positive = gets worse."""
    slope = np.polyfit(np.asarray(levels, float), np.asarray(accs, float), 1)[0]
    return -slope * 0.1 * 100


def breakdown_point(levels, accs, threshold=0.70):
    """First corruption level where accuracy drops below the threshold, else None (survives 50%)."""
    for lvl, a in zip(levels, accs):
        if a < threshold:
            return lvl
    return None


def intervals_overlap(m1, s1, m2, s2):
    """True if mean +/- std intervals overlap (then we do not claim a difference)."""
    return not (m1 - s1 > m2 + s2 or m2 - s2 > m1 + s1)
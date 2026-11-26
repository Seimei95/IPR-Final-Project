"""Load the UCI HAR dataset from the official train/ and test/ folders.

We read the official archive directly (not ucimlrepo) so the subject-wise
split made by the dataset authors is kept: 21 subjects train, 9 subjects test.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ACTIVITIES = ["WALKING", "UPSTAIRS", "DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]
DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "data" / "UCI HAR Dataset"


@dataclass
class HARData:
    X_train: np.ndarray  # (7352, 561)
    y_train: np.ndarray  # labels 0..5
    s_train: np.ndarray  # subject id per row
    X_test: np.ndarray  # (2947, 561)
    y_test: np.ndarray
    s_test: np.ndarray
    feature_names: list  # 561 unique names
    groups: dict  # signal group name -> array of column indices


def dedupe_names(names):
    """features.txt has 42 repeated names (bandsEnergy blocks). Add a suffix
    to every repeat so each column label is unique: name, name__2, name__3."""
    seen, out = {}, []
    for n in names:
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n}__{seen[n]}")
    return out


def signal_groups(names):
    """Group columns by signal family, e.g. 'tBodyAcc-mean()-X' -> 'tBodyAcc'.
    This gives the 17 time/frequency groups. The 7 'angle(...)' features are
    not part of any group, so dropout never removes them."""
    groups = {}
    for i, n in enumerate(names):
        if n.startswith("angle("):
            continue
        groups.setdefault(n.split("-")[0], []).append(i)
    return {g: np.array(idx) for g, idx in sorted(groups.items())}


def load_har(root=DEFAULT_ROOT):
    root = Path(root)
    with open(root / "features.txt") as f:
        raw = [line.split(" ", 1)[1].strip() for line in f]
    names = dedupe_names(raw)

    def part(split):
        X = np.loadtxt(root / split / f"X_{split}.txt")
        y = np.loadtxt(root / split / f"y_{split}.txt", dtype=int) - 1  # 1..6 -> 0..5
        s = np.loadtxt(root / split / f"subject_{split}.txt", dtype=int)
        return X, y, s

    Xtr, ytr, st = part("train")
    Xte, yte, se = part("test")
    return HARData(Xtr, ytr, st, Xte, yte, se, names, signal_groups(raw))
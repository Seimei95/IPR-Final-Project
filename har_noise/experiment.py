"""The main sweep: corruption type x level x seed x projection x classifier."""

import numpy as np
import pandas as pd

from .corruption import CORRUPTIONS, train_stats
from .metrics import classification_scores, confusion, fisher_criterion

LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
SEEDS = {"gaussian": 5, "dropout": 10, "ood": 5}
CORR_ID = {"gaussian": 1, "dropout": 2, "ood": 3}


def corruption_rng(corruption, level, seed):
    """Own random stream per (corruption, level, seed): results do not depend
    on run order, and seed s at 30% is reproducible on its own."""
    return np.random.default_rng([seed, CORR_ID[corruption], round(level * 100)])


def run_sweep(data, fitted, levels=LEVELS, seeds=SEEDS, cm_levels=(0.0, 0.3, 0.5)):
    """fitted: {proj_name: (projection, {clf_name: model})}.
    Returns (results DataFrame, dict of confusion matrices for seed 0)."""
    stats = train_stats(data.X_train, data.groups)
    rows, cms = [], {}
    clean_cache, clean_from = None, None
    for corr, n_seeds in seeds.items():
        fn = CORRUPTIONS[corr]
        for level in levels:
            for seed in range(n_seeds):
                if level == 0 and clean_cache is not None:  # 0% is identical for every seed/type
                    rows += [{**r, "corruption": corr, "seed": seed} for r in clean_cache]
                    for key in [k for k in cms if k[0] == clean_from and k[1] == 0.0]:
                        cms[(corr,) + key[1:]] = cms[key]
                    continue
                Xc = fn(data.X_test, level, corruption_rng(corr, level, seed), stats)
                cell = []
                for pname, (proj, models) in fitted.items():
                    Z = proj.transform(Xc)
                    J = fisher_criterion(Z, data.y_test)
                    for cname, model in models.items():
                        pred = model.predict(Z)
                        cell.append(
                            {
                                "projection": pname,
                                "classifier": cname,
                                "level": level,
                                "fisher": J,
                                **classification_scores(data.y_test, pred),
                            }
                        )
                        if seed == 0 and level in cm_levels:
                            cms[(corr, level, pname, cname)] = confusion(data.y_test, pred)
                if level == 0:
                    clean_cache, clean_from = cell, corr
                rows += [{**r, "corruption": corr, "seed": seed} for r in cell]
    cols = [
        "corruption",
        "level",
        "seed",
        "projection",
        "classifier",
        "acc",
        "macro_p",
        "macro_r",
        "macro_f1",
        "fisher",
    ]
    return pd.DataFrame(rows)[cols], cms
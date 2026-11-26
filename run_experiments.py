"""Single entry point: reproduces every number and figure in the report.

    uv run python run_experiments.py           (full run)
    uv run python run_experiments.py --quick   (tiny smoke run, about 1 minute)

Outputs go to results/ (CSV/JSON tables) and figures/ (PNG)."""

import argparse
import json
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from har_noise import plots
from har_noise.classifiers import fit_final, fit_quiet, make_classifier, tune
from har_noise.corruption import gaussian_noise, train_stats
from har_noise.data import load_har
from har_noise.experiment import LEVELS, SEEDS, corruption_rng, run_sweep
from har_noise.metrics import breakdown_point, classification_scores, degradation_slope, fisher_criterion
from har_noise.projection import LDA, PCA

PROJECTIONS = [
    "none",
    "pca10",
    "pca20",
    "pca50",
    "pca100",
    "ica10",
    "ica20",
    "ica50",
    "ica100",
    "lda5",
    "pca100-lda5",
]
SHRINKAGES = [1e-6, 1e-4, 1e-2, 1e-1, 0.5]
OUT, FIG = Path("results"), Path("figures")


def leakage_check(d):
    """Official subject-wise split vs random stratified splits (same test size)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        off = LinearSVC(dual="auto", max_iter=10000, random_state=0).fit(d.X_train, d.y_train)
        official = classification_scores(d.y_test, off.predict(d.X_test))["acc"]
        X = np.vstack([d.X_train, d.X_test])
        y = np.concatenate([d.y_train, d.y_test])
        rand = []
        for s in range(5):
            a, b, c, e = train_test_split(X, y, test_size=len(d.y_test), random_state=s, stratify=y)
            m = LinearSVC(dual="auto", max_iter=10000, random_state=0).fit(a, c)
            rand.append(classification_scores(e, m.predict(b))["acc"])
    return {
        "official_split_acc": official,
        "random_split_acc_mean": float(np.mean(rand)),
        "random_split_acc_std": float(np.std(rand)),
    }


def shrinkage_sweep(d, C, levels, n_seeds):
    stats = train_stats(d.X_train, d.groups)
    rows = []
    for lam in SHRINKAGES:
        lda = LDA(5, shrinkage=lam).fit(d.X_train, d.y_train)
        model = fit_quiet(make_classifier("svm", {"C": C}), lda.transform(d.X_train), d.y_train)
        for lvl in levels:
            for seed in range(n_seeds):
                Z = lda.transform(gaussian_noise(d.X_test, lvl, corruption_rng("gaussian", lvl, seed), stats))
                rows.append(
                    {
                        "shrinkage": lam,
                        "level": lvl,
                        "seed": seed,
                        "fisher": fisher_criterion(Z, d.y_test),
                        **classification_scores(d.y_test, model.predict(Z)),
                    }
                )
    return pd.DataFrame(rows)


def summarise(res):
    rows = []
    for (corr, proj, clf), g in res.groupby(["corruption", "projection", "classifier"]):
        st = g.groupby("level")["acc"].agg(["mean", "std"])
        f1 = g.groupby("level")["macro_f1"].mean()
        fi = g.groupby("level")["fisher"].mean()
        bp = breakdown_point(st.index, st["mean"])
        rows.append(
            {
                "corruption": corr,
                "projection": proj,
                "classifier": clf,
                "clean_acc": st["mean"].iloc[0],
                "acc_30": st["mean"].loc[0.3],
                "acc_50": st["mean"].iloc[-1],
                "acc_50_std": st["std"].iloc[-1],
                "f1_clean": f1.iloc[0],
                "f1_50": f1.iloc[-1],
                "slope_per10": degradation_slope(st.index, st["mean"]),
                "breakdown": "none (>50%)" if bp is None else f"{round(bp * 100)}%",
                "fisher_clean": fi.iloc[0],
                "fisher_ratio_50": fi.iloc[-1] / fi.iloc[0],
            }
        )
    return pd.DataFrame(rows)


def main(quick=False):
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    d = load_har()
    projections = ["none", "pca50", "ica50", "lda5", "pca100-lda5"] if quick else PROJECTIONS
    levels = [0.0, 0.3, 0.5] if quick else LEVELS
    seeds = {"gaussian": 1, "dropout": 2, "ood": 1} if quick else SEEDS
    print(f"train {d.X_train.shape}, test {d.X_test.shape}, groups {len(d.groups)}")

    rank_sw = LDA().fit(d.X_train, d.y_train).rank_Sw_
    leak = leakage_check(d)
    print("leakage check:", {k: round(v, 4) for k, v in leak.items()})

    best_params, cv_rows, fitted = {}, [], {}
    for p in projections:
        t = time.time()
        best, rows = tune(p, d.X_train, d.y_train, d.s_train)
        best_params[p] = best
        cv_rows += rows
        fitted[p] = fit_final(p, d.X_train, d.y_train, best)
        print(f"tuned {p:12s} {best}  ({time.time() - t:.1f}s)")
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(OUT / "cv_results.csv", index=False)

    t = time.time()
    res, cms = run_sweep(d, fitted, levels=levels, seeds=seeds)
    print(f"sweep: {len(res)} rows ({time.time() - t:.1f}s)")
    res.to_csv(OUT / "results.csv", index=False)
    np.savez(OUT / "confusion_matrices.npz", **{"|".join(map(str, k)): v for k, v in cms.items()})

    summary = summarise(res)
    summary.to_csv(OUT / "summary.csv", index=False)

    comp = []
    for p in projections:
        if p[:3] in ("pca", "ica") and "-" not in p:
            for clf in ("svm", "knn"):
                g = res[(res.corruption == "gaussian") & (res.projection == p) & (res.classifier == clf)]
                for lvl, a in g.groupby("level")["acc"].mean().items():
                    comp.append({"method": p[:3], "n": int(p[3:]), "classifier": clf, "level": lvl, "acc": a})
    comp = pd.DataFrame(comp)
    comp.to_csv(OUT / "component_sweep.csv", index=False)

    shrink = shrinkage_sweep(d, best_params["lda5"]["svm"]["C"], levels, seeds["gaussian"])
    shrink.to_csv(OUT / "shrinkage_sweep.csv", index=False)

    # figures
    main_projs = [p for p in plots.STYLE if p in projections]
    plots.accuracy_grid(res, FIG / "fig1_accuracy_vs_corruption.png", projs=main_projs)
    plots.accuracy_grid(res, FIG / "figA_macro_f1_vs_corruption.png", metric="macro_f1", projs=main_projs)
    plots.fisher_plot(res, FIG / "fig2_fisher_vs_corruption.png", projs=main_projs)
    plots.confusion_grid(
        cms,
        [
            ("PCA(50)+SVM, clean", ("gaussian", 0.0, "pca50", "svm")),
            ("PCA(50)+SVM, 30% Gaussian", ("gaussian", 0.3, "pca50", "svm")),
            ("PCA(50)+SVM, 30% dropout", ("dropout", 0.3, "pca50", "svm")),
            ("LDA(5)+SVM, clean", ("gaussian", 0.0, "lda5", "svm")),
            ("LDA(5)+SVM, 30% Gaussian", ("gaussian", 0.3, "lda5", "svm")),
            ("LDA(5)+SVM, 30% dropout", ("dropout", 0.3, "lda5", "svm")),
        ],
        FIG / "fig3_confusion_matrices.png",
    )
    stats = train_stats(d.X_train, d.groups)
    noisy = gaussian_noise(d.X_test, 0.3, corruption_rng("gaussian", 0.3, 0), stats)
    pca2 = PCA(2).fit(d.X_train)
    lda2 = fitted["lda5"][0]
    plots.scatter_2d(
        [
            ("PCA, clean test", pca2.transform(d.X_test), "pca"),
            ("PCA, 30% Gaussian noise", pca2.transform(noisy), "pca"),
            ("LDA, clean test", lda2.transform(d.X_test), "lda"),
            ("LDA, 30% Gaussian noise", lda2.transform(noisy), "lda"),
        ],
        d.y_test,
        FIG / "fig4_projections_2d.png",
    )
    evr = PCA(561).fit(d.X_train).explained_variance_ratio_all_
    plots.hyperparam_plot(cv, comp, shrink, evr, FIG / "fig5_hyperparameters.png")

    info = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
        "machine": platform.platform(),
        "quick": quick,
        "rank_Sw": int(rank_sw),
        "ica_final_iterations": {p: int(fitted[p][0].n_iter_) for p in projections if p.startswith("ica")},
        "best_params": best_params,
        "leakage": leak,
        "pca_cum_var": {n: float(np.cumsum(evr)[n - 1]) for n in (2, 10, 20, 50, 100)},
        "runtime_s": round(time.time() - t0, 1),
    }
    (OUT / "run_info.json").write_text(json.dumps(info, indent=2))
    print(f"done in {info['runtime_s']} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
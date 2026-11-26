"""All report figures. Colours: fixed categorical order (validated for colour
blindness), neutral grey for the no-reduction baseline, plus distinct markers
so lines can be told apart without colour."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import ACTIVITIES

STYLE = {  # projection -> (label, colour, marker)
    "none": ("No reduction (561-D)", "#52514e", "o"),
    "pca50": ("PCA (50)", "#2a78d6", "s"),
    "ica50": ("ICA (50)", "#eb6834", "^"),
    "lda5": ("LDA (5, shrinkage)", "#1baf7a", "D"),
    "pca100-lda5": ("PCA(100) then LDA (5)", "#eda100", "v"),
}
CLF_LABEL = {"knn": "k-NN", "svm": "Linear SVM", "gnb": "Gaussian NB"}
CORR_LABEL = {"gaussian": "Gaussian noise", "dropout": "Signal-group dropout", "ood": "OOD injection"}
CLASS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
CLASS_MARKERS = ["o", "s", "^", "D", "v", "P"]

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#8a8984",
        "axes.labelcolor": "#0b0b0b",
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "axes.grid": True,
        "grid.color": "#e6e5e0",
        "grid.linewidth": 0.6,
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
        "savefig.dpi": 200,
    }
)


def _line(ax, x, m, s, proj):
    label, color, marker = STYLE[proj]
    ax.plot(x, m, color=color, marker=marker, markersize=4, linewidth=1.5, label=label)
    ax.fill_between(x, m - s, m + s, color=color, alpha=0.15, linewidth=0)


def accuracy_grid(df, path, metric="acc", projs=tuple(STYLE)):
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 6.4), sharex=True, sharey=True)
    for i, corr in enumerate(CORR_LABEL):
        for j, clf in enumerate(CLF_LABEL):
            ax = axes[i, j]
            for p in projs:
                g = df[(df.corruption == corr) & (df.classifier == clf) & (df.projection == p)]
                st = g.groupby("level")[metric].agg(["mean", "std"])
                _line(ax, st.index * 100, st["mean"].values, st["std"].values, p)
            if corr == "ood":  # best achievable when OOD rows count as wrong except 1/6 by chance
                lv = np.array(sorted(df.level.unique()))
                ax.plot(
                    lv * 100,
                    (1 - lv) + lv / 6,
                    color="#8a8984",
                    linestyle="--",
                    linewidth=1,
                    label="OOD ceiling",
                )
            ax.axhline(0.70, color="#c3c2b7", linewidth=0.8, linestyle=":")
            if i == 0:
                ax.set_title(CLF_LABEL[clf])
            if j == 0:
                ax.set_ylabel(f"{CORR_LABEL[corr]}\n{'Accuracy' if metric == 'acc' else 'Macro-F1'}")
            if i == 2:
                ax.set_xlabel("Corruption level (%)")
    axes[0, 0].set_ylim(0, 1.02)
    h, l = axes[2, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fisher_plot(df, path, projs=tuple(STYLE)):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
    d = df[df.classifier == "knn"]  # Fisher does not depend on the classifier
    for ax, corr in zip(axes, CORR_LABEL):
        for p in projs:
            g = d[(d.corruption == corr) & (d.projection == p)]
            st = g.groupby("level")["fisher"].agg(["mean", "std"])
            j0 = st["mean"].iloc[0]
            _line(ax, st.index * 100, st["mean"].values / j0, st["std"].values / j0, p)
        ax.set_yscale("log")
        ax.set_title(CORR_LABEL[corr])
        ax.set_xlabel("Corruption level (%)")
    axes[0].set_ylabel("Fisher J / J(clean)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def confusion_grid(cms, panels, path):
    """panels: list of (title, key) where key = (corruption, level, projection, classifier)."""
    n = len(panels)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.2, 2.5 * rows))
    axes = np.atleast_2d(axes)
    short = ["WALK", "UP", "DOWN", "SIT", "STAND", "LAY"]
    for ax, (title, key) in zip(axes.flat, panels):
        cm = cms[key]
        norm = cm / cm.sum(1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for a in range(6):
            for b in range(6):
                ax.text(
                    b,
                    a,
                    f"{cm[a, b]}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="#ffffff" if norm[a, b] > 0.6 else "#0b0b0b",
                )
        ax.set_xticks(range(6), short, rotation=45, fontsize=6.5)
        ax.set_yticks(range(6), short, fontsize=6.5)
        ax.grid(False)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Predicted", fontsize=7)
        ax.set_ylabel("True", fontsize=7)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def scatter_2d(panels, y, path):
    """panels: list of (title, Z, key). Panels with the same key share axis limits,
    so clean and noisy versions of one projection are drawn on the same scale."""
    fig, axes = plt.subplots(1, len(panels), figsize=(7.2, 2.4))
    lims = {}
    for _, Z, key in panels:
        lo, hi = Z[:, :2].min(0), Z[:, :2].max(0)
        if key in lims:
            lo, hi = np.minimum(lo, lims[key][0]), np.maximum(hi, lims[key][1])
        lims[key] = (lo, hi)
    for ax, (title, Z, key) in zip(axes, panels):
        lo, hi = lims[key]
        pad = 0.03 * (hi - lo)
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
        ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        rng = np.random.default_rng(0)
        idx = rng.choice(len(Z), size=min(1500, len(Z)), replace=False)
        for c in range(6):
            m = idx[y[idx] == c]
            ax.scatter(
                Z[m, 0],
                Z[m, 1],
                s=6,
                alpha=0.6,
                color=CLASS_COLORS[c],
                marker=CLASS_MARKERS[c],
                linewidths=0,
                label=ACTIVITIES[c].title(),
            )
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False, markerscale=2, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def hyperparam_plot(cv, comp, shrink, evr, path):
    """cv: CV rows; comp: component sweep table; shrink: shrinkage table; evr: PCA explained-variance ratios."""
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.3))
    ax = axes[0]
    k = cv[cv.classifier == "knn"].copy()
    k["k"] = k.params.str.extract(r"(\d+)").astype(int)
    for p in ["none", "pca50", "ica50", "lda5"]:
        st = k[k.projection == p].groupby("k")["macro_f1"].mean()
        label, color, marker = STYLE[p]
        ax.plot(st.index, st.values, color=color, marker=marker, markersize=4, linewidth=1.5, label=label)
    ax.set_xticks([1, 3, 5, 11])
    ax.set_xlabel("k (neighbours)")
    ax.set_ylabel("CV macro-F1")
    ax.set_title("(a) k-NN: k (5-fold CV)")
    ax = axes[1]
    for method, color, marker in [("pca", "#2a78d6", "s"), ("ica", "#eb6834", "^")]:
        for lvl, ls in [(0.0, "-"), (0.3, "--")]:
            g = comp[(comp.method == method) & (comp.level == lvl) & (comp.classifier == "knn")].sort_values(
                "n"
            )
            ax.plot(g.n, g.acc, color=color, marker=marker, markersize=4, linewidth=1.5, linestyle=ls)
        end = comp[
            (comp.method == method) & (comp.level == 0.0) & (comp.classifier == "knn") & (comp.n == 20)
        ]
        ax.annotate(
            method.upper(),
            (20, end.acc.iloc[0]),
            xytext=(0, 7 if method == "pca" else -12),
            textcoords="offset points",
            fontsize=7,
            color="#0b0b0b",
            ha="center",
        )
    ax.plot([], [], color="#8a8984", linestyle="-", label="0% noise")
    ax.plot([], [], color="#8a8984", linestyle="--", label="30% noise")
    ax.set_xscale("log")
    ax.set_xticks([10, 20, 50, 100], ["10", "20", "50", "100"])
    ax.set_xlabel("Components")
    ax.set_ylabel("Test accuracy (k-NN)")
    ax.set_title("(b) Components (k-NN)")
    ax.legend(frameon=False, fontsize=6, loc="lower left")
    ax = axes[2]
    for lvl, ls in [(0.0, "-"), (0.3, "--")]:
        g = shrink[shrink.level == lvl].groupby("shrinkage")["acc"].mean()
        ax.plot(
            g.index,
            g.values,
            color="#1baf7a",
            marker="D",
            markersize=4,
            linewidth=1.5,
            linestyle=ls,
            label=f"{int(lvl * 100)}% noise",
        )
    ax.set_xscale("log")
    ax.set_xlabel("LDA shrinkage")
    ax.set_ylabel("Test accuracy (SVM)")
    ax.set_title("(c) LDA shrinkage")
    ax.legend(frameon=False, fontsize=6, loc="lower right")
    ax = axes[3]
    cum = np.cumsum(evr)[:150]
    ax.plot(np.arange(1, len(cum) + 1), cum, color="#2a78d6", linewidth=1.5)
    for n in (10, 50, 100):
        ax.plot(n, cum[n - 1], "o", color="#2a78d6", markersize=4)
        ax.annotate(
            f"{cum[n - 1]:.0%}", (n, cum[n - 1]), xytext=(3, -10), textcoords="offset points", fontsize=6.5
        )
    ax.set_xlabel("PCA components")
    ax.set_ylabel("Cumulative variance")
    ax.set_title("(d) PCA variance")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
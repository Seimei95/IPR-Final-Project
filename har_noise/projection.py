"""Projection methods. PCA and LDA are written from scratch with NumPy/SciPy.
FastICA comes from scikit-learn. Every class has fit(X, y) / transform(X)."""

import warnings

import numpy as np
import scipy.linalg as sla
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning


class Identity:
    """No reduction: the 561-dimensional baseline."""

    name = "none"

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


class PCA:
    """Eigendecomposition of the covariance matrix; keep the top-k eigenvectors."""

    def __init__(self, n_components):
        self.k = n_components
        self.name = f"pca{n_components}"

    def fit(self, X, y=None):
        self.mean_ = X.mean(0)
        Z = X - self.mean_
        cov = Z.T @ Z / (len(X) - 1)
        vals, vecs = np.linalg.eigh(cov)  # ascending order
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]
        # fix the sign of each eigenvector so results do not flip between machines
        signs = np.sign(vecs[np.abs(vecs).argmax(0), np.arange(vecs.shape[1])])
        vecs = vecs * signs
        self.components_ = vecs[:, : self.k]  # (d, k)
        self.explained_variance_ = vals[: self.k]
        self.explained_variance_ratio_all_ = np.clip(vals, 0, None) / np.clip(vals, 0, None).sum()
        return self

    def transform(self, X):
        return (X - self.mean_) @ self.components_


class LDA:
    """Fisher LDA. Maximise between-class scatter Sb relative to within-class
    scatter Sw by solving the generalized eigenproblem  Sb w = lambda Sw w.

    On UCI HAR, Sw is singular (rank 470 of 561), so the textbook
    inv(Sw) @ Sb fails. We add shrinkage: Sw + shrink * (trace(Sw)/d) * I.
    At most C-1 = 5 directions carry information."""

    def __init__(self, n_components=5, shrinkage=1e-4):
        self.k = n_components
        self.shrinkage = shrinkage
        self.name = f"lda{n_components}"

    def fit(self, X, y):
        classes = np.unique(y)
        self.k = min(self.k, len(classes) - 1)
        d = X.shape[1]
        self.mean_ = X.mean(0)
        Sw = np.zeros((d, d))
        Sb = np.zeros((d, d))
        for c in classes:
            Xc = X[y == c]
            mc = Xc.mean(0)
            Zc = Xc - mc
            Sw += Zc.T @ Zc
            diff = (mc - self.mean_)[:, None]
            Sb += len(Xc) * diff @ diff.T
        Sw_reg = Sw + self.shrinkage * np.trace(Sw) / d * np.eye(d)
        vals, vecs = sla.eigh(Sb, Sw_reg)  # generalized, ascending
        order = np.argsort(vals)[::-1][: self.k]
        W = vecs[:, order]
        # eigh scales W so W.T @ Sw_reg @ W = I, where Sw is a SUM over all n rows,
        # so the projected values come out ~1/sqrt(n) small. Rescale so the
        # within-class variance of each LDA axis is about 1.
        W = W * np.sqrt(len(X) - len(classes))
        signs = np.sign(W[np.abs(W).argmax(0), np.arange(W.shape[1])])
        self.components_ = W * signs
        self.eigenvalues_ = vals[order]
        self.rank_Sw_ = np.linalg.matrix_rank(Sw)
        return self

    def transform(self, X):
        return (X - self.mean_) @ self.components_


class PCAthenLDA:
    """Alternative fix for singular Sw: reduce to 100 dims with PCA (where Sw is
    full rank), then run LDA there (the 'Fisherfaces' idea)."""

    def __init__(self, n_pca=100, n_components=5):
        self.pca = PCA(n_pca)
        self.lda = LDA(n_components, shrinkage=0.0)
        self.name = f"pca{n_pca}-lda{n_components}"

    def fit(self, X, y):
        Z = self.pca.fit(X).transform(X)
        self.lda.fit(Z, y)
        return self

    def transform(self, X):
        return self.lda.transform(self.pca.transform(X))


class ICA:
    """FastICA (scikit-learn), whitened, fixed random_state."""

    def __init__(self, n_components, seed=0):
        self.k = n_components
        self.name = f"ica{n_components}"
        self.model = FastICA(
            n_components=n_components, whiten="unit-variance", max_iter=1000, tol=1e-3, random_state=seed
        )

    def fit(self, X, y=None):
        # FastICA sometimes stops at max_iter on this data (it oscillates). We
        # checked: k-NN and linear-SVM accuracy is identical with converged and
        # unconverged fits, because ICA output is a rotation of whitened PCA and
        # those two classifiers do not change under rotation. So we keep the
        # standard settings, hide the warning, and record the iteration count.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.model.fit(X)
        self.n_iter_ = self.model.n_iter_
        self.converged_ = self.n_iter_ < self.model.max_iter
        return self

    def transform(self, X):
        return self.model.transform(X)


def make_projection(name):
    """Build a projection from its short name, e.g. 'pca50', 'ica20', 'lda5'."""
    if name == "none":
        return Identity()
    if name.startswith("pca") and "-" in name:
        return PCAthenLDA()
    if name.startswith("pca"):
        return PCA(int(name[3:]))
    if name.startswith("ica"):
        return ICA(int(name[3:]))
    if name.startswith("lda"):
        return LDA(int(name[3:]))
    raise ValueError(name)
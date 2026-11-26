"""Unit tests. Run:  uv run pytest -q"""

import numpy as np
import pytest
import scipy.linalg as sla
from sklearn.decomposition import PCA as SkPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GroupKFold

from har_noise.classifiers import GRIDS, tune
from har_noise.corruption import (
    CORRUPTIONS,
    gaussian_noise,
    group_dropout,
    n_groups_to_drop,
    ood_injection,
    train_stats,
)
from har_noise.data import DEFAULT_ROOT, dedupe_names, load_har
from har_noise.experiment import corruption_rng, run_sweep
from har_noise.metrics import breakdown_point, degradation_slope, fisher_criterion, intervals_overlap
from har_noise.projection import ICA, LDA, PCA, PCAthenLDA, make_projection

needs_data = pytest.mark.skipif(not DEFAULT_ROOT.exists(), reason="UCI HAR data not downloaded")


@pytest.fixture(scope="module")
def har():
    return load_har()


@pytest.fixture
def toy():
    """Small 3-class dataset with a duplicated column (so Sw is singular)."""
    rng = np.random.default_rng(0)
    means = np.array([[0, 0, 0, 0], [3, 0, 1, 0], [0, 3, 0, 1]], float)
    X = np.vstack([rng.normal(m, 1.0, (60, 4)) for m in means])
    X = np.hstack([X, X[:, :1]])  # column 4 == column 0
    y = np.repeat([0, 1, 2], 60)
    groups = {"g1": np.array([0, 1]), "g2": np.array([2, 3]), "g3": np.array([4])}
    return X, y, groups


# ---------- data ----------
def test_dedupe_names():
    assert dedupe_names(["a", "b", "a", "a"]) == ["a", "b", "a__2", "a__3"]


@needs_data
def test_official_split(har):
    assert har.X_train.shape == (7352, 561) and har.X_test.shape == (2947, 561)
    assert set(har.s_train).isdisjoint(har.s_test)
    assert len(set(har.s_train)) == 21 and len(set(har.s_test)) == 9
    assert set(np.unique(har.y_train)) == set(range(6))
    assert not np.isnan(har.X_train).any()
    assert len(set(har.feature_names)) == 561


@needs_data
def test_signal_groups(har):
    assert len(har.groups) == 17
    cols = np.concatenate(list(har.groups.values()))
    assert len(cols) == len(set(cols)) == 554  # disjoint, angle() features excluded
    assert all(not har.feature_names[i].startswith("angle(") for i in cols)


# ---------- corruption ----------
@pytest.mark.parametrize("name", list(CORRUPTIONS))
def test_level_zero_is_unchanged_copy(name, toy):
    X, _, groups = toy
    out = CORRUPTIONS[name](X, 0.0, np.random.default_rng(0), train_stats(X, groups))
    assert np.array_equal(out, X) and out is not X


@pytest.mark.parametrize("name", list(CORRUPTIONS))
def test_input_never_modified(name, toy):
    X, _, groups = toy
    before = X.copy()
    CORRUPTIONS[name](X, 0.4, np.random.default_rng(1), train_stats(X, groups))
    assert np.array_equal(X, before)


@pytest.mark.parametrize("name", list(CORRUPTIONS))
def test_same_seed_same_corruption(name, toy):
    X, _, groups = toy
    st = train_stats(X, groups)
    a = CORRUPTIONS[name](X, 0.3, np.random.default_rng(7), st)
    b = CORRUPTIONS[name](X, 0.3, np.random.default_rng(7), st)
    assert np.array_equal(a, b)


def test_gaussian_noise_scale():
    X = np.zeros((20000, 3))
    st = {"std": np.array([1.0, 2.0, 0.5]), "mean": np.zeros(3), "groups": {}}
    noise = gaussian_noise(X, 0.3, np.random.default_rng(0), st) - X
    assert np.allclose(noise.std(0), 0.3 * st["std"], rtol=0.03)
    assert np.allclose(noise.mean(0), 0, atol=0.02)


def test_n_groups_to_drop():
    assert [n_groups_to_drop(l, 17) for l in (0, 0.1, 0.2, 0.3, 0.4, 0.5)] == [0, 2, 3, 5, 7, 9]


def test_dropout_zeros_whole_groups(toy):
    X, _, groups = toy
    X = X + 10  # no natural zeros
    out = group_dropout(X, 0.5, np.random.default_rng(3), train_stats(X, groups))
    zero_cols = set(np.where((out == 0).all(0))[0])
    dropped = [g for g, idx in groups.items() if set(idx) <= zero_cols]
    assert len(dropped) == n_groups_to_drop(0.5, 3) == 2
    assert zero_cols == set(np.concatenate([groups[g] for g in dropped]))


def test_ood_replaces_exact_rows(toy):
    X, _, groups = toy
    out = ood_injection(X, 0.25, np.random.default_rng(0), train_stats(X, groups))
    changed = (out != X).any(1)
    assert changed.sum() == round(0.25 * len(X))
    assert out[changed].min() >= -1 and out[changed].max() <= 1


# ---------- projection ----------
@needs_data
def test_pca_matches_sklearn(har):
    ours = PCA(20).fit(har.X_train)
    sk = SkPCA(20, svd_solver="full").fit(har.X_train)
    assert np.allclose(ours.explained_variance_, sk.explained_variance_, rtol=1e-6)
    # same directions up to sign
    cos = np.abs(np.sum(ours.components_ * sk.components_.T, axis=0))
    assert np.allclose(cos, 1, atol=1e-6)
    W = ours.components_
    assert np.allclose(W.T @ W, np.eye(20), atol=1e-8)


def test_lda_handles_singular_sw(toy):
    X, y, _ = toy
    lda = LDA(5).fit(X, y)  # asks for 5, only C-1 = 2 possible
    assert lda.rank_Sw_ == 4 and lda.components_.shape == (5, 2)
    assert np.isfinite(lda.transform(X)).all()
    # The textbook inv(Sw) is unusable here. Depending on the maths library it either
    # raises "Singular matrix" (OpenBLAS on Linux) or silently returns huge garbage
    # (Accelerate on macOS), so we check the singularity itself, not the error.
    Sw = sum(((X[y == c] - X[y == c].mean(0)).T @ (X[y == c] - X[y == c].mean(0))) for c in range(3))
    assert np.linalg.matrix_rank(Sw) < Sw.shape[0]
    assert np.linalg.cond(Sw) > 1e12


@needs_data
def test_lda_matches_sklearn_subspace(har):
    """Where Sw is full rank (after PCA to 100 dims) no shrinkage is needed, so our
    LDA must give the same 5-D subspace as sklearn's eigen solver.
    (On the raw 561-D data the two differ on purpose: sklearn's numeric shrinkage
    is applied after standardising each class, ours shrinks toward a scaled identity.)"""
    Z = PCA(100).fit(har.X_train).transform(har.X_train)
    ours = LDA(5, shrinkage=0.0).fit(Z, har.y_train)
    sk = LinearDiscriminantAnalysis(solver="eigen").fit(Z, har.y_train)
    angles = sla.subspace_angles(ours.components_, sk.scalings_[:, :5])
    assert np.degrees(angles).max() < 0.01


def test_lda_within_class_variance_about_one(toy):
    X, y, _ = toy
    Z = LDA(2, shrinkage=1e-6).fit(X, y).transform(X)
    within = np.mean([Z[y == c].var(0, ddof=1) for c in range(3)], axis=0)
    assert np.allclose(within, 1, atol=0.1)


def test_pca_then_lda_and_factory(toy):
    X, y, _ = toy
    p = PCAthenLDA(n_pca=3, n_components=2).fit(X, y)
    assert p.transform(X).shape == (len(X), 2)
    assert make_projection("pca100-lda5").name == "pca100-lda5"
    assert make_projection("ica20").name == "ica20" and make_projection("none").transform(X) is X


def test_ica_records_iterations(toy):
    X, _, _ = toy
    ica = ICA(3).fit(X)
    assert ica.n_iter_ >= 1 and ica.transform(X).shape == (len(X), 3)


# ---------- metrics ----------
def test_fisher_criterion():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 500)
    far = np.r_[rng.normal(0, 1, (500, 2)), rng.normal(10, 1, (500, 2))]
    near = np.r_[rng.normal(0, 1, (500, 2)), rng.normal(0.5, 1, (500, 2))]
    assert fisher_criterion(far, y) > 20
    assert fisher_criterion(near, y) < 1
    assert np.isclose(fisher_criterion(far + 5, y), fisher_criterion(far, y))


def test_slope_breakdown_overlap():
    lv = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    assert np.isclose(degradation_slope(lv, [1, 0.9, 0.8, 0.7, 0.6, 0.5]), 10.0)
    assert breakdown_point(lv, [0.9, 0.8, 0.75, 0.69, 0.6, 0.5]) == 0.3
    assert breakdown_point(lv, [0.9] * 6) is None
    assert intervals_overlap(0.8, 0.05, 0.85, 0.01) and not intervals_overlap(0.8, 0.01, 0.9, 0.01)


# ---------- CV and sweep ----------
@needs_data
def test_cv_folds_never_share_subjects(har):
    for tr, va in GroupKFold(5).split(har.X_train, har.y_train, har.s_train):
        assert set(har.s_train[tr]).isdisjoint(har.s_train[va])


def test_tune_picks_from_grid(toy):
    X, y, _ = toy
    subjects = np.tile(np.arange(10), 18)
    best, rows = tune("lda5", X, y, subjects)
    for clf, params in best.items():
        assert params in GRIDS[clf]
    assert len(rows) == 5 * sum(len(g) for g in GRIDS.values())


def test_corruption_rng_reproducible():
    a = corruption_rng("gaussian", 0.3, 2).normal(size=3)
    b = corruption_rng("gaussian", 0.3, 2).normal(size=3)
    c = corruption_rng("gaussian", 0.4, 2).normal(size=3)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_sweep_clean_rows_identical_across_types(toy):
    from types import SimpleNamespace

    from har_noise.classifiers import fit_final

    X, y, groups = toy
    data = SimpleNamespace(X_train=X, y_train=y, X_test=X, y_test=y, groups=groups)
    best = {"knn": {"n_neighbors": 3}, "svm": {"C": 0.1}, "gnb": {"var_smoothing": 1e-9}}
    fitted = {"lda5": fit_final("lda5", X, y, best)}
    res, cms = run_sweep(data, fitted, levels=[0.0, 0.3], seeds={"gaussian": 2, "dropout": 2, "ood": 1})
    clean = res[res.level == 0].groupby(["projection", "classifier"])["acc"].nunique()
    assert (clean == 1).all()
    assert ("dropout", 0.0, "lda5", "svm") in cms and ("ood", 0.3, "lda5", "knn") in cms
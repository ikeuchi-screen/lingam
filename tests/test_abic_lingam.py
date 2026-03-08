import importlib.util
import os

import numpy as np
import pandas as pd
import pytest
from lingam.abic_lingam import ABICLiNGAM, ABICLiNGAM_GPU
from lingam.utils._mggd import MGGD


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_data():
    """4変数・100サンプルの合成データ。fit / bootstrap テストで共用。"""
    d = 4
    B_true = np.zeros((d, d), dtype=float)
    B_true[0, 1] = 1.0   # x0 -> x1
    B_true[1, 2] = -1.5  # x1 -> x2
    B_true[2, 3] = 1.0   # x2 -> x3

    omega_true = np.array(
        [
            [1.2, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.6],  # x1 <-> x3
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.6, 0.0, 1.0],
        ],
        dtype=float,
    )

    beta_shape = 3.0
    mggd = MGGD(np.zeros(d), omega_true, beta_shape)
    eps = mggd.rvs(size=100)

    A = np.eye(d) - B_true
    X = eps @ np.linalg.inv(A)
    X -= X.mean(axis=0, keepdims=True)
    return X, beta_shape


# ---------------------------------------------------------------------------
# ABICLiNGAM (CPU / original)
# ---------------------------------------------------------------------------


def test_fit_success(sample_data):
    X, beta_shape = sample_data
    model = ABICLiNGAM(beta=beta_shape)
    model.fit(X)

    am = model.adjacency_matrix_
    cm = model.coefficient_matrix_
    ecm = model.error_covariance_matrix_


def test_fit_invalid_data():

    try:
        model = ABICLiNGAM(beta=-1.0)
    except ValueError:
        pass
    else:
        raise AssertionError

    try:
        model = ABICLiNGAM(lam=-0.5)
    except ValueError:
        pass
    else:
        raise AssertionError

    try:
        model = ABICLiNGAM(acyc_order=0)
    except ValueError:
        pass
    else:
        raise AssertionError

    try:
        model = ABICLiNGAM(acyc_order=1.5)
    except TypeError:
        pass
    else:
        raise AssertionError

    try:
        model = ABICLiNGAM(min_causal_effect=-0.1)
    except ValueError:
        pass
    else:
        raise AssertionError

    try:
        model = ABICLiNGAM(min_error_covariance=-0.1)
    except ValueError:
        pass
    else:
        raise AssertionError

    try:
        model = ABICLiNGAM(max_outer=0)
    except ValueError:
        pass
    else:
        raise AssertionError

    try:
        model = ABICLiNGAM(tol_h=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError

    try:
        model = ABICLiNGAM(rho_max=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError

    try:
        model = ABICLiNGAM(inner_start=0)
    except ValueError:
        pass
    else:
        raise AssertionError

    try:
        model = ABICLiNGAM(inner_growth=-1)
    except ValueError:
        pass
    else:
        raise AssertionError

    try:
        model = ABICLiNGAM(inner_tol=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError

    model = ABICLiNGAM()

    try:
        model.fit(np.array([1, 2, 3]))
    except ValueError:
        pass
    else:
        raise AssertionError

    try:
        model.fit(np.array([[1], [2], [3]]))
    except ValueError:
        pass
    else:
        raise AssertionError

    try:
        model.fit(pd.DataFrame())
    except ValueError:
        pass
    else:
        raise AssertionError


def test_bootstrap_success(sample_data):
    X, beta_shape = sample_data
    model = ABICLiNGAM(beta=beta_shape)
    result = model.bootstrap(X, n_sampling=2)

    am_samples = result.adjacency_matrices_
    cm_samples = result.coefficient_matrices_
    ecm_samples = result.error_covariance_matrices_


# ---------------------------------------------------------------------------
# ABICLiNGAM_GPU — torch 不要テスト
# ---------------------------------------------------------------------------


def test_gpu_is_subclass():
    """ABICLiNGAM_GPU は ABICLiNGAM のサブクラスであること。"""
    assert issubclass(ABICLiNGAM_GPU, ABICLiNGAM)


def test_gpu_importerror(sample_data):
    """torch 未インストール時に fit() が ImportError を送出すること。"""
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch is installed; skipping ImportError test")

    X, beta_shape = sample_data
    model = ABICLiNGAM_GPU(beta=beta_shape)
    with pytest.raises(ImportError, match="pip install torch"):
        model.fit(X)


# ---------------------------------------------------------------------------
# ABICLiNGAM_GPU — torch 必要テスト (torch 未インストール時は自動スキップ)
# ---------------------------------------------------------------------------


def test_gpu_fit_success(sample_data):
    """ABICLiNGAM_GPU.fit() が正常に完了し、出力 shape が正しいこと。"""
    pytest.importorskip("torch")

    X, beta_shape = sample_data
    d = X.shape[1]

    model = ABICLiNGAM_GPU(beta=beta_shape)
    model.fit(X)

    cm = model.coefficient_matrix_
    ecm = model.error_covariance_matrix_
    am = model.adjacency_matrix_
    co = model.causal_order_

    assert cm.shape == (d, d)
    assert ecm.shape == (d, d)
    assert am.shape == (d, d)
    assert isinstance(co, list) and len(co) == d


def test_gpu_fit_invalid_data():
    """ABICLiNGAM_GPU が ABICLiNGAM と同じパラメータ検証を行うこと。"""
    pytest.importorskip("torch")

    with pytest.raises(ValueError):
        ABICLiNGAM_GPU(beta=-1.0)

    with pytest.raises(ValueError):
        ABICLiNGAM_GPU(lam=-0.5)

    with pytest.raises(ValueError):
        ABICLiNGAM_GPU(acyc_order=0)

    with pytest.raises(TypeError):
        ABICLiNGAM_GPU(acyc_order=1.5)

    with pytest.raises(ValueError):
        ABICLiNGAM_GPU(min_causal_effect=-0.1)

    with pytest.raises(ValueError):
        ABICLiNGAM_GPU(min_error_covariance=-0.1)

    with pytest.raises(ValueError):
        ABICLiNGAM_GPU(max_outer=0)

    with pytest.raises(ValueError):
        ABICLiNGAM_GPU(tol_h=0.0)

    with pytest.raises(ValueError):
        ABICLiNGAM_GPU(rho_max=0.0)

    with pytest.raises(ValueError):
        ABICLiNGAM_GPU(inner_start=0)

    with pytest.raises(ValueError):
        ABICLiNGAM_GPU(inner_growth=-1)

    with pytest.raises(ValueError):
        ABICLiNGAM_GPU(inner_tol=0.0)

    model = ABICLiNGAM_GPU()

    with pytest.raises(ValueError):
        model.fit(np.array([1, 2, 3]))

    with pytest.raises(ValueError):
        model.fit(np.array([[1], [2], [3]]))

    with pytest.raises(ValueError):
        model.fit(pd.DataFrame())


def test_gpu_bootstrap_success(sample_data):
    """ABICLiNGAM_GPU.bootstrap() が正常に完了し、出力 shape が正しいこと。"""
    pytest.importorskip("torch")

    X, beta_shape = sample_data
    d = X.shape[1]
    n_sampling = 2

    model = ABICLiNGAM_GPU(beta=beta_shape)
    result = model.bootstrap(X, n_sampling=n_sampling)

    assert result.adjacency_matrices_.shape == (n_sampling, d, d)
    assert result.coefficient_matrices_.shape == (n_sampling, d, d)
    assert result.error_covariance_matrices_.shape == (n_sampling, d, d)

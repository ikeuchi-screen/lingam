"""
Python implementation of the LiNGAM algorithms.
The LiNGAM Project: https://sites.google.com/view/sshimizu06/lingam
"""

import math
import numpy as np
import warnings

import autograd.numpy as anp
from autograd import grad
import scipy.optimize as sopt
import functools
from sklearn.utils import check_array, resample
from sklearn.linear_model import LinearRegression

from .base import _BaseLiNGAM
from .bootstrap import BootstrapResult


class ABICLiNGAM(_BaseLiNGAM):
    """Implementation of ABIC-LiNGAM Algorithm. [1]_
    Original code: https://github.com/Yoshimitsu-try/ABIC_LiNGAM

    References
    ----------
    .. [1] Y. Morinishi and S. Shimizu. Differentiable causal discovery of
       linear non-Gaussian acyclic models under unmeasured confounding.
       Transactions on Machine Learning Research (TMLR), 2025.
    """

    def __init__(
        self,
        beta=1.0,
        lam=0.05,
        acyc_order=None,
        seed=0,
        max_outer=100,
        tol_h=1e-8,
        min_causal_effect=0.05,
        min_error_covariance=0.05,
        rho_max=1e16,
        inner_start=1,
        inner_growth=1,
        inner_tol=1e-4,
    ):
        """Construct a ABICLiNGAM model.

        Parameters
        ----------
        beta : float, optional (default=1.0)
            Power in residual loss, i.e., ||r||^(2*beta)
        lam : float, optional (default=0.05)
            The weight of the regularization term.
        acyc_order : int or None, optional (default=None)
            Order of the truncated series for acyclicity penalty. If None, defaults to the number of variables.
        min_causal_effect : float, optional (default=0.05)
            Threshold for detecting causal edge.
            Causal edges with absolute values of causal effects less than ``min_causal_effect`` are excluded.
        min_error_covariance : float, optional (default=0.05)
            Threshold for detecting error covariances.
            Error covariances with absolute values less than ``min_error_covariance`` are excluded.
        seed : int, optional (default=0)
            Seed for the random number generator.
        max_outer : int, optional (default=100)
            Maximum number of outer iterations.
        tol_h : float, optional (default=1e-8)
            Tolerance for acyclicity penalty to stop.
        rho_max : float, optional (default=1e16)
            Maximum value for Augmented Lagrangian penalty parameter rho.
        inner_start : int, optional (default=1)
            Initial number of inner refinement steps.
        inner_growth : int, optional (default=1)
            Growth of inner refinement steps per outer iteration.
        inner_tol : float, optional (default=1e-4)
            Tolerance for inner loop convergence.
        """
        # Check parameters
        if beta <= 0.0:
            raise ValueError("beta must be positive.")
        if lam < 0.0:
            raise ValueError("lam must be non-negative.")
        if acyc_order is not None:
            if not isinstance(acyc_order, int):
                raise TypeError("acyc_order must be an integer or None.")
            if acyc_order < 1:
                raise ValueError("acyc_order must be >= 1.")
        if min_causal_effect < 0.0:
            raise ValueError("min_causal_effect must be non-negative.")
        if min_error_covariance < 0.0:
            raise ValueError("min_error_covariance must be non-negative.")
        if max_outer < 1:
            raise ValueError("max_outer must be at least 1.")
        if tol_h <= 0.0:
            raise ValueError("tol_h must be positive.")
        if rho_max <= 0.0:
            raise ValueError("rho_max must be positive.")
        if inner_start < 1:
            raise ValueError("inner_start must be at least 1.")
        if inner_growth < 0:
            raise ValueError("inner_growth must be non-negative.")
        if inner_tol <= 0.0:
            raise ValueError("inner_tol must be positive.")

        self._beta = float(beta)
        self._lam = float(lam)
        self._acyc_order = acyc_order
        self._min_causal_effect = float(min_causal_effect)
        self._min_error_covariance = float(min_error_covariance)
        self._seed = int(seed)
        self._max_outer = int(max_outer)
        self._tol_h = float(tol_h)
        self._rho_max = float(rho_max)
        self._inner_start = int(inner_start)
        self._inner_growth = int(inner_growth)
        self._inner_tol = float(inner_tol)

        super().__init__()

    def fit(self, X):
        """Fit the model to X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Observed data matrix.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._X = anp.asarray(check_array(X))
        d = self._X.shape[1]

        # Random number generator
        self._rng = np.random.default_rng(self._seed)

        # Check parameters
        if d < 2:
            raise ValueError("Data must have at least two variables (features).")

        # Initialize parameters
        B = anp.array(self._rng.uniform(-0.5, 0.5, size=(d, d)))
        L = anp.array(self._rng.uniform(-0.05, 0.05, size=(d, d)))
        lower_mask = anp.array(np.tril(np.ones((d, d)), k=-1))
        L = L * lower_mask
        L = L + L.T
        L = L - anp.diag(anp.diag(L))
        D = anp.diag(anp.diag(anp.cov(self._X.T)))

        rho, alpha, h_prev = 1.0, 0.0, np.inf
        inner_cap = self._inner_start
        penalty_fn = self._bow_penalty

        bounds = self._build_bounds()  # no prior knowledge
        objective = functools.partial(self._objective)
        gradient = grad(objective)

        for _ in range(self._max_outer):
            B_new, L_new, D_new = None, None, None
            h_new = None

            while rho < self._rho_max:
                B_new = B.copy()
                L_new = L.copy()
                D_new = D.copy()

                # inner refinement
                for _ in range(inner_cap):
                    B_old, L_old, D_old = B_new, L_new, D_new
                    Z = self._pseudo(B_new, L_new + D_new)

                    theta0 = anp.concatenate([anp.ravel(B_new), anp.ravel(L_new)])
                    res = sopt.minimize(
                        self._objective,
                        theta0,
                        args=(rho, alpha, Z, penalty_fn),
                        method="L-BFGS-B",
                        jac=gradient,
                        bounds=bounds,
                        options={"disp": False},
                    )

                    B_new = anp.reshape(res.x[: d * d], (d, d))
                    L_new = anp.reshape(res.x[d * d :], (d, d))
                    L_new = L_new + L_new.T
                    L_new = L_new - anp.diag(anp.diag(L_new))

                    # refresh diagonal noise from residuals (different expression)
                    diag_vals = [
                        anp.var(self._X[:, j] - anp.dot(self._X, B_new[:, j]))
                        for j in range(d)
                    ]
                    D_new = anp.diag(anp.array(diag_vals))

                    # convergence of inner loop
                    delta = anp.sum(anp.abs(B_old - B_new)) + anp.sum(
                        anp.abs((L_old + D_old) - (L_new + D_new))
                    )
                    if float(delta) < self._inner_tol:
                        break

                h_new = self._acyclicity_penalty(B_new) + penalty_fn(B_new, L_new)

                # penalty schedule
                if float(h_new) < 0.25 * float(h_prev):
                    break
                else:
                    rho *= 10.0

            # AL update
            B, L, D = B_new.copy(), L_new.copy(), D_new.copy()
            h_prev = h_new
            alpha = alpha + rho * h_prev
            inner_cap += self._inner_growth

            if float(h_prev) <= self._tol_h or rho >= self._rho_max:
                break

        self._B = anp.where(anp.abs(B) < self._min_causal_effect, 0.0, B)
        self._omega = anp.where(
            anp.abs(L + D) < self._min_error_covariance, 0.0, (L + D)
        )

        # Merge coefficient matrix and error covariance matrix
        omega_copied = self._omega.copy()
        anp.fill_diagonal(omega_copied, 0.0)
        omega_copied[anp.abs(omega_copied) > 0] = anp.nan
        self._adjacency_matrix = self._B.T.copy()
        self._adjacency_matrix[anp.isnan(omega_copied)] = anp.nan

        # Estimate causal order from coefficient matrix
        self._causal_order = self._causal_order_from_adjacency_matrix(self._B.T)

        return self

    def bootstrap(self, X, n_sampling=100):
        """Bootstrap sampling to assess variability of estimates.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Observed data matrix.
        n_sampling : int, optional (default=100)
            Number of bootstrap samples.

        Returns
        -------
        Bs : array-like, shape (n_sampling, n_features, n_features)
            Bootstrap samples of estimated adjacency matrices.
        Omegas : array-like, shape (n_sampling, n_features, n_features)
            Bootstrap samples of estimated error covariance matrices.
        """
        X = anp.asarray(X)
        d = X.shape[1]

        adjacency_matrices = anp.zeros((n_sampling, d, d))
        Bs = anp.zeros((n_sampling, d, d))
        Omegas = anp.zeros((n_sampling, d, d))
        total_effects = anp.zeros((n_sampling, d, d))
        index = anp.arange(X.shape[0])
        resampled_indices = []

        for i in range(n_sampling):
            resampled_X, resampled_index = resample(X, index)
            resampled_indices.append(resampled_index)
            self.fit(resampled_X)
            adjacency_matrices[i] = self._adjacency_matrix
            Bs[i] = self._B
            Omegas[i] = self._omega

            # Calculate total effects
            for c, from_ in enumerate(self._causal_order):
                for to in self._causal_order[c + 1 :]:
                    if True in np.isnan(self._adjacency_matrix[from_]):
                        total_effects[i, to, from_] = np.nan
                    else:
                        total_effects[i, to, from_] = self.estimate_total_effect(
                            resampled_X, from_, to
                        )

        return ABICBootstrapResult(
            adjacency_matrices,
            Bs,
            Omegas,
            total_effects,
            resampled_indices=resampled_indices,
        )

    def _acyclicity_penalty(self, W, K=None):
        """Smooth acyclicity surrogate written as a truncated series:
            h(W) = sum_{k=1..K} trace((W∘W)^k) / k!
        where ∘ is Hadamard product. K defaults to d.
        This differs in form/implementation from common "M=I+..." variants
        and avoids any custom VJP; autograd handles the gradient.

        Parameters
        ----------
        W : array-like, shape (d, d)
            Directed adjacency matrix.
        K : int or None
            Order of the truncated series. If None, defaults to d.

        Returns
        -------
        penalty : float
            Value of the acyclicity penalty.
        """
        d = W.shape[0]
        if K is None:
            K = self._acyc_order or d
        A = W * W
        Ak = anp.eye(d)
        acc = 0.0
        for k in range(1, K + 1):
            Ak = anp.dot(Ak, A)
            acc = acc + anp.trace(Ak) / float(math.factorial(k))
        return acc

    @staticmethod
    def _bow_penalty(W1, W2):
        """Bow-freeness surrogate in an alternative form:
            || W1 ∘ W2 ||_F^2 / |W1|

        Parameters
        ----------
        W1 : array-like, shape (d, d)
            Directed adjacency matrix.
        W2 : array-like, shape (d, d)
            Bidirected adjacency matrix.

        Returns
        -------
        penalty : float
            Value of the bow-freeness penalty.
        """
        A = W1 * W2
        return anp.sum(A * A) / A.size

    def _objective(self, theta, rho, alpha, Z, penalty_fn):
        """Augmented Lagrangian objective. All pieces are auto-diff compatible.

        Parameters
        ----------
        theta : array-like, shape (d*d + d*d,)
            Concatenated parameter vector [vec(B), vec(L)], where L is strictly lower
            triangular part to be mirrored to form a symmetric matrix.
        rho : float
            The weight of the penalty term
        alpha : float
            The Lagrange multiplier
        Z : list of array-like, shape (n, d)
            Pseudo-variables for bidirected part.
        penalty_fn : callable
            Structural penalty function on (B, L_sym).

        Returns
        -------
        obj : float
            Value of the objective function.
        """

        n, d = self._X.shape

        # unpack and enforce symmetry for the bidirected part
        B = anp.reshape(theta[: d * d], (d, d))
        L = anp.reshape(theta[d * d :], (d, d))
        L = L + L.T
        L = L - anp.diag(anp.diag(L))  # zero diagonal

        # LS(theta) term (with generalized power 2*beta)
        LS = 0.0
        for j in range(d):
            r = self._X[:, j] - anp.dot(self._X, B[:, j]) - anp.dot(Z[j], L[:, j])
            LS = LS + 0.5 / n * (anp.linalg.norm(r) ** (2 * self._beta))

        # structural constraints
        h = self._acyclicity_penalty(B) + penalty_fn(B, L)
        aug = 0.5 * rho * (h**2) + alpha * h

        # smooth L0-ish (tanh-like) regularization on theta
        s = anp.log(n) * anp.abs(theta)
        t = (anp.exp(s) - 1) / (anp.exp(s) + 1)
        return LS + aug + self._lam * anp.sum(t)

    def _build_bounds(self, levels=None, exogenous=(), w_range=4.0):
        """Create L-BFGS-B bounds for theta

        Parameters
        ----------
        levels : list[list[index]] or None
            Prior knowledge about variable ordering. Each sublist represents a level,
            where variables in earlier levels cannot have incoming edges from later levels.
        exogenous : array-like, shape (index, ...)
            Indices of exogenous variables (no incoming edges).
        w_range : float
            Range for weights (default: 4.0). Bounds are set to [-w_range, w_range].

        Returns
        -------
        bounds : list[tuple[float, float]]
            Bounds for each parameter in theta = [vec(B), vec(L)].
        """
        d = self._X.shape[1]

        if levels is None:
            levels = [[i for i in range(d)]]

        tier = {v: t for t, group in enumerate(levels) for v in group}
        exo = set(exogenous)

        # Directed bounds B: start wide, zero diag, then forbid backward edges
        B_lo = -w_range * np.ones((d, d))
        B_hi = +w_range * np.ones((d, d))
        np.fill_diagonal(B_lo, 0.0)
        np.fill_diagonal(B_hi, 0.0)
        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                if tier[i] > tier[j]:
                    B_lo[i, j] = 0.0
                    B_hi[i, j] = 0.0

        # Bidirected bounds L (we optimize lower-tri; upper/diag fixed zero)
        L_lo = -w_range * np.ones((d, d))
        L_hi = +w_range * np.ones((d, d))
        for i in range(d):
            for j in range(d):
                if i <= j or (i in exo) or (j in exo):
                    L_lo[i, j] = 0.0
                    L_hi[i, j] = 0.0

        bounds_B = np.c_[B_lo.reshape(-1), B_hi.reshape(-1)].tolist()
        bounds_L = np.c_[L_lo.reshape(-1), L_hi.reshape(-1)].tolist()
        return bounds_B + bounds_L

    def _pseudo(self, B, omega):
        """
        Build pseudo-variables Z using solves instead of explicit inversion.

        Parameters
        ----------
        B : array-like, shape (d, d)
            Current estimate of directed adjacency matrix.
        omega : array-like, shape (d, d)
            Current estimate of error covariance matrix.

        Returns
        -------
        Z : list of array-like, shape (n, d)
            Returns a list Z such that Z[j] has a zero column at j (shape (n, d)).
        """
        d = B.shape[0]
        eps = self._X - anp.dot(self._X, B)
        Z = [None] * d
        for j in range(d):
            idx = [k for k in range(d) if k != j]
            omega_ = omega[anp.ix_(idx, idx)]
            Zij = anp.linalg.solve(omega_, eps[:, idx].T).T  # (n, d-1)
            Zj = anp.insert(Zij, j, 0.0, axis=1)  # (n, d)
            Z[j] = Zj
        return Z

    def _causal_order_from_adjacency_matrix(self, matrix, threshold=1e-8):
        """Estimate causal order from adjacency matrix using Kahn's algorithm."""
        d = matrix.shape[0]
        # 有向グラフの隣接リストと入次数を作成
        adj = {i: set() for i in range(d)}
        indegree = [0] * d
        for i in range(d):
            for j in range(d):
                if i != j and abs(matrix[i, j]) > threshold:
                    adj[i].add(j)
                    indegree[j] += 1

        # 入次数0のノードから順に並べる
        order = []
        queue = [i for i in range(d) if indegree[i] == 0]
        while queue:
            n = queue.pop(0)
            order.append(n)
            for m in adj[n]:
                indegree[m] -= 1
                if indegree[m] == 0:
                    queue.append(m)

        if len(order) != d:
            raise ValueError("The adjacency matrix contains a cycle (not a DAG).")

        return order[::-1]

    def estimate_total_effect(self, X, from_index, to_index):
        """Estimate total effect using causal model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Original data, where n_samples is the number of samples
            and n_features is the number of features.
        from_index :
            Index of source variable to estimate total effect.
        to_index :
            Index of destination variable to estimate total effect.

        Returns
        -------
        total_effect : float
            Estimated total effect.
        """
        # Check parameters
        X = check_array(X)

        # Check confounders
        if True in np.isnan(self._adjacency_matrix[from_index]):
            warnings.warn(
                f"The estimated causal effect may be incorrect because "
                f"the source variable (from_index={from_index}) is influenced by confounders."
            )
            return np.nan

        # from_index + parents indices
        parents = np.where(np.abs(self._adjacency_matrix[from_index]) > 0)[0]
        predictors = [from_index]
        predictors.extend(parents)

        # Estimate total effect
        lr = LinearRegression()
        lr.fit(X[:, predictors], X[:, to_index])

        return lr.coef_[0]

    @property
    def coefficient_matrix_(self):
        """Estimated coefficient matrix.

        Returns
        -------
        coefficient_matrix_ : array-like, shape (n_features, n_features)
            The coefficient matrix B of fitted model, where
            n_features is the number of features.
        """
        return self._B

    @property
    def error_covariance_matrix_(self):
        """Estimated error covariance matrix.

        Returns
        -------
        error_covariance_matrix_ : array-like, shape (n_features, n_features)
            The error covariance matrix Omega of fitted model, where
            n_features is the number of features.
        """
        return self._omega


class ABICLiNGAM_GPU(ABICLiNGAM):
    """GPU-accelerated implementation of ABIC-LiNGAM using PyTorch. [1]_

    Replaces the autograd + scipy L-BFGS-B optimization stack with
    PyTorch tensors and ``torch.optim.LBFGS``, enabling GPU acceleration
    via CUDA. Falls back to CPU if CUDA is unavailable.

    Requires PyTorch >= 2.0: ``pip install torch``

    References
    ----------
    .. [1] Y. Morinishi and S. Shimizu. Differentiable causal discovery of
       linear non-Gaussian acyclic models under unmeasured confounding.
       Transactions on Machine Learning Research (TMLR), 2025.
    """

    def __init__(
        self,
        beta=1.0,
        lam=0.05,
        acyc_order=None,
        seed=0,
        max_outer=100,
        tol_h=1e-8,
        min_causal_effect=0.05,
        min_error_covariance=0.05,
        rho_max=1e16,
        inner_start=1,
        inner_growth=1,
        inner_tol=1e-4,
    ):
        """Construct a ABICLiNGAM_GPU model.

        Parameters are identical to :class:`ABICLiNGAM`.
        PyTorch is required (``pip install torch``).
        CUDA is used automatically when available; otherwise falls back to CPU.
        """
        super().__init__(
            beta=beta,
            lam=lam,
            acyc_order=acyc_order,
            seed=seed,
            max_outer=max_outer,
            tol_h=tol_h,
            min_causal_effect=min_causal_effect,
            min_error_covariance=min_error_covariance,
            rho_max=rho_max,
            inner_start=inner_start,
            inner_growth=inner_growth,
            inner_tol=inner_tol,
        )

    def fit(self, X):
        """Fit the model to X using PyTorch (GPU if CUDA available).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Observed data matrix.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        try:
            import torch
        except ImportError:
            raise ImportError(
                "PyTorch is required for ABICLiNGAM_GPU. "
                "Install with: pip install torch"
            )

        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._X = anp.asarray(check_array(X))
        d = self._X.shape[1]
        self._rng = np.random.default_rng(self._seed)

        if d < 2:
            raise ValueError("Data must have at least two variables (features).")

        # Precompute index structures for batched ops
        self._precompute_index_structures(d)
        self._precompute_free_masks(d)

        # Upload data to device
        self._Xt = torch.tensor(
            np.asarray(self._X), dtype=torch.float64, device=self._device
        )

        # Pre-allocate GPU-resident tensors to avoid CUDA malloc in the inner loop
        B_t = torch.empty(d, d, dtype=torch.float64, device=self._device)
        omega_t = torch.empty(d, d, dtype=torch.float64, device=self._device)
        L_t = torch.empty(d, d, dtype=torch.float64, device=self._device)
        self._b_free_buf = torch.empty(
            self._n_free_B, dtype=torch.float64, device=self._device
        )
        self._l_free_buf = torch.empty(
            self._n_free_L, dtype=torch.float64, device=self._device
        )

        # JIT-compile hot paths (PyTorch >= 2.0 + triton required; skip silently otherwise)
        if hasattr(torch, "compile"):
            try:
                import triton  # noqa: F401 — required by torch.compile backend
                self._pseudo_torch = torch.compile(self._pseudo_torch)
                self._objective_torch = torch.compile(self._objective_torch)
            except (ImportError, Exception):
                pass

        # Initialize parameters (numpy)
        B = np.array(self._rng.uniform(-0.5, 0.5, size=(d, d)))
        L = np.array(self._rng.uniform(-0.05, 0.05, size=(d, d)))
        lower_mask = np.tril(np.ones((d, d)), k=-1)
        L = L * lower_mask
        L = L + L.T
        L = L - np.diag(np.diag(L))
        D = np.diag(np.diag(np.cov(self._X.T)))

        rho, alpha, h_prev = 1.0, 0.0, np.inf
        inner_cap = self._inner_start

        for _ in range(self._max_outer):
            B_new, L_new, D_new = None, None, None
            h_new = None

            while rho < self._rho_max:
                B_new = B.copy()
                L_new = L.copy()
                D_new = D.copy()

                for _ in range(inner_cap):
                    B_old, L_old, D_old = B_new.copy(), L_new.copy(), D_new.copy()

                    # Compute pseudo-variables on device (batched solve)
                    B_t.copy_(torch.from_numpy(B_new))
                    omega_t.copy_(torch.from_numpy(L_new + D_new))
                    Z_t = self._pseudo_torch(B_t, omega_t)

                    # Optimize via PyTorch LBFGS
                    b_np, l_np = self._pack_numpy(B_new, L_new)
                    b_np, l_np = self._lbfgs_torch_inner(
                        b_np, l_np, rho, alpha, Z_t, self._bow_penalty_torch
                    )
                    B_new, L_new_lower = self._unpack_numpy(b_np, l_np, d)
                    L_new = L_new_lower + L_new_lower.T
                    L_new = L_new - np.diag(np.diag(L_new))

                    # Refresh diagonal noise from residuals (GPU vectorised)
                    B_t.copy_(torch.from_numpy(B_new))
                    with torch.no_grad():
                        resid = self._Xt - self._Xt @ B_t  # (n, d)
                        D_new = torch.diag(resid.var(dim=0)).cpu().numpy()

                    delta = np.sum(np.abs(B_old - B_new)) + np.sum(
                        np.abs((L_old + D_old) - (L_new + D_new))
                    )
                    if float(delta) < self._inner_tol:
                        break

                # Compute h via torch (no grad needed) — B_t already up-to-date
                L_t.copy_(torch.from_numpy(L_new))
                with torch.no_grad():
                    h_new = float(
                        (self._acyclicity_penalty_torch(B_t)
                         + self._bow_penalty_torch(B_t, L_t)).item()
                    )

                if float(h_new) < 0.25 * float(h_prev):
                    break
                else:
                    rho *= 10.0

            B, L, D = B_new.copy(), L_new.copy(), D_new.copy()
            h_prev = h_new
            alpha = alpha + rho * h_prev
            inner_cap += self._inner_growth

            if float(h_prev) <= self._tol_h or rho >= self._rho_max:
                break

        self._B = np.where(np.abs(B) < self._min_causal_effect, 0.0, B)
        self._omega = np.where(
            np.abs(L + D) < self._min_error_covariance, 0.0, (L + D)
        )

        # Merge coefficient matrix and error covariance matrix
        omega_copied = self._omega.copy()
        np.fill_diagonal(omega_copied, 0.0)
        omega_copied[np.abs(omega_copied) > 0] = np.nan
        self._adjacency_matrix = self._B.T.copy()
        self._adjacency_matrix[np.isnan(omega_copied)] = np.nan

        self._causal_order = self._causal_order_from_adjacency_matrix(self._B.T)

        return self

    # -------------------------------------------------------------------------
    # PyTorch helper methods
    # -------------------------------------------------------------------------

    def _precompute_index_structures(self, d):
        """Precompute index tensors for batched pseudo-variable computation."""
        torch = self._torch
        device = self._device

        # sub_idx[j] = all indices 0..d-1 except j, shape (d, d-1)
        sub_idx = torch.zeros(d, d - 1, dtype=torch.long, device=device)
        for j in range(d):
            sub_idx[j] = torch.cat([
                torch.arange(j, device=device),
                torch.arange(j + 1, d, device=device),
            ])
        self._sub_idx = sub_idx  # (d, d-1)

        # Indices for building omega_sub: (d, d-1, d-1)
        self._omega_row_idx = sub_idx.unsqueeze(2).expand(d, d - 1, d - 1)
        self._omega_col_idx = sub_idx.unsqueeze(1).expand(d, d - 1, d - 1)

        # insert_target[j] = sub_idx[j], used for scatter into zero column
        self._insert_target = sub_idx  # (d, d-1)

    def _precompute_free_masks(self, d, exogenous=()):
        """Build boolean masks for free (non-fixed-zero) parameters."""
        exo = set(exogenous)

        # B: all off-diagonal entries are free (diagonal fixed to 0)
        free_B = np.ones((d, d), dtype=bool)
        np.fill_diagonal(free_B, False)

        free_B_rows, free_B_cols = np.where(free_B)
        self._free_B_rows_np = free_B_rows
        self._free_B_cols_np = free_B_cols
        self._n_free_B = int(free_B.sum())

        torch = self._torch
        device = self._device
        self._free_B_rows_t = torch.tensor(free_B_rows, dtype=torch.long, device=device)
        self._free_B_cols_t = torch.tensor(free_B_cols, dtype=torch.long, device=device)

        # L: only strictly lower-triangular entries, excluding exogenous vars
        lower_tri_r, lower_tri_c = np.tril_indices(d, k=-1)
        free_L_1d = np.array([
            (i not in exo) and (j not in exo)
            for i, j in zip(lower_tri_r, lower_tri_c)
        ], dtype=bool)

        self._lower_tri_r = lower_tri_r
        self._lower_tri_c = lower_tri_c
        self._free_L_mask_1d_np = free_L_1d
        self._n_free_L = int(free_L_1d.sum())
        self._n_lower_tri = len(lower_tri_r)

        self._lower_tri_r_t = torch.tensor(lower_tri_r, dtype=torch.long, device=device)
        self._lower_tri_c_t = torch.tensor(lower_tri_c, dtype=torch.long, device=device)
        self._free_L_idx_in_lower_t = torch.tensor(
            np.where(free_L_1d)[0], dtype=torch.long, device=device
        )

    def _pack_numpy(self, B_np, L_np):
        """Extract free parameters as numpy arrays."""
        b_free = B_np[self._free_B_rows_np, self._free_B_cols_np]
        l_lower = L_np[self._lower_tri_r, self._lower_tri_c]
        l_free = l_lower[self._free_L_mask_1d_np]
        return b_free, l_free

    def _unpack_numpy(self, b_free_np, l_free_np, d):
        """Reconstruct full B (d,d) and lower-tri L (d,d) from free params."""
        B = np.zeros((d, d), dtype=np.float64)
        B[self._free_B_rows_np, self._free_B_cols_np] = b_free_np

        l_lower_full = np.zeros(self._n_lower_tri, dtype=np.float64)
        l_lower_full[self._free_L_mask_1d_np] = l_free_np
        L = np.zeros((d, d), dtype=np.float64)
        L[self._lower_tri_r, self._lower_tri_c] = l_lower_full
        return B, L

    def _unpack_torch(self, b_free, l_free):
        """Reconstruct B (d,d) and symmetric L (d,d) from free tensors.

        Differentiable w.r.t. b_free and l_free via index_put/scatter.
        """
        torch = self._torch
        device = self._device
        d = self._Xt.shape[1]

        B = torch.zeros(d, d, dtype=torch.float64, device=device)
        B = B.index_put((self._free_B_rows_t, self._free_B_cols_t), b_free)

        l_lower_full = torch.zeros(
            self._n_lower_tri, dtype=torch.float64, device=device
        )
        l_lower_full = l_lower_full.scatter(0, self._free_L_idx_in_lower_t, l_free)
        L_lower = torch.zeros(d, d, dtype=torch.float64, device=device)
        L_lower = L_lower.index_put(
            (self._lower_tri_r_t, self._lower_tri_c_t), l_lower_full
        )
        L = L_lower + L_lower.T  # symmetrize; diagonal stays zero

        return B, L

    def _pseudo_torch(self, B_t, omega_t):
        """Batched pseudo-variable computation via torch.linalg.solve.

        Parameters
        ----------
        B_t : torch.Tensor, shape (d, d)
        omega_t : torch.Tensor, shape (d, d)

        Returns
        -------
        Z_t : torch.Tensor, shape (d, n, d)
            Z_t[j] is the (n, d) pseudo-variable matrix for variable j,
            with column j zeroed out.
        """
        torch = self._torch
        device = self._device
        X_t = self._Xt
        n, d = X_t.shape

        eps_t = X_t - X_t @ B_t  # (n, d)

        # Build batched (d, d-1, d-1) omega submatrices
        omega_sub = omega_t[self._omega_row_idx, self._omega_col_idx]

        # Build batched (d, n, d-1) eps submatrices
        eps_sub = eps_t[:, self._sub_idx].permute(1, 0, 2)  # (d, n, d-1)

        # Batched solve: omega_sub[j] @ Z_sub[j].T = eps_sub[j].T
        Z_sub = torch.linalg.solve(
            omega_sub, eps_sub.permute(0, 2, 1)
        ).permute(0, 2, 1)  # (d, n, d-1)

        # Scatter Z_sub into zero-column output Z_t
        Z_t = torch.zeros(d, n, d, dtype=torch.float64, device=device)
        idx_expanded = self._insert_target.unsqueeze(1).expand(d, n, d - 1)
        Z_t.scatter_(2, idx_expanded, Z_sub)

        return Z_t  # (d, n, d)

    def _acyclicity_penalty_torch(self, W_t, K=None):
        """Smooth acyclicity surrogate (truncated series) in PyTorch.

        h(W) = sum_{k=1..K} trace((W*W)^k) / k!
        """
        torch = self._torch
        device = self._device
        d = W_t.shape[0]
        if K is None:
            K = self._acyc_order or d
        A = W_t * W_t
        Ak = torch.eye(d, dtype=torch.float64, device=device)
        acc = torch.zeros(1, dtype=torch.float64, device=device)
        for k in range(1, K + 1):
            Ak = Ak @ A
            acc = acc + torch.trace(Ak) / float(math.factorial(k))
        return acc

    @staticmethod
    def _bow_penalty_torch(W1_t, W2_t):
        """Bow-freeness surrogate in PyTorch."""
        A = W1_t * W2_t
        return (A * A).sum() / A.numel()

    def _objective_torch(self, b_free, l_free, rho, alpha, Z_t, penalty_fn_torch):
        """Augmented Lagrangian objective with PyTorch autograd.

        Parameters
        ----------
        b_free : torch.Tensor, shape (n_free_B,), requires_grad=True
        l_free : torch.Tensor, shape (n_free_L,), requires_grad=True
        rho : float
        alpha : float
        Z_t : torch.Tensor, shape (d, n, d)  -- treated as constant
        penalty_fn_torch : callable(B_t, L_t) -> scalar tensor
        """
        torch = self._torch
        X_t = self._Xt
        n = X_t.shape[0]

        B_t, L_t = self._unpack_torch(b_free, l_free)

        # Least-squares term (vectorised over j)
        ZL = torch.einsum('jnd,dj->nj', Z_t, L_t)  # (n, d)
        R = X_t - X_t @ B_t - ZL                    # (n, d)
        norms = torch.linalg.norm(R, dim=0)          # (d,)
        LS = 0.5 / n * (norms ** (2.0 * self._beta)).sum()

        # Structural penalty
        h = self._acyclicity_penalty_torch(B_t) + penalty_fn_torch(B_t, L_t)
        aug = 0.5 * rho * (h ** 2) + alpha * h

        # Smooth L0 regularization — numerically stable tanh form
        # tanh(s/2) == (e^s - 1)/(e^s + 1), avoids float64 overflow
        theta_free = torch.cat([b_free, l_free])
        s = math.log(float(n)) * torch.abs(theta_free)
        reg = self._lam * torch.sum(torch.tanh(s / 2.0))

        return LS + aug + reg

    def _lbfgs_torch_inner(
        self, b_free_np, l_free_np, rho, alpha, Z_t, penalty_fn_torch
    ):
        """Run torch.optim.LBFGS, replacing scipy L-BFGS-B.

        Parameters
        ----------
        b_free_np : numpy array, shape (n_free_B,)
        l_free_np : numpy array, shape (n_free_L,)

        Returns
        -------
        b_free_np : numpy array
        l_free_np : numpy array
        """
        torch = self._torch

        # Reuse pre-allocated GPU buffers to avoid CUDA malloc per iteration
        self._b_free_buf.copy_(torch.from_numpy(b_free_np))
        self._l_free_buf.copy_(torch.from_numpy(l_free_np))
        b_free = self._b_free_buf.clone().requires_grad_(True)
        l_free = self._l_free_buf.clone().requires_grad_(True)

        optimizer = torch.optim.LBFGS(
            [b_free, l_free],
            max_iter=100,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            history_size=100,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            loss = self._objective_torch(
                b_free, l_free, rho, alpha, Z_t, penalty_fn_torch
            )
            loss.backward()
            return loss

        optimizer.step(closure)

        return b_free.detach().cpu().numpy(), l_free.detach().cpu().numpy()


class ABICBootstrapResult(BootstrapResult):
    """The result of bootstrapping for Time series algorithm."""

    def __init__(
        self,
        adjacency_matrices,
        coefficient_matrices,
        error_covariance_matrices,
        total_effects,
        resampled_indices=None,
    ):
        """Construct a BootstrapResult.

        Parameters
        ----------
        adjacency_matrices : array-like, shape (n_sampling)
            The adjacency matrix list by bootstrapping.
        coefficient_matrices : array-like, shape (n_sampling)
            The coefficient matrix list by bootstrapping.
        error_covariance_matrices : array-like, shape (n_sampling)
            The error covariance matrix list by bootstrapping.
        total_effects : array-like, shape (n_sampling)
            The total effects list by bootstrapping.
        resampled_indices : list of array-like, shape (n_sampling), optional (default=None)
            The list of resampled indices used in bootstrapping.
        """
        super().__init__(
            adjacency_matrices, total_effects, resampled_indices=resampled_indices
        )
        self._coefficient_matrices = coefficient_matrices
        self._error_covariance_matrices = error_covariance_matrices

    @property
    def coefficient_matrices_(self):
        """The coefficient matrix list by bootstrapping.

        Returns
        -------
        coefficient_matrices_ : array-like, shape (n_sampling)
            The coefficient matrix list, where ``n_sampling`` is
            the number of bootstrap sampling.
        """
        return self._coefficient_matrices

    @property
    def error_covariance_matrices_(self):
        """The error covariance matrix list by bootstrapping.

        Returns
        -------
        error_covariance_matrices_ : array-like, shape (n_sampling)
            The error covariance matrix list, where ``n_sampling`` is
            the number of bootstrap sampling.
        """
        return self._error_covariance_matrices

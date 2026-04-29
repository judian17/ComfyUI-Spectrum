"""
ChebyshevForecaster + Spectrum predictor — ported from the original Spectrum paper.
https://github.com/hanjq17/Spectrum/blob/main/src/utils/basis_utils.py

Core idea: treat each feature channel of the denoiser as a function over time h(t),
approximate it with Chebyshev polynomials via ridge regression, and forecast features
at future timesteps to skip expensive transformer block evaluations.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


def _flatten(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Size]:
    shape = x.shape
    return x.reshape(1, -1) if x.ndim == 1 else x.reshape(1, -1), shape


def _unflatten(x_flat: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    return x_flat.reshape(shape)


class BaseForecaster(nn.Module):
    """Ridge-regression forecaster on a set of basis functions.

    Maintains a sliding window of (t, h) pairs and fits coefficients C
    solving:  Φ·C = H   via ridge regression  (ΦᵀΦ + λI)·C = ΦᵀH
    """

    def __init__(self, M: int = 4, K: int = 100, lam: float = 0.1,
                 device: Optional[torch.device] = None,
                 feature_shape: Optional[torch.Size] = None):
        super().__init__()
        assert K >= M + 2, "K should exceed basis size for stability"
        self.M = M
        self.K = K
        self.lam = lam
        self.register_buffer("t_buf", torch.empty(0))
        self._H_buf: Optional[torch.Tensor] = None
        self._shape: Optional[torch.Size] = None
        self._coef: Optional[torch.Tensor] = None
        self._XtX_fac: Optional[torch.Tensor] = None
        self._last_delta_norm: Optional[torch.Tensor] = None
        self.device_ref = device
        self.feature_shape = feature_shape

    def _taus(self, t: torch.Tensor) -> torch.Tensor:
        """Map normalized time t ∈ [0, 1] → τ ∈ [-1, 1] for Chebyshev domain."""
        return 2.0 * t - 1.0

    def _build_design(self, taus: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    def P(self) -> int:
        raise NotImplementedError

    # ---- core methods ----

    def update(self, t: float, h: torch.Tensor) -> None:
        """Append (t, h) to the buffer and invalidate cached coefficients."""
        device = self.device_ref or h.device
        t = torch.as_tensor(t, dtype=torch.float32, device=device)
        h_flat, shape = _flatten(h)
        h_flat = h_flat.to(device)

        if self._shape is None:
            self._shape = shape
        else:
            assert shape == self._shape, "Feature shape must remain constant"

        if self.t_buf.numel() == 0:
            self.t_buf = t[None]
            self._H_buf = h_flat
        else:
            delta = h_flat - self._H_buf[-1]
            self._last_delta_norm = delta.norm(p=2)
            self.t_buf = torch.cat([self.t_buf, t[None]], dim=0)
            self._H_buf = torch.cat([self._H_buf, h_flat], dim=0)
            if self.t_buf.numel() > self.K:
                self.t_buf = self.t_buf[-self.K:]
                self._H_buf = self._H_buf[-self.K:]

        self._coef = None
        self._XtX_fac = None

    def last_delta(self) -> torch.Tensor:
        if self._last_delta_norm is None:
            return torch.tensor(1e-6, device=self.t_buf.device)
        return self._last_delta_norm

    def ready(self) -> bool:
        return self.t_buf.numel() >= 2

    def _fit_if_needed(self) -> None:
        if self._coef is not None:
            return
        assert self.ready(), f"Need at least 2 points, got {self.t_buf.numel()}"
        taus = self._taus(self.t_buf)
        X = self._build_design(taus).to(torch.float32)    # (K, P)
        H = self._H_buf.to(torch.float32)                 # (K, F)
        K_, P = X.shape
        F = H.shape[1]
        assert P == self.P

        lamI = self.lam * torch.eye(P, device=X.device, dtype=X.dtype)
        Xt = X.transpose(0, 1)                             # (P, K)
        XtX = Xt @ X + lamI                                # (P, P)
        XtH = Xt @ H                                       # (P, F)

        try:
            L = torch.linalg.cholesky(XtX)
        except RuntimeError:
            jitter = 1e-6 * XtX.diag().mean()
            L = torch.linalg.cholesky(XtX + jitter * torch.eye(P, device=X.device))

        C = torch.cholesky_solve(XtH, L).to(torch.float32)
        self._coef = C
        self._XtX_fac = L

    @torch.no_grad()
    def predict(self, t_star: float) -> torch.Tensor:
        assert self._shape is not None, "No features cached yet"
        device = self.t_buf.device
        t_star = torch.as_tensor(t_star, dtype=torch.float32, device=device)
        self._fit_if_needed()

        tau_star = self._taus(t_star)
        x_star = self._build_design(tau_star[None])       # (1, P)
        h_flat = x_star @ self._coef                      # (1, F)
        return _unflatten(h_flat, self._shape)


class ChebyshevForecaster(BaseForecaster):
    """Chebyshev T-polynomials on τ ∈ [-1, 1]: T_0, T_1, ..., T_M via recurrence.

    Columns of design matrix: [T_0(τ), T_1(τ), ..., T_M(τ)] → P = M + 1
    Recurrence: T_0=1, T_1=τ, T_m = 2τ·T_{m-1} − T_{m-2}
    """

    def __init__(self, M: int = 4, K: int = 100, lam: float = 0.1,
                 device: Optional[torch.device] = None,
                 feature_shape: Optional[torch.Size] = None):
        super().__init__(M, K, lam, device, feature_shape)

    @property
    def P(self) -> int:
        return self.M + 1

    def _build_design(self, taus: torch.Tensor) -> torch.Tensor:
        taus = taus.reshape(-1, 1)
        K = taus.shape[0]
        T0 = torch.ones((K, 1), device=taus.device, dtype=taus.dtype)
        if self.M == 0:
            return T0
        T1 = taus
        cols = [T0, T1]
        for _ in range(2, self.M + 1):
            Tm = 2 * taus * cols[-1] - cols[-2]
            cols.append(Tm)
        return torch.cat(cols[:self.M + 1], dim=1)


class Spectrum(nn.Module):
    """Blends Chebyshev spectral prediction with local Taylor (Newton forward diffs).

    h_mix = (1 − w) * h_taylor + w * h_cheb

    The weight w can be dynamically adjusted — higher w favours the global Chebyshev
    predictor (better for long skips), lower w favours the local Taylor predictor
    (better for short skips with high-frequency detail).
    """

    def __init__(self, cheb: ChebyshevForecaster,
                 taylor_order: int = 1,
                 w: float = 0.5,
                 alpha: float = 6.0,
                 ema_beta: float = 0.9):
        super().__init__()
        assert taylor_order in (1, 2, 3)
        self.cheb = cheb
        self.taylor_order = taylor_order
        self.w = w
        self.alpha = alpha
        self.ema_beta = ema_beta

    @torch.no_grad()
    def _local_taylor_discrete(self, t_star: torch.Tensor) -> torch.Tensor:
        """Newton forward-difference interpolation (order 1-3)."""
        H = self.cheb._H_buf
        t = self.cheb.t_buf
        if t.numel() < 2:
            return _unflatten(H[-1:], self.cheb._shape)

        h_i = H[-1]
        t_i = t[-1]
        h_im1 = H[-2]
        t_im1 = t[-2]
        dt_last = (t_i - t_im1).clamp_min(1e-8)
        k = ((t_star - t_i) / dt_last).to(h_i.dtype)

        out = h_i + k * (h_i - h_im1)

        if self.taylor_order >= 2 and t.numel() >= 3:
            h_im2 = H[-3]
            d2 = h_i - 2 * h_im1 + h_im2
            out = out + 0.5 * k * (k - 1.0) * d2

        if self.taylor_order >= 3 and t.numel() >= 4:
            h_im3 = H[-4]
            d3 = h_i - 3 * h_im1 + 3 * h_im2 - h_im3
            out = out + (k * (k - 1.0) * (k - 2.0) / 6.0) * d3

        return _unflatten(out.unsqueeze(0), self.cheb._shape)

    @torch.no_grad()
    def predict(self, t_star: float, return_weight: bool = False):
        h_cheb = self.cheb.predict(t_star)
        h_taylor = self._local_taylor_discrete(t_star)
        w = self.w
        h_mix = (1 - w) * h_taylor + w * h_cheb
        if return_weight:
            return h_mix, float(w)
        return h_mix

    def update_w(self, new_w: float):
        self.w = new_w

    def update(self, t, h):
        return self.cheb.update(t, h)

    def last_delta(self):
        return self.cheb.last_delta()

    def ready(self):
        return self.cheb.ready()

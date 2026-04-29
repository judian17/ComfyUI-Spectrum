"""
SpectrumState — per-sampling-run state manager.

Owns the forecaster instances, counter logic, adaptive scheduling, and
dynamic Chebyshev/Taylor blend weight computation.
"""

from __future__ import annotations
import sys, os
_spec_dir = os.path.dirname(os.path.abspath(__file__))
if _spec_dir not in sys.path:
    sys.path.insert(0, _spec_dir)

import torch
from typing import Optional
from forecaster import Spectrum, ChebyshevForecaster


class SpectrumState:
    """All mutable state for one sampling run.

    Placed in model_options["transformer_options"]["spectrum"] so every
    block patch closure can access it without global state.
    """

    def __init__(
        self,
        w: float = 0.5,
        M: int = 4,
        lam: float = 0.1,
        warmup_steps: int = 3,
        window_size: float = 2.0,
        flex_window: float = 0.75,
        max_ws: float = 8.0,
        min_ws: float = 1.0,
        max_w: float = 0.9,
    ):
        # Hyper-parameters
        self.w = w
        self.M = M
        self.lam = lam
        self.warmup_steps = warmup_steps
        self.window_size = window_size
        self.flex_window = flex_window
        self.max_ws = max_ws
        self.min_ws = min_ws
        self.max_w = max_w

        # Step counters
        self.num_steps: int = 0
        self.cnt: int = 0
        self.num_consecutive_cached_steps: int = 0
        self.curr_ws: float = window_size

        # Per-step decision
        self.actual_forward: bool = True

        # Forecasters: key = cond_or_uncond index (0=cond, 1=uncond)
        self.forecasters: dict[int, Spectrum] = {}
        self.feature_shape: Optional[torch.Size] = None
        self.txt_len: int = 0

        # Model info
        self.model_type: str = "unknown"

        # Accumulated count of actual forwards (for reporting)
        self.total_actual_forwards: int = 0
        self.total_skipped: int = 0

        # Safety: set when run completes, cleared on reset. Blocks all Spectrum
        # activity when True (prevents cross-contamination between flows).
        self.finished: bool = False

        # Debug flag (set True to see per-step logs)
        self.verbose: bool = False


    # ---- scheduling ----

    def stop(self):
        """Mark this sampling run as finished and disable Spectrum logic."""
        self.finished = True
        if self.verbose:
            self._report_stats()

    def should_actual_forward(self) -> bool:
        """Decide whether this step runs the full transformer or reuses cached features."""
        if self.finished:
            self.actual_forward = True
            return True
        if self.cnt < self.warmup_steps:
            self.actual_forward = True
        elif not self._any_forecaster_ready():
            # Not enough cached points for prediction — force forward
            self.actual_forward = True
        else:
            ws = max(1, int(self.curr_ws))
            self.actual_forward = (self.num_consecutive_cached_steps + 1) % ws == 0
            if self.actual_forward:
                self.curr_ws += self.flex_window
                self.curr_ws = round(self.curr_ws, 3)
        if self.verbose:
            tag = "FWD" if self.actual_forward else "SKIP"
            print(f"[Spectrum] step {self.cnt}/{self.num_steps} [{tag}] "
                  f"ws={self.curr_ws:.2f} consec={self.num_consecutive_cached_steps}")
        return self.actual_forward

    def _any_forecaster_ready(self) -> bool:
        if not self.forecasters:
            return False
        return any(fc.ready() for fc in self.forecasters.values())

    def advance_step(self):
        """Update counters after each step."""
        if self.finished:
            return
        if self.actual_forward:
            self.num_consecutive_cached_steps = 0
            self.total_actual_forwards += 1
        else:
            self.num_consecutive_cached_steps += 1
            self.total_skipped += 1
        self.cnt += 1
        if self.cnt >= self.num_steps and self.num_steps > 0:
            self._report_stats()
            self.stop()

    def _report_stats(self):
        print(f"[Spectrum] ======== SAMPLING COMPLETE ========")
        print(f"[Spectrum] Total steps: {self.num_steps}")
        print(f"[Spectrum] Actual forwards: {self.total_actual_forwards}")
        print(f"[Spectrum] Skipped (predicted): {self.total_skipped}")
        print(f"[Spectrum] Speedup from block skip: "
              f"{self.num_steps / max(self.total_actual_forwards, 1):.1f}x")
        print(f"[Spectrum] Forecasters used: {len(self.forecasters)}")
        print(f"[Spectrum] Feature shape: {self.feature_shape}")
        print(f"[Spectrum] Model type: {self.model_type}")
        print(f"[Spectrum] ======================================")

    def compute_dynamic_w(self) -> float:
        """Map remaining window size to Chebyshev blend weight.

        Larger window → trust Chebyshev more (bigger w).
        Smaller window → trust Taylor more (smaller w).
        """
        remaining = min(self.curr_ws, self.num_steps - max(self.cnt, 1))
        remaining = int(remaining)
        if self.max_ws <= self.min_ws:
            return self.w
        ratio = (remaining - self.min_ws) / (self.max_ws - self.min_ws)
        return max(0.0, min(self.max_w, ratio * self.max_w))

    # ---- forecaster management ----

    def get_or_create_forecaster(self, cond_idx: int, device: torch.device) -> Spectrum:
        """Lazily create (or retrieve) the forecaster for a cond/uncond path."""
        if cond_idx not in self.forecasters:
            cheb = ChebyshevForecaster(
                M=self.M, K=100, lam=self.lam,
                device=device, feature_shape=self.feature_shape,
            )
            self.forecasters[cond_idx] = Spectrum(
                cheb, taylor_order=1, w=self.w,
            )
        return self.forecasters[cond_idx]

    def cache_features(self, features: torch.Tensor, cond_indices: list[int]):
        if self.finished:
            return
        B = features.shape[0]
        n_paths = len(cond_indices)
        batch_per = B // n_paths

        if self.feature_shape is None:
            self.feature_shape = features.shape[1:]
            if self.verbose:
                print(f"[Spectrum] FIRST CACHE: shape={self.feature_shape}, "
                      f"features mean={features.mean().item():.4f} std={features.std().item():.4f}")

        t = self.cnt / max(self.num_steps, 1)

        for i, idx in enumerate(cond_indices):
            f = features[i * batch_per:(i + 1) * batch_per]
            forecaster = self.get_or_create_forecaster(idx, features.device)
            forecaster.update(t, f.reshape(-1)[0] if f.numel() == 1 else f.reshape(-1))

        if self.verbose:
            print(f"[Spectrum] CACHE step={self.cnt} t={t:.3f} "
                  f"shape={features.shape[1:]} cond_idx={cond_indices}")


    def predict_features(self, cond_indices: list[int],
                         device: torch.device,
                         batch_per: int) -> torch.Tensor:
        if self.finished:
            raise RuntimeError("[Spectrum] predict_features called while finished")
        t = self.cnt / max(self.num_steps, 1)

        new_w = self.compute_dynamic_w()
        for idx in cond_indices:
            fc = self.get_or_create_forecaster(idx, device)
            fc.update_w(new_w)

        chunks = []
        for idx in cond_indices:
            fc = self.get_or_create_forecaster(idx, device)
            pred = fc.predict(t)
            pred = pred.reshape(batch_per, *self.feature_shape)
            chunks.append(pred)

        result = torch.cat(chunks, dim=0)

        if self.verbose:
            print(f"[Spectrum] PREDICT step={self.cnt} t={t:.3f} w={new_w:.2f} "
                  f"shape={result.shape} "
                  f"pred mean={result.mean().item():.4f} std={result.std().item():.4f}")

        return result

    def reset(self):
        """Reset all per-sampling-run state."""
        self.cnt = 0
        self.num_steps = 0
        self.num_consecutive_cached_steps = 0
        self.curr_ws = self.window_size
        self.actual_forward = True
        self.finished = False
        self.forecasters.clear()
        self.feature_shape = None
        self.txt_len = 0
        self.total_actual_forwards = 0

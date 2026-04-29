"""
Spectrum v1.0 — Training-free diffusion acceleration via Chebyshev polynomial
feature forecasting.  CVPR 2026, Han et al.  https://arxiv.org/abs/2603.01623

Supported models (auto-detected):
  Flux-like    — Flux.1, Klein 9b, Longcat, Chroma  (double_blocks + single_blocks)
  MMDiT        — Qwen Image, SD3.5, Chroma           (transformer_blocks)
  Z Image      — Z Image Turbo                       (layers, noise_refiner)
  ErnieImage   — Ernie                               (layers, forward patched)
"""

from __future__ import annotations
import sys, os
_spec_dir = os.path.dirname(os.path.abspath(__file__))
if _spec_dir not in sys.path:
    sys.path.insert(0, _spec_dir)

from comfy_api.latest import io
from spectrum_state import SpectrumState
from spectrum_patches import detect_and_register, make_predict_noise_wrapper
import comfy.patcher_extension


def _make_reset_wrapper(state: SpectrumState):
    def wrapper(executor, *args, **kwargs):
        state.reset()
        return executor(*args, **kwargs)
    return wrapper


class SpectrumNode(io.ComfyNode):

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Spectrum",
            display_name="Spectrum v1.0 (Diffusion Acceleration)",
            description=(
                "Training-free diffusion acceleration via spectral feature forecasting. "
                "Supports: Flux, Klein, Longcat, Qwen Image, SD3, Z Image, Ernie. "
                "Default params (w=0.5, window=2, flex=0.75) give ~2-3x speedup."
            ),
            category="model_patches",
            inputs=[
                io.Model.Input("model",
                               tooltip="DiT diffusion model."),
                io.Float.Input("w", default=0.5, min=0.0, max=1.0, step=0.05,
                               tooltip="Chebyshev/Taylor blend weight. 0=pure Taylor (local), "
                                       "1=pure Chebyshev (global). 0.5 recommended."),
                io.Int.Input("M", default=4, min=1, max=10,
                             tooltip="Chebyshev polynomial degree. 2=rough, 4=sweet spot, 6+ diminishing returns."),
                io.Float.Input("lam", default=0.1, min=0.001, max=10.0, step=0.01,
                               tooltip="Ridge regularization λ. Too small→unstable, too large→underfit. 0.1 is optimal."),
                io.Int.Input("warmup_steps", default=3, min=0, max=20,
                             tooltip="Full-precision steps before acceleration begins. "
                                     "Builds initial cache for reliable predictions."),
                io.Float.Input("window_size", default=2.0, min=1.0, max=16.0, step=0.5,
                               tooltip="Initial skip interval N. 1=no acceleration, 2=every-other-step, "
                                       "higher=fewer actual forwards."),
                io.Float.Input("flex_window", default=0.75, min=0.0, max=4.0, step=0.01,
                               tooltip="Acceleration ramp α. 0=fixed interval, 0.75=gradual (recommended), "
                                       "3.0=aggressive speedup. Larger values skip more later steps."),
                io.Float.Input("max_w", default=0.8, min=0.0, max=1.0, step=0.05,
                               tooltip="Upper bound on dynamic Chebyshev weight. "
                                       "0.8 works for most cases; raise to 0.9 for extreme speedups."),
                io.Boolean.Input("verbose", default=False,
                                 tooltip="Enable per-step logging + feature-drift diagnostics."),
            ],
            outputs=[io.Model.Output(tooltip="Model with Spectrum acceleration.")],
        )

    @classmethod
    def execute(cls, model, w, M, lam, warmup_steps,
                window_size, flex_window, max_w, verbose) -> io.NodeOutput:

        model = model.clone()

        state = SpectrumState(
            w=w, M=M, lam=lam,
            warmup_steps=warmup_steps,
            window_size=window_size, flex_window=flex_window,
            max_w=max_w,
            max_ws=window_size + flex_window * 20, min_ws=1.0,
        )
        state.verbose = verbose

        model.model_options.setdefault("transformer_options", {})["spectrum"] = state

        model_type = detect_and_register(model, state)

        if model_type == "unsupported":
            print("[Spectrum] Unsupported model — acceleration disabled, "
                  "sampling proceeds normally.")
            return io.NodeOutput(model)

        model.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.PREDICT_NOISE,
            "spectrum",
            make_predict_noise_wrapper(state),
        )
        # Reset state at the start of every sampling run
        model.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
            "spectrum_reset",
            _make_reset_wrapper(state),
        )
        return io.NodeOutput(model)


NODE_CLASS_MAPPINGS = {"Spectrum": SpectrumNode}
NODE_DISPLAY_NAME_MAPPINGS = {"Spectrum": "Spectrum v1.0 (Acceleration)"}

# ComfyUI-Spectrum v1.0
# Training-free diffusion acceleration via spectral feature forecasting.
# https://arxiv.org/abs/2603.01623

import sys, os
_sd = os.path.dirname(os.path.abspath(__file__))
if _sd not in sys.path:
    sys.path.insert(0, _sd)
from spectrum_node import SpectrumNode

NODE_CLASS_MAPPINGS = {"Spectrum": SpectrumNode}
NODE_DISPLAY_NAME_MAPPINGS = {"Spectrum": "Spectrum v1.0 (Acceleration)"}

"""
Z Image / Lumina2 (NextDiT) support for Spectrum.

Z Image uses the older patches["double_block"] post-hook mechanism,
NOT patches_replace["dit"]. So we cannot use set_model_patch_replace.
Instead we wrap each JointTransformerBlock in dm.layers with a
predict/cache wrapper, and patch patchify_and_embed to expose cap_size.

Supports: NextDiT (latent-space) and NextDiTPixelSpace (pixel-space).
"""

from __future__ import annotations
import types
import torch
import torch.nn as nn
from spectrum_state import SpectrumState


class _ZImageLayerWrapper(nn.Module):
    """Wraps one JointTransformerBlock for Spectrum acceleration.

    - first layer + skip:  predict features, inject into img portion, return
    - other layers + skip: identity pass-through
    - last layer + actual: cache img portion of output
    - all layers + actual: run original layer
    """

    def __init__(self, original_layer: nn.Module, state: SpectrumState,
                 is_first: bool, is_last: bool):
        super().__init__()
        self.original_layer = original_layer
        self.state = state
        self.is_first = is_first
        self.is_last = is_last

    def forward(self, x: torch.Tensor, x_mask, freqs_cis,
                adaln_input=None, timestep_zero_index=None,
                transformer_options=None):
        if transformer_options is None:
            transformer_options = {}

        if self.state.finished:
            return self.original_layer(
                x, x_mask, freqs_cis, adaln_input,
                timestep_zero_index, transformer_options,
            )

        if not self.state.actual_forward:
            # ---- SKIP ----
            if self.is_first:
                cap_size = transformer_options.get('_spectrum_cap_size', [0])
                cond_or_uncond = transformer_options.get('cond_or_uncond', [0])
                batch_size = x.shape[0]
                bp = batch_size // max(len(cond_or_uncond), 1)

                pred_img = self.state.predict_features(
                    cond_indices=cond_or_uncond,
                    device=x.device,
                    batch_per=bp,
                )
                x_out = x.clone()
                x_out[:, cap_size[0]:] = pred_img.to(
                    device=x.device, dtype=x.dtype)
                return x_out
            return x  # identity for non-first layers

        # ---- ACTUAL FORWARD ----
        result = self.original_layer(
            x, x_mask, freqs_cis, adaln_input,
            timestep_zero_index, transformer_options,
        )

        if self.is_last:
            cap_size = transformer_options.get('_spectrum_cap_size', [0])
            cond_or_uncond = transformer_options.get('cond_or_uncond', [0])
            img_features = result[:, cap_size[0]:].clone()
            self.state.cache_features(
                features=img_features,
                cond_indices=cond_or_uncond,
            )

        return result


def patch_zimage(dm, state: SpectrumState) -> int:
    """Wrap every layer in dm.layers and patch patchify_and_embed.

    Returns the number of layers wrapped.

    Guard against re-wrapping: if dm.layers are already _ZImageLayerWrapper
    instances from a previous execute() call, restore the originals first.
    Otherwise nested wrapping traps old SpectrumState objects (and their
    forecasters' GPU tensors) inside stale inner wrappers.
    """

    n = len(dm.layers)

    # If already patched from a previous run, restore the originals.
    # dm.layers is a nn.ModuleList; we must replace items one-by-one because
    # assigning a plain list to it fails PyTorch's __setattr__ type check.
    if hasattr(dm, '_spectrum_original_layers'):
        for i, orig in enumerate(dm._spectrum_original_layers):
            dm.layers[i] = orig
        if hasattr(dm, '_spectrum_original_patchify'):
            dm.patchify_and_embed = dm._spectrum_original_patchify

    # Save originals so we can restore them later if needed
    dm._spectrum_original_layers = list(dm.layers)
    dm._spectrum_original_patchify = dm.patchify_and_embed

    # ---- patch patchify_and_embed to stash cap_size in transformer_options ----
    original_patchify = dm.patchify_and_embed

    def _patched_patchify(self, x, cap_feats, cap_mask, t, num_tokens,
                          ref_latents=[], ref_contexts=[], siglip_feats=[],
                          transformer_options={}):
        result = original_patchify(
            x, cap_feats, cap_mask, t, num_tokens,
            ref_latents, ref_contexts, siglip_feats,
            transformer_options,
        )
        # result[3] is the cap_size list (image token start index per sample)
        transformer_options['_spectrum_cap_size'] = result[3]
        return result

    dm.patchify_and_embed = types.MethodType(_patched_patchify, dm)

    # ---- replace each layer ----
    for i in range(n):
        dm.layers[i] = _ZImageLayerWrapper(
            dm._spectrum_original_layers[i],
            state,
            is_first=(i == 0),
            is_last=(i == n - 1),
        )

    return n

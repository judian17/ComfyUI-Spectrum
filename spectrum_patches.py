"""
Block replacement factories for Spectrum.

Uses ComfyUI's patches_replace["dit"] mechanism to intercept every transformer
block. For models without patches_replace (Ernie, Z Image), directly patches layers/forward.
"""

from __future__ import annotations
import sys, os
_spec_dir = os.path.dirname(os.path.abspath(__file__))
if _spec_dir not in sys.path:
    sys.path.insert(0, _spec_dir)

import torch
from spectrum_state import SpectrumState


# ======================================================================
#  Unified block-wrapper factory  (Flux / MMDiT with patches_replace)
# ======================================================================

def make_block_patch(is_last: bool, is_first: bool, block_type: str,
                     state: SpectrumState):
    _first_call = [True]

    def handler(args: dict, extra: dict):
        transformer_options = args.get("transformer_options", {})
        cond_or_uncond = transformer_options.get("cond_or_uncond", [0])

        # If state is finished (previous run ended), just pass through.
        # This prevents cross-contamination between workflow branches.
        if state.finished:
            original_block = extra.get("original_block")
            if original_block is not None:
                return original_block(args)
            if block_type in ("flux_double", "mmdit"):
                return {"img": args["img"], "txt": args.get("txt", args["img"])}
            return {"img": args["img"]}

        if _first_call[0] and state.verbose:
            _first_call[0] = False
            print(f"[Spectrum] BLOCK PATCH: type={block_type} "
                  f"first={is_first} last={is_last} step={state.cnt}")

        # ---- SKIP ----
        if not state.actual_forward:
            if is_first:
                pred_feat = state.predict_features(
                    cond_indices=cond_or_uncond,
                    device=args["img"].device,
                    batch_per=args["img"].shape[0] // max(len(cond_or_uncond), 1),
                )
                img = args["img"].clone()
                bp = img.shape[0] // max(len(cond_or_uncond), 1)
                for i, idx in enumerate(cond_or_uncond):
                    img[i * bp:(i + 1) * bp] = pred_feat[i * bp:(i + 1) * bp].to(
                        device=img.device, dtype=img.dtype)
                args["img"] = img

            if block_type in ("flux_double", "mmdit"):
                return {"img": args["img"], "txt": args.get("txt", args["img"])}
            return {"img": args["img"]}

        # ---- ACTUAL FORWARD ----
        original_block = extra.get("original_block")
        if original_block is None:
            if block_type in ("flux_double", "mmdit"):
                return {"img": args["img"], "txt": args.get("txt", args["img"])}
            return {"img": args["img"]}

        out = original_block(args)

        if is_last:
            if block_type == "flux_single":
                img_slice = transformer_options.get("img_slice", None)
                txt_len = img_slice[0] if img_slice else 0
                img_features = out["img"][:, txt_len:, :].clone() if txt_len else out["img"].clone()
            elif block_type == "mmdit":
                img_features = out["img"].clone()
            else:
                img_features = out["img"].clone()
            state.cache_features(features=img_features, cond_indices=cond_or_uncond)

        return out

    return handler


# ======================================================================
#  Model registration — patches_replace path
# ======================================================================

def register_flux_patches(model_patcher, state: SpectrumState) -> int:
    dm = model_patcher.model.diffusion_model
    n_d, n_s = len(dm.double_blocks), len(dm.single_blocks)
    for i in range(n_d):
        model_patcher.set_model_patch_replace(
            make_block_patch(False, i == 0, "flux_double", state), "dit", "double_block", i)
    for i in range(n_s):
        model_patcher.set_model_patch_replace(
            make_block_patch(i == n_s - 1, False, "flux_single", state), "dit", "single_block", i)
    return n_d + n_s


def register_mmdit_patches(model_patcher, state: SpectrumState) -> int:
    dm = model_patcher.model.diffusion_model
    blocks = getattr(dm, 'transformer_blocks', None) or dm.joint_blocks
    n = len(blocks)
    for i in range(n):
        model_patcher.set_model_patch_replace(
            make_block_patch(i == n - 1, i == 0, "mmdit", state), "dit", "double_block", i)
    return n


# ======================================================================
#  PREDICT_NOISE wrapper — drives the scheduling state machine
# ======================================================================

def make_predict_noise_wrapper(state: SpectrumState):
    def wrapper(executor, x, timestep, model_options=None, seed=None):
        if model_options is None:
            model_options = {}
        to = model_options.get("transformer_options", {})

        # Safety: only act if spectrum state is still attached to this model.
        # Prevents cross-contamination when ComfyUI shares model internals
        # between Spectrum and non-Spectrum workflow branches.
        if to.get("spectrum") is not state:
            return executor(x, timestep, model_options, seed)

        sigmas = to.get("sample_sigmas")
        if sigmas is not None and len(sigmas) > 1:
            new_n = len(sigmas) - 1
            if state.num_steps == 0 or state.cnt >= state.num_steps:
                state.reset()
                state.num_steps = new_n

        state.should_actual_forward()
        result = executor(x, timestep, model_options, seed)
        state.advance_step()
        return result

    return wrapper


# ======================================================================
#  ErnieImage — direct forward() patching  (no WrapperExecutor)
# ======================================================================

def _patch_ernie_forward(dm, state: SpectrumState):
    # Guard against re-patching: if we've already wrapped dm.forward in a
    # previous execute() call, restore the real original first.  Otherwise
    # each run creates a nested closure chain: patched_v2(patched_v1(orig)),
    # which traps every old SpectrumState + its multi-GB _H_buf tensors alive
    # until the model is destroyed.
    if hasattr(dm, '_spectrum_original_forward'):
        dm.forward = dm._spectrum_original_forward
        if hasattr(dm, '_spectrum_handle'):
            dm._spectrum_handle.remove()
            del dm._spectrum_handle

    original_forward = dm.forward
    dm._spectrum_original_forward = original_forward
    cache_buf = []

    def hook(module, inp, out):
        cache_buf.append(out.detach().clone())

    def patched_forward(x, timesteps, context, **kwargs):
        to = kwargs.get("transformer_options", {})
        if state.finished:
            return original_forward(x, timesteps, context, **kwargs)

        if not state.actual_forward:
            return _ernie_skip_forward(dm, x, timesteps, context, state, to)

        if not hasattr(patched_forward, '_handle'):
            patched_forward._handle = dm.layers[-1].register_forward_hook(hook)
            dm._spectrum_handle = patched_forward._handle
            if state.verbose:
                print(f"[Spectrum] Ernie hook on {len(dm.layers)} layers")

        result = original_forward(x, timesteps, context, **kwargs)

        if cache_buf and state.actual_forward:
            hidden = cache_buf.pop(0)
            cache_buf.clear()
            p = dm.patch_size
            N_img = (x.shape[2] // p) * (x.shape[3] // p)
            img_feat = hidden[:, :N_img, :].clone()
            cond = to.get("cond_or_uncond", [0])
            state.cache_features(features=img_feat, cond_indices=cond)

        return result

    dm.forward = patched_forward


def _ernie_skip_forward(model, x, timesteps, context,
                        state, transformer_options):
    B, C, H, W = x.shape
    p = model.patch_size
    Hp, Wp = H // p, W // p
    N_img = Hp * Wp
    cond_or_uncond = transformer_options.get("cond_or_uncond", [0])
    bp = B // max(len(cond_or_uncond), 1)

    pred_img = state.predict_features(cond_indices=cond_or_uncond,
                                       device=x.device, batch_per=bp)
    pred_img = pred_img.reshape(B, N_img, -1).to(device=x.device, dtype=x.dtype)

    text_bth = model.text_proj(context) if (model.text_proj is not None and
                                            context.numel() > 0) else context
    hidden = torch.cat([pred_img, text_bth], dim=1)

    c = model.time_embedding(model.time_proj(timesteps).to(x.dtype))
    hidden = model.final_norm(hidden, c).type_as(hidden)
    patches = model.final_linear(hidden)[:, :N_img, :]

    return (patches.view(B, Hp, Wp, p, p, model.out_channels)
            .permute(0, 5, 1, 3, 2, 4).contiguous()
            .view(B, model.out_channels, H, W))


# ======================================================================
#  Model detection
# ======================================================================

SUPPORTED_MODELS = []

def detect_and_register(model_patcher, state: SpectrumState) -> str:
    """Detect model architecture and register appropriate Spectrum hooks.
    Returns model_type string. Unsupported models fall back to normal sampling.
    """
    dm = model_patcher.model.diffusion_model

    # Flux / Klein / Longcat: double_blocks + non-empty single_blocks
    if (hasattr(dm, 'double_blocks') and hasattr(dm, 'single_blocks')
            and len(dm.single_blocks) > 0):
        n = register_flux_patches(model_patcher, state)
        state.model_type = "flux"
        SUPPORTED_MODELS.append(
            f"Flux-like ({len(dm.double_blocks)}d+{len(dm.single_blocks)}s, {n} blocks)")
        print(f"[Spectrum] ✓ Flux-like: {len(dm.double_blocks)} double + "
              f"{len(dm.single_blocks)} single blocks ({n} patched)")
        return "flux"

    # MMDiT (Qwen Image, SD3, Chroma, etc.)
    if hasattr(dm, 'transformer_blocks') or hasattr(dm, 'joint_blocks'):
        n = register_mmdit_patches(model_patcher, state)
        state.model_type = "qwen_image"
        SUPPORTED_MODELS.append(f"MMDiT ({n} blocks)")
        print(f"[Spectrum] ✓ MMDiT: {n} double blocks patched")
        return "qwen_image"

    # HunyuanVideo 1.5 / OmniWeaving: double_blocks, (single_blocks missing or empty)
    if (hasattr(dm, 'double_blocks')
            and (not hasattr(dm, 'single_blocks') or len(dm.single_blocks) == 0)):
        n = register_double_only_patches(model_patcher, state, 'double_blocks')
        state.model_type = "hunyuan"
        SUPPORTED_MODELS.append(f"HunyuanVideo 1.5 ({n} double blocks)")
        print(f"[Spectrum] ✓ HunyuanVideo 1.5: {n} double blocks patched")
        return "hunyuan"

    # Wan 2.x: blocks + head (unified stream, cross-attention to context)
    if hasattr(dm, 'blocks') and hasattr(dm, 'head'):
        n = register_double_only_patches(model_patcher, state, 'blocks')
        state.model_type = "wan"
        SUPPORTED_MODELS.append(f"Wan ({n} blocks)")
        print(f"[Spectrum] ✓ Wan: {n} blocks patched")
        return "wan"

    # Z Image / Lumina2 (NextDiT): noise_refiner + cap_embedder + layers
    # Uses patches["double_block"] (post-hook), not patches_replace.
    if hasattr(dm, 'noise_refiner') and hasattr(dm, 'cap_embedder') and hasattr(dm, 'layers'):
        from spectrum_patches_zimage import patch_zimage
        n = patch_zimage(dm, state)
        state.model_type = "zimage"
        SUPPORTED_MODELS.append(f"Z Image/Lumina2 ({n} layers)")
        print(f"[Spectrum] ✓ Z Image / Lumina2: {n} layers wrapped")
        return "zimage"

    # ErnieImage: layers + final_norm + text_proj (no patches_replace)
    if hasattr(dm, 'layers') and hasattr(dm, 'final_norm') and hasattr(dm, 'text_proj'):
        state.model_type = "ernie"
        _patch_ernie_forward(dm, state)
        SUPPORTED_MODELS.append(f"ErnieImage ({len(dm.layers)} layers)")
        print(f"[Spectrum] ✓ ErnieImage: {len(dm.layers)} layers (forward patched)")
        return "ernie"

    # Unsupported — print model attributes to help diagnose
    state.model_type = "unsupported"
    # List likely block-related attributes for debugging
    clues = [a for a in dir(dm)
             if ('block' in a.lower() or 'layer' in a.lower())
             and not a.startswith('_')]
    print(f"[Spectrum] Unsupported model. dm type={type(dm).__name__}, "
          f"has={clues}")
    return "unsupported"


def register_double_only_patches(model_patcher, state: SpectrumState, block_attr: str = 'double_blocks') -> int:
    """Register patches for models with only double blocks (no single blocks).
    Works for HunyuanVideo 1.5, Wan, and similar.
    """
    dm = model_patcher.model.diffusion_model
    blocks = getattr(dm, block_attr)
    n = len(blocks)
    for i in range(n):
        model_patcher.set_model_patch_replace(
            make_block_patch(i == n - 1, i == 0, "mmdit", state),
            "dit", "double_block", i)
    return n

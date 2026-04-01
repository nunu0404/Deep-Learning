#!/usr/bin/env python3
"""GUI-Aware Pruning (GAP) for Qwen2-VL-style vision encoders."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from scipy.stats import entropy as scipy_entropy
from tqdm import tqdm

# Avoid Xet-based partial fetch issues on some HF mirrors.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

try:
    from evaluation.evaluate_baseline import (
        BaseBackend,
        PROMPT,
        PredictionResult,
        build_test_split,
        compute_metrics,
        load_metadata,
        percentile,
        safe_mean,
        run_inference_with_retry,
    )
except ImportError:  # pragma: no cover - package-relative execution fallback
    from ..evaluation.evaluate_baseline import (
        BaseBackend,
        PROMPT,
        PredictionResult,
        build_test_split,
        compute_metrics,
        load_metadata,
        percentile,
        safe_mean,
        run_inference_with_retry,
    )


DEFAULT_DROP_RATES = [round(value, 1) for value in np.arange(0.0, 1.0, 0.1)]
DEFAULT_MODEL_SPECS = {
    "qwen2vl": {
        "model_id": "Qwen/Qwen2-VL-7B-Instruct",
        "loader": "qwen2vl",
    },
}


def setup_gap_logging(results_dir: Path) -> logging.Logger:
    results_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("gap_pruning")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(results_dir / "errors.log", encoding="utf-8")
    file_handler.setLevel(logging.ERROR)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def normalize_zero_one(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32)
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if math.isclose(minimum, maximum):
        return np.zeros_like(values, dtype=np.float32)
    return ((values - minimum) / (maximum - minimum)).astype(np.float32)


def align_signal(values: np.ndarray, target_length: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if target_length <= 0:
        return np.zeros((0,), dtype=np.float32)
    if values.size == 0:
        return np.zeros((target_length,), dtype=np.float32)
    if values.size == target_length:
        return values.astype(np.float32)
    if target_length == 1:
        return np.array([float(np.mean(values))], dtype=np.float32)

    source = np.linspace(0.0, 1.0, num=values.size, dtype=np.float32)
    target = np.linspace(0.0, 1.0, num=target_length, dtype=np.float32)
    return np.interp(target, source, values).astype(np.float32)


def sanitize_tag(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def format_drop_rate(drop_rate: float) -> str:
    return f"{drop_rate:.1f}"


def parse_drop_rates(text: str) -> list[float]:
    if not text.strip():
        return DEFAULT_DROP_RATES.copy()

    values: list[float] = []
    for raw in text.split(","):
        candidate = float(raw.strip())
        if not 0.0 <= candidate <= 0.9:
            raise ValueError(f"Drop rate must be between 0.0 and 0.9. Received {candidate}.")
        values.append(round(candidate, 4))
    if not values:
        raise ValueError("At least one drop rate is required.")
    return values


def extract_grid_thw(grid_thw: Any) -> tuple[int, int, int] | None:
    if grid_thw is None:
        return None
    if hasattr(grid_thw, "detach"):
        grid_thw = grid_thw.detach().cpu().numpy()
    array = np.asarray(grid_thw)
    if array.size < 3:
        return None
    flat = array.reshape(-1)
    return int(flat[-3]), int(flat[-2]), int(flat[-1])


def get_nested_attr(root: Any, dotted_path: str) -> Any | None:
    current = root
    for part in dotted_path.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


class GAPPruner:
    def __init__(self, model, drop_rate: float = 0.5, alpha: float = 0.4, beta: float = 0.3, gamma: float = 0.3):
        """
        model: a HuggingFace VLM with a ViT vision encoder
        drop_rate: fraction of tokens to drop (0.0 to 0.9)
        alpha: weight for attention score signal
        beta: weight for (1 - color_entropy) signal
        gamma: weight for edge_density signal
        """
        if not 0.0 <= drop_rate <= 0.9:
            raise ValueError(f"drop_rate must be between 0.0 and 0.9. Received {drop_rate}.")

        self.model = model
        self.drop_rate = float(drop_rate)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.logger = logging.getLogger("gap_pruning")

        self.vision_model, self.vision_layers, self.vision_style = self._locate_vision_stack(model)
        self.vision_config = self._get_vision_config(model)
        self.patch_size = int(getattr(self.vision_config, "patch_size", 14))
        self.spatial_merge_size = int(
            getattr(self.vision_model, "spatial_merge_size", None)
            or getattr(self.vision_config, "merge_size", None)
            or getattr(self.vision_config, "spatial_merge_size", None)
            or 1
        )

        self.current_image: Image.Image | None = None
        self.current_grid_thw: tuple[int, int, int] | None = None
        self.cls_attention: np.ndarray | None = None
        self.last_pruning_info: dict[str, Any] = {}
        self._runtime_state: dict[str, Any] = {}
        self._hook_handles: list[Any] = []
        self._warned_edge_fallback = False
        self._used_uniform_attention = False

    def _locate_vision_stack(self, model: Any) -> tuple[Any, Any, str]:
        candidates = [
            ("visual", "blocks", "qwen2_flat"),
            ("vision_tower.vision_model", "encoder.layers", "generic_cls"),
            ("vision_model", "encoder.layers", "generic_cls"),
        ]
        for root_path, layers_path, style in candidates:
            vision_root = get_nested_attr(model, root_path)
            layers = get_nested_attr(vision_root, layers_path) if vision_root is not None else None
            if vision_root is not None and layers is not None and len(layers) > 2:
                return vision_root, layers, style
        raise ValueError("Unable to locate a supported vision encoder on the provided model.")

    def _get_vision_config(self, model: Any) -> Any:
        for attr_name in ("vision_config", "visual_config"):
            config = getattr(getattr(model, "config", None), attr_name, None)
            if config is not None:
                return config
        raise ValueError("Model config is missing vision_config.")

    def set_current_image(self, image: Image.Image, grid_thw: Any | None = None) -> None:
        self.current_image = image.convert("RGB")
        self.current_grid_thw = extract_grid_thw(grid_thw)
        self.cls_attention = None
        self.last_pruning_info = {}
        self._runtime_state = {}
        self._used_uniform_attention = False

    def compute_patch_signals(self, image: Image.Image) -> dict:
        """
        Given a PIL image, compute per-patch signals.
        Patch size = model.config.vision_config.patch_size (e.g. 14)
        Returns dict with keys: 'entropy', 'edge_density'
        Both are numpy arrays of shape (num_patches,), normalized to [0,1]

        entropy: use scipy.stats.entropy on pixel value histogram per patch
        edge_density: apply cv2.Sobel, compute mean absolute gradient per patch
        """
        rgb_image = image.convert("RGB")
        grid = self.current_grid_thw
        if grid is not None:
            grid_t, grid_h, grid_w = grid
            grid_t = max(1, int(grid_t))
            grid_h = max(1, int(grid_h))
            grid_w = max(1, int(grid_w))
        else:
            grid_t = 1
            grid_h = max(1, math.ceil(rgb_image.height / self.patch_size))
            grid_w = max(1, math.ceil(rgb_image.width / self.patch_size))

        resized_width = grid_w * self.patch_size
        resized_height = grid_h * self.patch_size
        resized = rgb_image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
        rgb = np.asarray(resized, dtype=np.uint8)
        grayscale = np.asarray(resized.convert("L"), dtype=np.float32)
        gradients = self._compute_gradient_magnitude(grayscale)

        entropy_values: list[float] = []
        edge_values: list[float] = []

        for row_index in range(grid_h):
            row_start = row_index * self.patch_size
            row_end = row_start + self.patch_size
            for column_index in range(grid_w):
                column_start = column_index * self.patch_size
                column_end = column_start + self.patch_size

                patch_rgb = rgb[row_start:row_end, column_start:column_end]
                histogram, _ = np.histogram(patch_rgb.reshape(-1), bins=256, range=(0, 256), density=False)
                patch_entropy = float(scipy_entropy(histogram + 1e-12)) / math.log(256)
                entropy_values.append(float(np.clip(patch_entropy, 0.0, 1.0)))

                patch_gradients = gradients[row_start:row_end, column_start:column_end]
                edge_values.append(float(np.mean(np.abs(patch_gradients))))

        entropy_array = np.asarray(entropy_values, dtype=np.float32)
        edge_array = normalize_zero_one(np.asarray(edge_values, dtype=np.float32))

        if grid_t > 1:
            entropy_array = np.tile(entropy_array, grid_t)
            edge_array = np.tile(edge_array, grid_t)

        return {
            "entropy": np.clip(entropy_array, 0.0, 1.0),
            "edge_density": np.clip(edge_array, 0.0, 1.0),
        }

    def _compute_gradient_magnitude(self, grayscale: np.ndarray) -> np.ndarray:
        try:
            import cv2

            grad_x = cv2.Sobel(grayscale, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(grayscale, cv2.CV_32F, 0, 1, ksize=3)
        except Exception:
            if not self._warned_edge_fallback:
                self.logger.warning("cv2 is unavailable. Falling back to scipy.ndimage.sobel for edge density.")
                self._warned_edge_fallback = True
            grad_x = ndimage.sobel(grayscale, axis=1, mode="reflect")
            grad_y = ndimage.sobel(grayscale, axis=0, mode="reflect")
        return np.abs(grad_x) + np.abs(grad_y)

    def register_hooks(self):
        """
        Register a forward hook on the vision encoder's layer at index 2.
        The hook captures the attention weights (shape: [heads, N, N]).
        Compute mean across heads -> shape [N, N].
        Extract CLS attention: row 0 -> shape [N].
        Store as self.cls_attention.
        """
        attention_module = self._get_attention_module(self.vision_layers[2])
        if attention_module is None:
            raise ValueError("Layer 2 does not expose an attention submodule.")

        def attention_hook(module, args, kwargs, output):
            try:
                attention = self._extract_attention_tensor(module, args, kwargs, output)
                if attention is None:
                    raise RuntimeError("Attention hook did not yield attention weights.")
                if attention.ndim == 4:
                    attention = attention[0]
                if attention.ndim != 3:
                    raise RuntimeError(f"Unexpected attention shape: {tuple(attention.shape)}")
                mean_attention = attention.float().mean(dim=0)
                self.cls_attention = mean_attention[0].detach().cpu().numpy().astype(np.float32)
                self._used_uniform_attention = False
            except Exception as exc:  # noqa: BLE001
                self.cls_attention = None
                self.logger.warning("Attention capture failed on layer 2. Falling back to uniform attention. %s", exc)
            return output

        self._hook_handles.append(self._register_forward_hook(attention_module, attention_hook))

    def _get_attention_module(self, layer: Any) -> Any | None:
        for attr_name in ("attn", "self_attn", "attention"):
            if hasattr(layer, attr_name):
                return getattr(layer, attr_name)
        return None

    def _extract_attention_tensor(self, module: Any, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> Any | None:
        import torch

        if isinstance(output, tuple):
            for candidate in output[1:]:
                if torch.is_tensor(candidate) and candidate.ndim in (3, 4):
                    return candidate

        if self.vision_style == "qwen2_flat" and hasattr(module, "qkv"):
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            if hidden_states is None:
                return None
            seq_length = int(hidden_states.shape[0])
            query_states, key_states, _ = (
                module.qkv(hidden_states).reshape(seq_length, 3, module.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
            )
            query_states = query_states.transpose(0, 1)
            key_states = key_states.transpose(0, 1)
            attention_scores = torch.matmul(query_states, key_states.transpose(1, 2)) * float(module.scaling)

            cu_seqlens = kwargs.get("cu_seqlens")
            if cu_seqlens is None and len(args) > 1:
                cu_seqlens = args[1]
            if cu_seqlens is not None and getattr(cu_seqlens, "numel", lambda: 0)() >= 2:
                first_sequence = int((cu_seqlens[1] - cu_seqlens[0]).item())
                attention_scores = attention_scores[:, :first_sequence, :first_sequence]

            return torch.softmax(attention_scores.float(), dim=-1)

        return None

    def compute_gui_ss(self, image: Image.Image) -> np.ndarray:
        """
        Combine signals into GUI Saliency Score per patch.
        GUI_SS = alpha * A + beta * (1 - E) + gamma * D
        Returns array of shape (num_patches,)
        """
        signals = self.compute_patch_signals(image)
        entropy = signals["entropy"]
        edge_density = signals["edge_density"]
        attention = self._get_attention_signal(entropy.shape[0])
        return (
            self.alpha * attention
            + self.beta * (1.0 - entropy)
            + self.gamma * edge_density
        ).astype(np.float32)

    def _get_attention_signal(self, target_length: int) -> np.ndarray:
        if self.cls_attention is None or len(self.cls_attention) == 0:
            self._used_uniform_attention = True
            return np.ones((target_length,), dtype=np.float32)

        attention = np.asarray(self.cls_attention, dtype=np.float32).reshape(-1)
        if attention.size == target_length + 1:
            attention = attention[1:]
        else:
            attention = align_signal(attention, target_length)

        attention = np.maximum(attention, 0.0)
        max_value = float(np.max(attention)) if attention.size else 0.0
        if max_value <= 0.0:
            self._used_uniform_attention = True
            return np.ones((target_length,), dtype=np.float32)

        self._used_uniform_attention = False
        return (attention / max_value).astype(np.float32)

    def get_tokens_to_keep(self, image: Image.Image, num_tokens: int) -> list[int]:
        """
        Returns indices of top-(1-drop_rate)*num_tokens patches by GUI_SS score.
        These are the patches to KEEP.
        """
        if num_tokens <= 0:
            self.last_pruning_info = {
                "keep_indices": [],
                "drop_indices": [],
                "blank_dropped_pct": 0.0,
                "used_uniform_attention": bool(self._used_uniform_attention),
            }
            return []

        signals = self.compute_patch_signals(image)
        entropy = align_signal(signals["entropy"], num_tokens)
        edge_density = align_signal(signals["edge_density"], num_tokens)
        attention = self._get_attention_signal(num_tokens)
        gui_scores = (
            self.alpha * attention
            + self.beta * (1.0 - entropy)
            + self.gamma * edge_density
        ).astype(np.float32)

        dropped_count = min(num_tokens, int(math.floor(num_tokens * self.drop_rate)))
        keep_count = max(1, num_tokens - dropped_count)
        ranked_indices = np.argsort(-gui_scores, kind="mergesort")
        selected = ranked_indices[:keep_count]
        keep_indices = np.sort(selected).astype(int).tolist()

        drop_mask = np.ones((num_tokens,), dtype=bool)
        drop_mask[selected] = False
        drop_indices = np.nonzero(drop_mask)[0].astype(int).tolist()
        blank_dropped = int(np.sum(entropy[drop_indices] < 0.1)) if drop_indices else 0
        blank_dropped_pct = (100.0 * blank_dropped / len(drop_indices)) if drop_indices else 0.0

        self.last_pruning_info = {
            "num_tokens": int(num_tokens),
            "kept_tokens": int(len(keep_indices)),
            "dropped_tokens": int(len(drop_indices)),
            "keep_indices": keep_indices,
            "drop_indices": drop_indices,
            "blank_dropped_count": int(blank_dropped),
            "blank_dropped_pct": float(blank_dropped_pct),
            "entropy": entropy,
            "edge_density": edge_density,
            "attention": attention,
            "gui_scores": gui_scores,
            "used_uniform_attention": bool(self._used_uniform_attention),
        }

        self.logger.info(
            "GAP drop_rate=%.2f | kept=%d/%d | dropped_blank=%.1f%% (%d/%d) | attention=%s",
            self.drop_rate,
            len(keep_indices),
            num_tokens,
            blank_dropped_pct,
            blank_dropped,
            len(drop_indices),
            "uniform" if self._used_uniform_attention else "captured",
        )

        return keep_indices

    def apply_pruning_hook(self):
        """
        Register a hook AFTER layer 2 of the ViT that:
        1. Computes which tokens to drop (using get_tokens_to_keep)
        2. Removes those token positions from the hidden states tensor
        3. Ensures subsequent layers only process kept tokens
        Note: [CLS] token (index 0) is NEVER dropped.
        """
        def vision_pre_hook(module, args, kwargs):
            self._runtime_state = {}
            return args, kwargs

        def vision_post_hook(module, args, kwargs, output):
            self._runtime_state = {}
            return output

        self._hook_handles.append(self._register_forward_pre_hook(self.vision_model, vision_pre_hook))
        self._hook_handles.append(self._register_forward_hook(self.vision_model, vision_post_hook))

        def prune_after_layer_two(module, args, kwargs, output):
            import torch

            if self.current_image is None:
                return output

            if self.vision_style == "qwen2_flat":
                if not torch.is_tensor(output) or output.ndim != 2:
                    return output
                original_length = int(output.shape[0])
                keep_indices = self.get_tokens_to_keep(self.current_image, original_length)
                keep_indices = self._ensure_token_zero(keep_indices, original_length)
                keep_tensor = torch.tensor(keep_indices, dtype=torch.long, device=output.device)
                self._runtime_state = {
                    "original_length": original_length,
                    "keep_indices": keep_tensor,
                }
                if keep_tensor.numel() == original_length:
                    return output
                return output.index_select(0, keep_tensor)

            if not torch.is_tensor(output) or output.ndim != 3 or output.shape[0] != 1:
                return output
            total_tokens = int(output.shape[1])
            patch_tokens = max(0, total_tokens - 1)
            keep_patch_indices = self.get_tokens_to_keep(self.current_image, patch_tokens)
            full_indices = [0] + [index + 1 for index in keep_patch_indices]
            keep_tensor = torch.tensor(full_indices, dtype=torch.long, device=output.device)
            self._runtime_state = {
                "original_length": total_tokens,
                "keep_indices": keep_tensor,
            }
            if keep_tensor.numel() == total_tokens:
                return output
            return output.index_select(1, keep_tensor)

        self._hook_handles.append(self._register_forward_hook(self.vision_layers[2], prune_after_layer_two))

        if self.vision_style == "qwen2_flat":
            for layer in self.vision_layers[3:]:
                self._hook_handles.append(self._register_forward_pre_hook(layer, self._qwen_block_pre_hook))
            if hasattr(self.vision_model, "merger"):
                self._hook_handles.append(self._register_forward_pre_hook(self.vision_model.merger, self._qwen_merger_pre_hook))
        else:
            for layer in self.vision_layers[3:]:
                self._hook_handles.append(self._register_forward_pre_hook(layer, self._generic_block_pre_hook))

    def _ensure_token_zero(self, keep_indices: list[int], num_tokens: int) -> list[int]:
        if num_tokens <= 0:
            return []
        if not keep_indices:
            keep_indices = [0]
        if 0 in keep_indices:
            return sorted(set(keep_indices))

        keep_count = len(keep_indices)
        gui_scores = np.asarray(self.last_pruning_info.get("gui_scores", np.zeros((num_tokens,), dtype=np.float32)))
        augmented = set(keep_indices)
        augmented.add(0)
        if len(augmented) > keep_count:
            removable = sorted(
                (index for index in augmented if index != 0),
                key=lambda index: (float(gui_scores[index]) if index < gui_scores.size else float("inf"), index),
            )
            augmented.remove(removable[0])

        keep_indices = sorted(augmented)
        drop_indices = [index for index in range(num_tokens) if index not in set(keep_indices)]
        entropy = np.asarray(self.last_pruning_info.get("entropy", np.zeros((num_tokens,), dtype=np.float32)))
        blank_dropped = int(np.sum(entropy[drop_indices] < 0.1)) if drop_indices else 0
        blank_dropped_pct = (100.0 * blank_dropped / len(drop_indices)) if drop_indices else 0.0

        self.last_pruning_info.update(
            {
                "keep_indices": keep_indices,
                "drop_indices": drop_indices,
                "kept_tokens": int(len(keep_indices)),
                "dropped_tokens": int(len(drop_indices)),
                "blank_dropped_count": int(blank_dropped),
                "blank_dropped_pct": float(blank_dropped_pct),
            }
        )
        return keep_indices

    def _qwen_block_pre_hook(self, module, args, kwargs):
        state = self._runtime_state
        if not state:
            return args, kwargs

        keep_indices = state["keep_indices"]
        hidden_states = args[0] if args else kwargs.get("hidden_states")
        if hidden_states is None:
            return args, kwargs

        if hidden_states.shape[0] != keep_indices.numel():
            hidden_states = hidden_states.index_select(0, keep_indices.to(hidden_states.device))

        new_args = (hidden_states,) + args[1:]
        new_kwargs = dict(kwargs)

        cu_seqlens = new_kwargs.get("cu_seqlens")
        if cu_seqlens is not None:
            new_kwargs["cu_seqlens"] = self._build_single_image_cu_seqlens(
                keep_indices.numel(),
                cu_seqlens.device,
                cu_seqlens.dtype,
            )

        position_embeddings = new_kwargs.get("position_embeddings")
        if position_embeddings is not None:
            cos, sin = position_embeddings
            gather_index = keep_indices.to(cos.device)
            if cos.shape[0] != keep_indices.numel():
                cos = cos.index_select(0, gather_index)
                sin = sin.index_select(0, gather_index)
            new_kwargs["position_embeddings"] = (cos, sin)

        return new_args, new_kwargs

    def _qwen_merger_pre_hook(self, module, args, kwargs):
        state = self._runtime_state
        if not state:
            return args, kwargs

        hidden_states = args[0] if args else kwargs.get("x")
        if hidden_states is None:
            return args, kwargs

        original_length = int(state["original_length"])
        keep_indices = state["keep_indices"].to(hidden_states.device)
        if hidden_states.shape[0] == original_length:
            return args, kwargs

        restored = hidden_states.new_zeros((original_length, hidden_states.shape[-1]))
        restored.index_copy_(0, keep_indices, hidden_states)

        if args:
            new_args = (restored,) + args[1:]
            return new_args, kwargs

        new_kwargs = dict(kwargs)
        new_kwargs["x"] = restored
        return args, new_kwargs

    def _generic_block_pre_hook(self, module, args, kwargs):
        state = self._runtime_state
        if not state:
            return args, kwargs

        keep_indices = state["keep_indices"]
        hidden_states = args[0] if args else kwargs.get("hidden_states")
        if hidden_states is None:
            return args, kwargs

        if hidden_states.shape[1] != keep_indices.numel():
            hidden_states = hidden_states.index_select(1, keep_indices.to(hidden_states.device))

        new_args = (hidden_states,) + args[1:]
        new_kwargs = dict(kwargs)

        attention_mask = new_kwargs.get("attention_mask")
        if attention_mask is not None:
            gather_index = keep_indices.to(attention_mask.device)
            if attention_mask.ndim == 2:
                attention_mask = attention_mask.index_select(1, gather_index)
            elif attention_mask.ndim == 4 and attention_mask.shape[-1] == state["original_length"]:
                attention_mask = attention_mask.index_select(-1, gather_index)
                if attention_mask.shape[-2] == state["original_length"]:
                    attention_mask = attention_mask.index_select(-2, gather_index)
            new_kwargs["attention_mask"] = attention_mask

        return new_args, new_kwargs

    def _build_single_image_cu_seqlens(self, kept_tokens: int, device: Any, dtype: Any) -> Any:
        import torch

        return torch.tensor([0, kept_tokens], dtype=dtype, device=device)

    def _register_forward_hook(self, module: Any, hook: Any) -> Any:
        try:
            return module.register_forward_hook(hook, with_kwargs=True)
        except TypeError:
            def wrapper(bound_module, args, output):
                return hook(bound_module, args, {}, output)

            return module.register_forward_hook(wrapper)

    def _register_forward_pre_hook(self, module: Any, hook: Any) -> Any:
        try:
            return module.register_forward_pre_hook(hook, with_kwargs=True)
        except TypeError:
            def wrapper(bound_module, args):
                new_args, _ = hook(bound_module, args, {})
                return new_args

            return module.register_forward_pre_hook(wrapper)

    def remove_hooks(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()


class Qwen2VLGapBackend(BaseBackend):
    def __init__(
        self,
        model_name: str,
        model_id: str,
        max_new_tokens: int,
        drop_rate: float,
        alpha: float,
        beta: float,
        gamma: float,
        hf_token: str | None,
    ) -> None:
        super().__init__(model_name=model_name, model_id=model_id, max_new_tokens=max_new_tokens, hf_token=hf_token)
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.torch = torch
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device_index = 0 if torch.cuda.is_available() else 0

        model_kwargs: dict[str, Any] = {
            "token": hf_token,
            "low_cpu_mem_usage": True,
        }
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
            model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = torch.float32

        self.processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.device_index = self.device.index or 0

        self.pruner = GAPPruner(
            self.model,
            drop_rate=drop_rate,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
        self.pruner.register_hooks()
        self.pruner.apply_pruning_hook()

    def set_drop_rate(self, drop_rate: float) -> None:
        self.pruner.drop_rate = float(drop_rate)

    def infer(self, image: Image.Image, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        prompt_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[prompt_text],
            images=[image],
            return_tensors="pt",
        )
        self.pruner.set_current_image(image, grid_thw=inputs.get("image_grid_thw"))

        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        prompt_length = int(inputs["input_ids"].shape[1])
        trimmed = generated[:, prompt_length:]
        decoded = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0].strip()

    def get_library_versions(self) -> dict[str, str]:
        versions = super().get_library_versions()
        versions["transformers"] = getattr(__import__("transformers"), "__version__", "unknown")
        return versions

    def cleanup(self) -> None:
        self.pruner.remove_hooks()
        del self.model
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def resolve_model_spec(model_name: str, model_id: str | None = None) -> dict[str, str]:
    if model_id:
        return {"model_id": model_id, "loader": "qwen2vl"}

    if model_name in DEFAULT_MODEL_SPECS:
        return DEFAULT_MODEL_SPECS[model_name]

    normalized = model_name.lower()
    if "qwen2-vl" in normalized:
        return {"model_id": model_name, "loader": "qwen2vl"}

    raise ValueError(
        "GAP evaluation currently supports Qwen2-VL architectures. "
        f"Received model_name={model_name!r}."
    )


def build_gap_backend(
    model_name: str,
    model_id: str | None,
    drop_rate: float,
    max_new_tokens: int,
    alpha: float,
    beta: float,
    gamma: float,
    hf_token: str | None,
) -> Qwen2VLGapBackend:
    spec = resolve_model_spec(model_name=model_name, model_id=model_id)
    if spec["loader"] != "qwen2vl":
        raise ValueError(f"Unsupported GAP loader type: {spec['loader']}")
    return Qwen2VLGapBackend(
        model_name=model_name,
        model_id=spec["model_id"],
        max_new_tokens=max_new_tokens,
        drop_rate=drop_rate,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        hf_token=hf_token,
    )


def summarize_pruning_info(pruning_info: dict[str, Any]) -> dict[str, Any]:
    return {
        "num_tokens": int(pruning_info.get("num_tokens", 0)),
        "kept_tokens": int(pruning_info.get("kept_tokens", 0)),
        "dropped_tokens": int(pruning_info.get("dropped_tokens", 0)),
        "blank_dropped_count": int(pruning_info.get("blank_dropped_count", 0)),
        "blank_dropped_pct": float(pruning_info.get("blank_dropped_pct", 0.0)),
        "used_uniform_attention": bool(pruning_info.get("used_uniform_attention", False)),
    }


def evaluate_single_drop_rate(
    backend: Qwen2VLGapBackend,
    model_name: str,
    model_id: str,
    drop_rate: float,
    dataset: pd.DataFrame,
    metadata_path: Path,
    output_dir: Path,
    split_strategy: str,
    dry_run: bool,
    logger: logging.Logger,
) -> dict[str, Any]:
    prediction_rows: list[dict[str, Any]] = []
    true_classes: list[str] = []
    predicted_classes: list[str] = []
    latencies: list[float] = []
    peak_vram_values: list[float] = []
    blank_drop_pct_values: list[float] = []
    kept_token_values: list[float] = []
    dropped_token_values: list[float] = []
    error_count = 0

    tag = format_drop_rate(drop_rate)
    desc = f"GAP {model_name} dr={tag}"

    with tqdm(dataset.itertuples(index=False), total=len(dataset), desc=desc, unit="image") as progress:
        for row in progress:
            image_path = Path(row.image_path)
            pruning_info: dict[str, Any] = {}

            if not image_path.exists():
                error_message = f"Image not found: {image_path}"
                logger.error(error_message)
                result = PredictionResult(
                    raw_output="",
                    predicted_class="INVALID",
                    predicted_label=-1,
                    predicted_bug_type=None,
                    latency_ms=0.0,
                    peak_vram_mb=0.0,
                    used_retry=False,
                    error=error_message,
                )
            else:
                result = run_inference_with_retry(
                    backend=backend,
                    image_path=image_path,
                    prompt=PROMPT,
                    logger=logger,
                )
                pruning_info = summarize_pruning_info(backend.pruner.last_pruning_info)
                if result.error:
                    logger.error("Inference failed for %s: %s", image_path, result.error)

            true_classes.append(row.true_class)
            predicted_classes.append(result.predicted_class)
            latencies.append(float(result.latency_ms))
            peak_vram_values.append(float(result.peak_vram_mb))
            blank_drop_pct_values.append(float(pruning_info.get("blank_dropped_pct", 0.0)))
            kept_token_values.append(float(pruning_info.get("kept_tokens", 0)))
            dropped_token_values.append(float(pruning_info.get("dropped_tokens", 0)))
            if result.error:
                error_count += 1

            prediction_rows.append(
                {
                    "sample_id": row.sample_id,
                    "image_path": row.image_path,
                    "true_class": row.true_class,
                    "true_label": int(row.true_label),
                    "true_bug_type": None if row.true_class == "CLEAN" else row.true_class,
                    "predicted_class": result.predicted_class,
                    "predicted_label": int(result.predicted_label),
                    "predicted_bug_type": result.predicted_bug_type,
                    "raw_output": result.raw_output,
                    "wall_clock_latency_ms": round(float(result.latency_ms), 3),
                    "peak_vram_mb": round(float(result.peak_vram_mb), 3),
                    "used_half_resolution_retry": result.used_retry,
                    "drop_rate": float(drop_rate),
                    "num_tokens": int(pruning_info.get("num_tokens", 0)),
                    "kept_tokens": int(pruning_info.get("kept_tokens", 0)),
                    "dropped_tokens": int(pruning_info.get("dropped_tokens", 0)),
                    "blank_dropped_pct": round(float(pruning_info.get("blank_dropped_pct", 0.0)), 3),
                    "used_uniform_attention": bool(pruning_info.get("used_uniform_attention", False)),
                    "error": result.error,
                }
            )

    metrics = compute_metrics(true_classes=true_classes, predicted_classes=predicted_classes)

    safe_model_name = sanitize_tag(model_name)
    predictions_path = output_dir / f"{safe_model_name}_dr{tag}_predictions.csv"
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)

    payload = {
        "model_name": model_name,
        "model_id": model_id,
        "backend": "transformers",
        "metadata_path": str(metadata_path.resolve()),
        "results_dir": str(output_dir.resolve()),
        "split_strategy": split_strategy,
        "dry_run": bool(dry_run),
        "num_samples": int(len(dataset)),
        "num_errors": int(error_count),
        "prompt": PROMPT,
        "metrics": metrics,
        "latency_ms": {
            "mean": safe_mean(latencies),
            "p95": percentile(latencies, 0.95),
        },
        "vram_mb": {
            "peak": max(peak_vram_values) if peak_vram_values else 0.0,
            "mean": safe_mean(peak_vram_values),
            "p95": percentile(peak_vram_values, 0.95),
        },
        "artifacts": {
            "predictions_csv": str(predictions_path.resolve()),
            "errors_log": str((output_dir / "errors.log").resolve()),
        },
        "library_versions": backend.get_library_versions(),
        "gap": {
            "drop_rate": float(drop_rate),
            "alpha": float(backend.pruner.alpha),
            "beta": float(backend.pruner.beta),
            "gamma": float(backend.pruner.gamma),
            "blank_dropped_pct_mean": safe_mean(blank_drop_pct_values),
            "blank_dropped_pct_p95": percentile(blank_drop_pct_values, 0.95),
            "kept_tokens_mean": safe_mean(kept_token_values),
            "dropped_tokens_mean": safe_mean(dropped_token_values),
        },
    }

    result_path = output_dir / f"{safe_model_name}_dr{tag}.json"
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def evaluate_gap(
    model_name,
    drop_rates=DEFAULT_DROP_RATES,
    metadata_csv="data/metadata.csv",
    output_dir="results/gap/",
    test_size: int = 750,
    seed: int = 42,
    dry_run: bool = False,
    max_new_tokens: int = 16,
    alpha: float = 0.4,
    beta: float = 0.3,
    gamma: float = 0.3,
    hf_token: str | None = None,
    model_id: str | None = None,
):
    """
    For each drop_rate, run full test set evaluation using GAPPruner.
    Save results to results/gap/{model_name}_dr{drop_rate}.json
    Each result file has same schema as baseline evaluation.
    """
    drop_rates = [float(rate) for rate in drop_rates]
    if not drop_rates:
        raise ValueError("evaluate_gap requires at least one drop rate.")

    metadata_path = Path(metadata_csv)
    results_dir = Path(output_dir)
    logger = setup_gap_logging(results_dir)

    metadata = load_metadata(metadata_path)
    dataset, split_strategy = build_test_split(metadata, test_size=test_size, seed=seed)
    if dry_run:
        dataset = dataset.head(10).copy()

    resolved = resolve_model_spec(model_name=model_name, model_id=model_id)
    backend = build_gap_backend(
        model_name=model_name,
        model_id=resolved["model_id"],
        drop_rate=float(drop_rates[0]),
        max_new_tokens=max_new_tokens,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        hf_token=hf_token,
    )

    results: dict[str, Any] = {}
    try:
        for drop_rate in drop_rates:
            backend.set_drop_rate(float(drop_rate))
            results[format_drop_rate(float(drop_rate))] = evaluate_single_drop_rate(
                backend=backend,
                model_name=model_name,
                model_id=resolved["model_id"],
                drop_rate=float(drop_rate),
                dataset=dataset,
                metadata_path=metadata_path,
                output_dir=results_dir,
                split_strategy=split_strategy,
                dry_run=dry_run,
                logger=logger,
            )
            if backend.torch is not None and backend.torch.cuda.is_available():
                backend.torch.cuda.empty_cache()
    finally:
        backend.cleanup()

    return results


def test_single_image(
    image_path: Path,
    model_name: str,
    drop_rate: float,
    output_dir: Path,
    max_new_tokens: int,
    alpha: float,
    beta: float,
    gamma: float,
    hf_token: str | None,
    model_id: str | None = None,
) -> dict[str, Any]:
    logger = setup_gap_logging(output_dir)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    resolved = resolve_model_spec(model_name=model_name, model_id=model_id)
    backend = build_gap_backend(
        model_name=model_name,
        model_id=resolved["model_id"],
        drop_rate=drop_rate,
        max_new_tokens=max_new_tokens,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        hf_token=hf_token,
    )

    try:
        result = run_inference_with_retry(
            backend=backend,
            image_path=image_path,
            prompt=PROMPT,
            logger=logger,
        )
        pruning_info = summarize_pruning_info(backend.pruner.last_pruning_info)
        payload = {
            "image_path": str(image_path.resolve()),
            "model_name": model_name,
            "model_id": resolved["model_id"],
            "drop_rate": float(drop_rate),
            "prompt": PROMPT,
            "prediction": {
                "raw_output": result.raw_output,
                "predicted_class": result.predicted_class,
                "predicted_label": int(result.predicted_label),
                "predicted_bug_type": result.predicted_bug_type,
                "latency_ms": float(result.latency_ms),
                "peak_vram_mb": float(result.peak_vram_mb),
                "used_half_resolution_retry": bool(result.used_retry),
                "error": result.error,
            },
            "gap": pruning_info,
        }
        result_path = output_dir / f"test_single_{sanitize_tag(model_name)}_dr{format_drop_rate(drop_rate)}.json"
        result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
    finally:
        backend.cleanup()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen2vl")
    parser.add_argument("--model-id")
    parser.add_argument("--metadata-csv", type=Path, default=Path("data/metadata.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/gap"))
    parser.add_argument("--drop-rates", default=",".join(format_drop_rate(rate) for rate in DEFAULT_DROP_RATES))
    parser.add_argument("--drop-rate", type=float, default=0.5, help="Single-image test drop rate.")
    parser.add_argument("--test-size", type=int, default=750)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--test_single", type=Path)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.drop_rate <= 0.9:
        raise ValueError(f"--drop-rate must be between 0.0 and 0.9. Received {args.drop_rate}.")

    if args.test_single is not None:
        payload = test_single_image(
            image_path=args.test_single,
            model_name=args.model,
            model_id=args.model_id,
            drop_rate=float(args.drop_rate),
            output_dir=args.output_dir,
            max_new_tokens=args.max_new_tokens,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            hf_token=args.hf_token,
        )
        print(json.dumps(payload, indent=2))
        return

    drop_rates = parse_drop_rates(args.drop_rates)
    payload = evaluate_gap(
        model_name=args.model,
        model_id=args.model_id,
        drop_rates=drop_rates,
        metadata_csv=str(args.metadata_csv),
        output_dir=str(args.output_dir),
        test_size=args.test_size,
        seed=args.seed,
        dry_run=args.dry_run,
        max_new_tokens=args.max_new_tokens,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        hf_token=args.hf_token,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

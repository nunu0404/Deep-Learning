#!/usr/bin/env python3
"""Evaluate FastV text-conditioned vision-token pruning baselines."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    from evaluation.evaluate_baseline import PROMPT, is_oom_error
    from evaluation.pruning_sweep import evaluate_pruning_sweep
    from models.gap_pruning import (
        DEFAULT_DROP_RATES,
        GAPPruner,
        Qwen2VLGapBackend,
        align_signal,
        format_drop_rate,
        get_nested_attr,
        parse_drop_rates,
        resolve_model_spec,
    )
except ImportError:  # pragma: no cover - package-relative execution fallback
    from .evaluate_baseline import PROMPT, is_oom_error
    from .pruning_sweep import evaluate_pruning_sweep
    from ..models.gap_pruning import (
        DEFAULT_DROP_RATES,
        GAPPruner,
        Qwen2VLGapBackend,
        align_signal,
        format_drop_rate,
        get_nested_attr,
        parse_drop_rates,
        resolve_model_spec,
    )


def compute_fastv_image_scores(
    attention: Any,
    text_token_indices: list[int],
    image_token_indices: list[int],
) -> np.ndarray:
    """Average text-query to image-key attention across heads and text positions."""
    if hasattr(attention, "detach"):
        attention = attention.detach().float().cpu().numpy()
    array = np.asarray(attention, dtype=np.float32)
    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"FastV attention must have shape [heads, seq, seq], got {array.shape}.")

    sequence_length = int(array.shape[-1])
    valid_image = [index for index in image_token_indices if 0 <= index < sequence_length]
    valid_text = [index for index in text_token_indices if 0 <= index < array.shape[-2]]
    if not valid_image:
        return np.zeros((0,), dtype=np.float32)
    if not valid_text:
        valid_text = [index for index in range(array.shape[-2]) if index not in set(valid_image)]
    if not valid_text:
        valid_text = list(range(array.shape[-2]))

    text_to_image = array[:, valid_text, :][:, :, valid_image]
    return text_to_image.mean(axis=(0, 1)).astype(np.float32)


def select_fastv_keep_indices(
    attention: Any,
    text_token_indices: list[int],
    image_token_indices: list[int],
    num_tokens: int,
    drop_rate: float,
) -> list[int]:
    scores = compute_fastv_image_scores(
        attention=attention,
        text_token_indices=text_token_indices,
        image_token_indices=image_token_indices,
    )
    scores = align_signal(scores, num_tokens)
    dropped_count = min(num_tokens, int(math.floor(num_tokens * drop_rate)))
    keep_count = max(1, num_tokens - dropped_count)
    selected = np.argsort(-scores, kind="mergesort")[:keep_count]
    keep_indices = sorted(set(selected.astype(int).tolist()))
    if num_tokens > 0 and 0 not in keep_indices:
        keep_indices = sorted({0, *keep_indices})
        if len(keep_indices) > keep_count:
            removable = sorted(
                (index for index in keep_indices if index != 0),
                key=lambda index: (float(scores[index]) if index < scores.size else float("inf"), index),
            )[0]
            keep_indices.remove(removable)
    return keep_indices


class FastVPruner(GAPPruner):
    """FastV selector using decoder-layer text-to-image attention."""

    def __init__(self, model: Any, drop_rate: float = 0.5, prune_layer: int = 2) -> None:
        super().__init__(model=model, drop_rate=drop_rate)
        self.prune_layer = int(prune_layer)
        self.fastv_scores: np.ndarray | None = None
        self.image_token_indices: list[int] = []
        self.text_token_indices: list[int] = []
        self.pruning_enabled = True
        self.used_attention_fallback = False
        self.capture_failures = 0

    def set_token_context(self, input_ids: Any, attention_mask: Any | None, image_token_id: int | None) -> None:
        if hasattr(input_ids, "detach"):
            ids = input_ids.detach().cpu().numpy()
        else:
            ids = np.asarray(input_ids)
        ids = ids.reshape(ids.shape[-1])

        if attention_mask is not None:
            if hasattr(attention_mask, "detach"):
                mask = attention_mask.detach().cpu().numpy().reshape(-1)
            else:
                mask = np.asarray(attention_mask).reshape(-1)
            active = {int(index) for index in np.nonzero(mask > 0)[0]}
        else:
            active = set(range(ids.shape[0]))

        if image_token_id is None:
            self.image_token_indices = []
        else:
            self.image_token_indices = [
                int(index)
                for index, token_id in enumerate(ids.tolist())
                if index in active and int(token_id) == int(image_token_id)
            ]
        image_set = set(self.image_token_indices)
        self.text_token_indices = [int(index) for index in sorted(active) if index not in image_set]

    def clear_fastv_scores(self) -> None:
        self.fastv_scores = None
        self.used_attention_fallback = False

    def register_language_attention_hook(self) -> None:
        layer = self._get_language_layer(self.model, self.prune_layer)

        def language_layer_hook(module, args, kwargs, output):
            attention = self._extract_language_attention(output)
            if attention is None:
                return output
            self.update_fastv_scores(attention)
            return output

        self._hook_handles.append(self._register_forward_hook(layer, language_layer_hook))

    def _get_language_layer(self, model: Any, layer_index: int) -> Any:
        layers = get_nested_attr(model, "model.layers")
        if layers is None:
            layers = get_nested_attr(model, "model.model.layers")
        if layers is None:
            raise ValueError("FastV requires a decoder stack at model.model.layers.")
        if layer_index < 0 or layer_index >= len(layers):
            raise ValueError(f"FastV prune_layer={layer_index} is outside decoder depth {len(layers)}.")
        return layers[layer_index]

    def _extract_language_attention(self, output: Any) -> Any | None:
        import torch

        if torch.is_tensor(output) and output.ndim in (3, 4):
            return output
        if isinstance(output, dict):
            for key in ("attentions", "attention", "self_attn_weights"):
                candidate = output.get(key)
                if torch.is_tensor(candidate) and candidate.ndim in (3, 4):
                    return candidate
        if isinstance(output, tuple):
            for candidate in output[1:]:
                if torch.is_tensor(candidate) and candidate.ndim in (3, 4):
                    return candidate
        return None

    def update_fastv_scores(self, attention: Any) -> None:
        scores = compute_fastv_image_scores(
            attention=attention,
            text_token_indices=self.text_token_indices,
            image_token_indices=self.image_token_indices,
        )
        if scores.size:
            self.fastv_scores = scores
            self.used_attention_fallback = False

    def get_tokens_to_keep(self, image: Image.Image, num_tokens: int) -> list[int]:
        if num_tokens <= 0:
            self.last_pruning_info = {
                "keep_indices": [],
                "drop_indices": [],
                "blank_dropped_pct": 0.0,
                "used_uniform_attention": False,
            }
            return []

        if not self.pruning_enabled:
            keep_indices = list(range(num_tokens))
            self.last_pruning_info = {
                "num_tokens": int(num_tokens),
                "kept_tokens": int(num_tokens),
                "dropped_tokens": 0,
                "keep_indices": keep_indices,
                "drop_indices": [],
                "blank_dropped_count": 0,
                "blank_dropped_pct": 0.0,
                "entropy": np.ones((num_tokens,), dtype=np.float32),
                "edge_density": np.zeros((num_tokens,), dtype=np.float32),
                "attention": np.ones((num_tokens,), dtype=np.float32),
                "gui_scores": np.ones((num_tokens,), dtype=np.float32),
                "used_uniform_attention": False,
            }
            return keep_indices

        if self.fastv_scores is None or self.fastv_scores.size == 0:
            self.used_attention_fallback = True
            self.capture_failures += 1
            scores = np.ones((num_tokens,), dtype=np.float32)
        else:
            scores = align_signal(self.fastv_scores, num_tokens)

        dropped_count = min(num_tokens, int(math.floor(num_tokens * self.drop_rate)))
        keep_count = max(1, num_tokens - dropped_count)
        selected = np.argsort(-scores, kind="mergesort")[:keep_count]
        keep_indices = np.sort(selected).astype(int).tolist()

        drop_mask = np.ones((num_tokens,), dtype=bool)
        drop_mask[selected] = False
        drop_indices = np.nonzero(drop_mask)[0].astype(int).tolist()

        self.last_pruning_info = {
            "num_tokens": int(num_tokens),
            "kept_tokens": int(len(keep_indices)),
            "dropped_tokens": int(len(drop_indices)),
            "keep_indices": keep_indices,
            "drop_indices": drop_indices,
            "blank_dropped_count": 0,
            "blank_dropped_pct": 0.0,
            "entropy": np.ones((num_tokens,), dtype=np.float32),
            "edge_density": np.zeros((num_tokens,), dtype=np.float32),
            "attention": scores.astype(np.float32),
            "gui_scores": scores.astype(np.float32),
            "used_uniform_attention": bool(self.used_attention_fallback),
            "prune_layer": int(self.prune_layer),
            "image_token_count": int(len(self.image_token_indices)),
            "text_token_count": int(len(self.text_token_indices)),
        }
        return self._ensure_token_zero(keep_indices, num_tokens)


class Qwen2VLFastVBackend(Qwen2VLGapBackend):
    """Qwen2-VL backend with FastV text-conditioned token selection."""

    def __init__(
        self,
        model_name: str,
        model_id: str,
        max_new_tokens: int,
        drop_rate: float,
        prune_layer: int,
        hf_token: str | None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            drop_rate=drop_rate,
            alpha=0.0,
            beta=0.0,
            gamma=0.0,
            hf_token=hf_token,
        )
        self.logger = logging.getLogger("evaluate_fastv")
        self.image_token_id = self._resolve_image_token_id()
        self.pruner.remove_hooks()
        self.pruner = FastVPruner(self.model, drop_rate=drop_rate, prune_layer=prune_layer)
        self.pruner.register_language_attention_hook()
        self.pruner.apply_pruning_hook()

    def _resolve_image_token_id(self) -> int | None:
        for root in (getattr(self.model, "config", None), getattr(getattr(self.model, "model", None), "config", None)):
            token_id = getattr(root, "image_token_id", None)
            if token_id is not None:
                return int(token_id)
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            return None
        for token in ("<|image_pad|>", "<image>"):
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is not None and token_id != getattr(tokenizer, "unk_token_id", None):
                return int(token_id)
        return None

    def set_drop_rate(self, drop_rate: float) -> None:
        super().set_drop_rate(drop_rate)
        self.pruner.capture_failures = 0

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
        self.pruner.set_token_context(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            image_token_id=self.image_token_id,
        )

        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        self._capture_fastv_scores(inputs)

        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )

        prompt_length = int(inputs["input_ids"].shape[1])
        trimmed = generated[:, prompt_length:]
        decoded = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0].strip()

    def _capture_fastv_scores(self, inputs: dict[str, Any]) -> None:
        self.pruner.clear_fastv_scores()
        original_enabled = self.pruner.pruning_enabled
        self.pruner.pruning_enabled = False
        try:
            with self.torch.inference_mode():
                outputs = self.model(
                    **inputs,
                    output_attentions=True,
                    use_cache=False,
                )
            if self.pruner.fastv_scores is None:
                attentions = getattr(outputs, "attentions", None)
                if attentions is not None and len(attentions) > self.pruner.prune_layer:
                    self.pruner.update_fastv_scores(attentions[self.pruner.prune_layer])
        except Exception as exc:  # noqa: BLE001
            if is_oom_error(exc):
                raise
            self.logger.warning("FastV attention capture failed; using deterministic uniform scores. %s: %s", type(exc).__name__, exc)
            self.pruner.used_attention_fallback = True
        finally:
            self.pruner.pruning_enabled = original_enabled


def build_fastv_backend(
    model_name: str,
    model_id: str | None,
    drop_rate: float,
    max_new_tokens: int,
    prune_layer: int,
    hf_token: str | None,
) -> Qwen2VLFastVBackend:
    spec = resolve_model_spec(model_name=model_name, model_id=model_id)
    if spec["loader"] != "qwen2vl":
        raise ValueError(f"Unsupported FastV loader type: {spec['loader']}")
    return Qwen2VLFastVBackend(
        model_name=model_name,
        model_id=spec["model_id"],
        max_new_tokens=max_new_tokens,
        drop_rate=drop_rate,
        prune_layer=prune_layer,
        hf_token=hf_token,
    )


def evaluate_fastv(
    model_name: str,
    drop_rates: list[float] = DEFAULT_DROP_RATES,
    metadata_csv: str | Path = "data/metadata.csv",
    output_dir: str | Path = "results/fastv",
    test_size: int = 750,
    seed: int = 42,
    dry_run: bool = False,
    max_new_tokens: int = 16,
    prune_layer: int = 2,
    hf_token: str | None = None,
    model_id: str | None = None,
) -> dict[str, Any]:
    resolved = resolve_model_spec(model_name=model_name, model_id=model_id)

    def factory(initial_drop_rate: float) -> Qwen2VLFastVBackend:
        return build_fastv_backend(
            model_name=model_name,
            model_id=resolved["model_id"],
            drop_rate=initial_drop_rate,
            max_new_tokens=max_new_tokens,
            prune_layer=prune_layer,
            hf_token=hf_token,
        )

    return evaluate_pruning_sweep(
        method_key="fastv",
        model_name=model_name,
        model_id=resolved["model_id"],
        drop_rates=[float(rate) for rate in drop_rates],
        metadata_csv=metadata_csv,
        output_dir=output_dir,
        test_size=test_size,
        seed=seed,
        dry_run=dry_run,
        logger_name="evaluate_fastv",
        backend_factory=factory,
        section_extras=lambda backend: {
            "prune_layer": int(backend.pruner.prune_layer),
            "text_conditioned": True,
            "attention_fallback_count": int(backend.pruner.capture_failures),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen2vl")
    parser.add_argument("--model-id")
    parser.add_argument("--metadata-csv", type=Path, default=Path("data/metadata.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/fastv"))
    parser.add_argument("--drop-rates", default=",".join(format_drop_rate(rate) for rate in DEFAULT_DROP_RATES))
    parser.add_argument("--test-size", type=int, default=750)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--prune-layer", type=int, default=2)
    parser.add_argument("--dry-run", dest="dry_run", action="store_true")
    parser.add_argument("--dry_run", dest="dry_run", action="store_true")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = evaluate_fastv(
        model_name=args.model,
        model_id=args.model_id,
        drop_rates=parse_drop_rates(args.drop_rates),
        metadata_csv=args.metadata_csv,
        output_dir=args.output_dir,
        test_size=args.test_size,
        seed=args.seed,
        dry_run=args.dry_run,
        max_new_tokens=args.max_new_tokens,
        prune_layer=args.prune_layer,
        hf_token=args.hf_token,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

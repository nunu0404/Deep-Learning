#!/usr/bin/env python3
"""Run an InternVL2-26B smoke test on demo GUI-Bug screenshots."""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
import traceback
import types
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms import InterpolationMode

VENDOR_DEPS = Path(__file__).resolve().parents[1] / ".vendor" / "internvl2_pydeps"
if VENDOR_DEPS.exists():
    sys.path.insert(0, str(VENDOR_DEPS))

from transformers import AutoModel, AutoTokenizer
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.utils import GenerationMixin
from transformers.modeling_utils import PreTrainedModel


MODEL_ID = "OpenGVLab/InternVL2-26B"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BUG_LABELS = {
    "1": "OVERLAP",
    "2": "OVERFLOW",
    "3": "ZINDEX",
    "4": "TRUNCATION",
    "5": "CONTRAST",
}
PROMPT = """You are inspecting a web UI screenshot for visual bugs.

Examine carefully. Does the screenshot contain ANY of these visual bugs?
- OVERLAP: elements that visually overlap when they shouldn't
- OVERFLOW: text spilling outside its container box
- ZINDEX: elements layered incorrectly
- TRUNCATION: text cut off mid-word
- CONTRAST: insufficient foreground/background contrast

Respond with EXACTLY one line:
BUG: OVERLAP
BUG: OVERFLOW
BUG: ZINDEX
BUG: TRUNCATION
BUG: CONTRAST
CLEAN

Your answer:"""


def patch_transformers_internvl_compatibility() -> None:
    """Bridge older InternVL2 remote code with newer transformers loaders."""
    if hasattr(PreTrainedModel, "all_tied_weights_keys"):
        return

    def get_all_tied_weights_keys(self: PreTrainedModel) -> dict[str, None]:
        cached = getattr(self, "_compat_all_tied_weights_keys", None)
        if isinstance(cached, dict):
            return cached
        keys = getattr(self, "_tied_weights_keys", None) or []
        return {str(key): None for key in keys}

    def set_all_tied_weights_keys(self: PreTrainedModel, value: object) -> None:
        if isinstance(value, dict):
            self.__dict__["_compat_all_tied_weights_keys"] = value
        elif isinstance(value, (list, tuple, set)):
            self.__dict__["_compat_all_tied_weights_keys"] = {str(key): None for key in value}
        else:
            self.__dict__["_compat_all_tied_weights_keys"] = {}

    PreTrainedModel.all_tied_weights_keys = property(  # type: ignore[attr-defined]
        get_all_tied_weights_keys,
        set_all_tied_weights_keys,
    )


def patch_generation_mixin(model: torch.nn.Module) -> None:
    """Attach generation helpers expected by older InternVL2 chat code."""
    for module in (model, getattr(model, "language_model", None)):
        if module is None:
            continue
        module_class = module.__class__
        if not hasattr(module, "generate"):
            for name in dir(GenerationMixin):
                if name.startswith("__") or hasattr(module_class, name):
                    continue
                attribute = getattr(GenerationMixin, name)
                if callable(attribute):
                    setattr(module_class, name, attribute)
        if not hasattr(module, "generation_config") and hasattr(module, "config"):
            module.generation_config = GenerationConfig.from_model_config(module.config)
        if hasattr(module, "generation_config"):
            module.generation_config.use_cache = False

        def expand_inputs_for_generation(
            self: torch.nn.Module,
            input_ids: torch.LongTensor | None = None,
            expand_size: int = 1,
            is_encoder_decoder: bool = False,
            **model_kwargs: object,
        ) -> tuple[torch.LongTensor | None, dict[str, object]]:
            return GenerationMixin._expand_inputs_for_generation(
                expand_size=expand_size,
                is_encoder_decoder=is_encoder_decoder,
                input_ids=input_ids,
                **model_kwargs,
            )

        module_class._expand_inputs_for_generation = expand_inputs_for_generation

    language_model = getattr(model, "language_model", None)
    if language_model is not None and hasattr(language_model, "prepare_inputs_for_generation"):
        original_prepare = language_model.prepare_inputs_for_generation

        def prepare_inputs_for_generation_compat(
            self: torch.nn.Module,
            input_ids: torch.LongTensor | None = None,
            past_key_values: object | None = None,
            inputs_embeds: torch.Tensor | None = None,
            **kwargs: object,
        ) -> object:
            past_key_values = past_key_values if past_key_values is not None else kwargs.pop("past_key_values", None)
            kwargs.pop("past_key_values", None)
            if hasattr(past_key_values, "to_legacy_cache"):
                kwargs["past_key_values"] = past_key_values.to_legacy_cache()
            elif type(past_key_values).__name__ == "DynamicCache":
                kwargs["past_key_values"] = None
            elif past_key_values is not None:
                kwargs["past_key_values"] = past_key_values
            if inputs_embeds is not None:
                kwargs["inputs_embeds"] = inputs_embeds
            return original_prepare(input_ids, **kwargs)

        language_model.prepare_inputs_for_generation = types.MethodType(
            prepare_inputs_for_generation_compat,
            language_model,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--screenshots-glob", default="demo_artifacts/data/screenshots/*_1280x800_*.png")
    parser.add_argument("--limit", type=int, help="Limit the number of images for debugging.")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    return parser.parse_args()


def build_transform() -> T.Compose:
    return T.Compose(
        [
            T.Lambda(lambda image: image.convert("RGB") if image.mode != "RGB" else image),
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def true_label_for(path: Path) -> str:
    name = path.name
    if "clean" in name:
        return "CLEAN"
    if "_B" not in name:
        raise ValueError(f"Cannot infer bug label from filename: {name}")
    bug_code = name.split("_B", maxsplit=1)[1][0]
    if bug_code not in BUG_LABELS:
        raise ValueError(f"Unsupported bug code B{bug_code} in filename: {name}")
    return BUG_LABELS[bug_code]


def is_correct(true_label: str, response: str) -> bool:
    normalized = response.upper()
    if true_label == "CLEAN":
        return "CLEAN" in normalized and "BUG" not in normalized
    return true_label in normalized


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for InternVL2-26B.")

    print("=" * 80)
    print(f"Loading {args.model_id} (takes 3-5 min)...")
    print("=" * 80)

    started_at = time.time()
    patch_transformers_internvl_compatibility()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cuda:0",
        trust_remote_code=True,
        use_flash_attn=False,
    ).eval()
    patch_generation_mixin(model)
    print(f"\nLoaded in {time.time() - started_at:.1f}s")
    print(f"GPU memory: {torch.cuda.memory_allocated(0) / 1024**3:.1f} GB / 96 GB")

    print("\n=== Top-level modules ===")
    for name, module in model.named_children():
        print(f"  model.{name}: {type(module).__name__}")

    vision_model = getattr(model, "vision_model", None)
    if vision_model is not None:
        print(f"\nvision_model: {type(vision_model).__name__}")
        encoder = getattr(vision_model, "encoder", None)
        if encoder is not None and hasattr(encoder, "layers"):
            print(f"   .encoder.layers: len={len(encoder.layers)}")

    transform = build_transform()
    files = [Path(path) for path in sorted(glob.glob(args.screenshots_glob))]
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise FileNotFoundError(f"No screenshots matched: {args.screenshots_glob}")

    print(f"\n=== Testing {len(files)} images ===\n")
    results: list[tuple[str, str, bool]] = []
    for path in files:
        true_label = true_label_for(path)
        image = Image.open(path).convert("RGB")
        pixel_values = transform(image).unsqueeze(0).to(torch.float16).cuda()

        started_at = time.time()
        try:
            with torch.inference_mode():
                response = model.chat(
                    tokenizer,
                    pixel_values,
                    f"<image>\n{PROMPT}",
                    generation_config={
                        "max_new_tokens": args.max_new_tokens,
                        "do_sample": False,
                    },
                )
            response = str(response).strip()
        except Exception as exc:  # noqa: BLE001
            response = f"ERROR: {type(exc).__name__}: {exc}"
            traceback.print_exc()

        elapsed = time.time() - started_at
        correct = is_correct(true_label, response)
        mark = "OK" if correct else "NO"
        response_preview = response.replace("\n", " ")[:48]
        print(f"{mark} {path.name:<35} true={true_label:<11} -> {response_preview:<48} ({elapsed:.1f}s)")
        results.append((true_label, response, correct))

    bug_results = [row for row in results if row[0] != "CLEAN"]
    bug_correct = sum(1 for _, _, correct in bug_results if correct)
    total_correct = sum(1 for _, _, correct in results if correct)

    print(f"\n{'=' * 80}")
    print("=== 7B vs 26B Comparison ===")
    print("  Qwen2-VL-7B   bug detection: 0/4 (0%)")
    print(f"  InternVL2-26B bug detection: {bug_correct}/{len(bug_results)} ({bug_correct / max(len(bug_results), 1):.0%})")
    print(f"  InternVL2-26B overall:       {total_correct}/{len(results)} ({total_correct / len(results):.0%})")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()

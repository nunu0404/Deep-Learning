#!/usr/bin/env python3
"""Run InternVL2-26B prompt variants on selected demo GUI-Bug screenshots."""

from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

import torch
from PIL import Image

from test_internvl2_26b_demo import (
    MODEL_ID,
    build_transform,
    is_correct,
    patch_generation_mixin,
    patch_transformers_internvl_compatibility,
)
from transformers import AutoModel, AutoTokenizer


PROMPTS = {
    "Detailed-CoT": """Look at this web UI screenshot very carefully.

Examine these specific things:
1. Are any text labels or buttons overlapping each other?
2. Does any text appear to be cut off, ending abruptly mid-word?
3. Does any text spill out of its container box?
4. Are any visual elements positioned wrong (covering things they shouldn't)?

After looking, write a brief observation (1 sentence), then on a new line:
ANSWER: BUG: <type> or CLEAN

Where <type> is one of: OVERLAP, OVERFLOW, ZINDEX, TRUNCATION""",
    "Forced-Yes": """This is from a dataset where exactly half the screenshots contain visual bugs that were intentionally injected. The bugs are clearly visible if you look carefully.

Look at this screenshot. Even subtle visual problems count. The four bug types:
- OVERLAP: elements visually overlapping (margins, positions)
- OVERFLOW: text/content extending beyond its container
- ZINDEX: wrong stacking order (something covers something it shouldn't)
- TRUNCATION: text cut off, ending abruptly, or "..." visible

Respond ONLY with one of:
BUG: OVERLAP
BUG: OVERFLOW
BUG: ZINDEX
BUG: TRUNCATION
CLEAN""",
    "Examples-First": """I'll teach you to spot UI bugs by example:

EXAMPLE 1: A header reading "Welcome to our..." with text cut off -> BUG: TRUNCATION
EXAMPLE 2: Two boxes that visually overlap each other -> BUG: OVERLAP
EXAMPLE 3: Long sentence extending past the right edge of its box -> BUG: OVERFLOW
EXAMPLE 4: Footer appearing on top of header -> BUG: ZINDEX
EXAMPLE 5: Clean, well-aligned page -> CLEAN

Now classify this screenshot using the same single-line format:""",
}

DEFAULT_TEST_IMAGES = [
    ("00000_1280x800_clean.png", "CLEAN"),
    ("00000_1280x800_B1.png", "OVERLAP"),
    ("00001_1280x800_B2.png", "OVERFLOW"),
    ("00002_1280x800_B3.png", "ZINDEX"),
    ("00003_1280x800_B4.png", "TRUNCATION"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--screenshots-dir", default="demo_artifacts/data/screenshots")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument(
        "--prompts",
        default=",".join(PROMPTS),
        help=f"Comma-separated prompt names. Available: {', '.join(PROMPTS)}",
    )
    return parser.parse_args()


def load_model(model_id: str) -> tuple[AutoTokenizer, AutoModel]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for InternVL2-26B.")

    print("Loading InternVL2-26B...")
    started_at = time.time()
    patch_transformers_internvl_compatibility()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cuda:0",
        trust_remote_code=True,
        use_flash_attn=False,
    ).eval()
    patch_generation_mixin(model)
    print(f"Loaded in {time.time() - started_at:.1f}s")
    print(f"GPU: {torch.cuda.memory_allocated(0) / 1024**3:.1f} GB\n")
    return tokenizer, model


def main() -> None:
    args = parse_args()
    selected_prompt_names = [name.strip() for name in args.prompts.split(",") if name.strip()]
    unknown_prompts = sorted(set(selected_prompt_names) - set(PROMPTS))
    if unknown_prompts:
        raise ValueError(f"Unknown prompt name(s): {', '.join(unknown_prompts)}")

    tokenizer, model = load_model(args.model_id)
    transform = build_transform()
    screenshots_dir = Path(args.screenshots_dir)

    for prompt_name in selected_prompt_names:
        prompt_text = PROMPTS[prompt_name]
        print(f"\n{'=' * 100}\n=== Prompt: {prompt_name} ===\n{'=' * 100}")
        correct_count = 0

        for filename, true_label in DEFAULT_TEST_IMAGES:
            path = screenshots_dir / filename
            if not path.exists():
                raise FileNotFoundError(path)

            image = Image.open(path).convert("RGB")
            pixel_values = transform(image).unsqueeze(0).to(torch.float16).cuda()

            try:
                with torch.inference_mode():
                    response = model.chat(
                        tokenizer,
                        pixel_values,
                        f"<image>\n{prompt_text}",
                        generation_config={
                            "max_new_tokens": args.max_new_tokens,
                            "do_sample": False,
                        },
                    )
                response = str(response).strip()
            except Exception as exc:  # noqa: BLE001
                response = f"ERROR: {type(exc).__name__}: {exc}"
                traceback.print_exc()

            correct = is_correct(true_label, response)
            if correct:
                correct_count += 1
            mark = "OK" if correct else "NO"
            response_preview = response.replace("\n", " | ")[:80]
            print(f"  {mark} [{true_label:<10}] {filename:<30} -> {response_preview}")

        print(f"  Score: {correct_count}/{len(DEFAULT_TEST_IMAGES)}")


if __name__ == "__main__":
    main()

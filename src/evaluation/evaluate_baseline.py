#!/usr/bin/env python3
"""Evaluate quantized VLM baselines on GUI-BugBench screenshots."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from tqdm import tqdm

# Avoid Xet-based partial fetch issues on some HF mirrors.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


PROMPT = """You are a QA engineer reviewing a web UI screenshot.
Does this UI screenshot contain any visual bugs such as:
- overlapping elements
- text overflow outside its container
- z-index layering errors
- text truncation (unwanted ellipsis)
- insufficient color contrast

Answer with ONLY: "BUG: <bug_type>" or "CLEAN"
Bug types: OVERLAP, OVERFLOW, ZINDEX, TRUNCATION, CONTRAST
"""

LABELS = ["CLEAN", "OVERLAP", "OVERFLOW", "ZINDEX", "TRUNCATION", "CONTRAST"]
BUG_LABELS = LABELS[1:]
BUG_TYPE_MAP = {
    None: "CLEAN",
    "": "CLEAN",
    "NONE": "CLEAN",
    "B1": "OVERLAP",
    "B2": "OVERFLOW",
    "B3": "ZINDEX",
    "B4": "TRUNCATION",
    "B5": "CONTRAST",
}
MODEL_SPECS = {
    "qwen2vl": {
        "backend": "vllm",
        "model_id": "Qwen/Qwen2-VL-7B-Instruct-AWQ",
        "trust_remote_code": False,
    },
    "llava": {
        "backend": "transformers",
        "model_id": "llava-hf/llava-v1.6-mistral-7b-hf",
        "trust_remote_code": False,
    },
    "internvl": {
        "backend": "vllm",
        "model_id": "OpenGVLab/InternVL2-8B-AWQ",
        "trust_remote_code": True,
    },
}
INVALID_PREDICTION = "INVALID"


@dataclass
class PredictionResult:
    raw_output: str
    predicted_class: str
    predicted_label: int
    predicted_bug_type: str | None
    latency_ms: float
    peak_vram_mb: float
    used_retry: bool
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--metadata-path", type=Path, default=Path("data/metadata.csv"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--test-size", type=int, default=750)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    return parser.parse_args()


def setup_logging(results_dir: Path) -> logging.Logger:
    results_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("evaluate_baseline")
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


def normalize_bug_type(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text or text == "NAN":
        return None
    return text


def derive_true_class(row: pd.Series) -> str:
    label_value = int(row["label"])
    bug_type = normalize_bug_type(row.get("bug_type"))

    if label_value == 0:
        return "CLEAN"

    if bug_type not in BUG_TYPE_MAP:
        raise ValueError(f"Unsupported bug_type in metadata: {row.get('bug_type')!r}")
    return BUG_TYPE_MAP[bug_type]


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    dataframe = pd.read_csv(metadata_path, dtype={"sample_id": str, "image_path": str, "bug_type": str})
    required_columns = {"sample_id", "image_path", "label", "bug_type"}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing_columns)}")

    dataframe["label"] = pd.to_numeric(dataframe["label"], errors="raise").astype(int)
    dataframe["bug_type"] = dataframe["bug_type"].apply(normalize_bug_type)
    dataframe["true_class"] = dataframe.apply(derive_true_class, axis=1)
    dataframe["true_label"] = (dataframe["true_class"] != "CLEAN").astype(int)
    dataframe["image_path"] = dataframe["image_path"].map(lambda value: str((Path.cwd() / value).resolve()))
    dataframe = dataframe.sort_values(["sample_id", "image_path"], kind="stable").reset_index(drop=True)
    return dataframe


def _allocate_group_counts(group_sizes: dict[str, int], target_total: int) -> dict[str, int]:
    total_items = sum(group_sizes.values())
    if total_items == 0:
        return {key: 0 for key in group_sizes}

    target_total = min(target_total, total_items)
    raw_allocations = {
        key: (group_sizes[key] / total_items) * target_total
        for key in group_sizes
    }
    allocations = {
        key: min(group_sizes[key], int(raw_allocations[key]))
        for key in group_sizes
    }
    remaining = target_total - sum(allocations.values())

    remainders = sorted(
        (
            (raw_allocations[key] - allocations[key], key)
            for key in group_sizes
            if allocations[key] < group_sizes[key]
        ),
        reverse=True,
    )
    for _, key in remainders:
        if remaining <= 0:
            break
        allocations[key] += 1
        remaining -= 1

    return allocations


def build_test_split(dataframe: pd.DataFrame, test_size: int, seed: int) -> tuple[pd.DataFrame, str]:
    if "split" in dataframe.columns:
        split_series = dataframe["split"].astype(str).str.strip().str.lower()
        test_df = dataframe.loc[split_series == "test"].copy()
        if not test_df.empty:
            test_df = test_df.sort_values(["sample_id", "image_path"], kind="stable")
            if len(test_df) > test_size:
                test_df = test_df.head(test_size).copy()
            return test_df.reset_index(drop=True), "metadata:test"

    group_sizes = dataframe.groupby("true_class").size().to_dict()
    allocations = _allocate_group_counts(group_sizes, test_size)

    parts: list[pd.DataFrame] = []
    for class_name in LABELS:
        class_df = dataframe.loc[dataframe["true_class"] == class_name].copy()
        if class_df.empty:
            continue
        take = allocations.get(class_name, 0)
        if take <= 0:
            continue
        sampled = class_df.sample(n=take, random_state=seed, replace=False)
        parts.append(sampled)

    if not parts:
        raise ValueError("Unable to derive a non-empty test split from metadata.")

    test_df = pd.concat(parts, ignore_index=True)
    test_df = test_df.sort_values(["sample_id", "image_path"], kind="stable").reset_index(drop=True)
    return test_df, "derived:stratified_test"


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def safe_mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def parse_model_output(text: str) -> tuple[str, int, str | None]:
    normalized = re.sub(r"\s+", " ", text).strip().upper()
    match = re.search(r"\bBUG\s*:\s*(OVERLAP|OVERFLOW|ZINDEX|TRUNCATION|CONTRAST)\b", normalized)
    if match:
        bug_type = match.group(1)
        return bug_type, 1, bug_type

    if re.fullmatch(r"CLEAN\b", normalized) or re.search(r"\bCLEAN\b", normalized):
        return "CLEAN", 0, None

    for bug_type in BUG_LABELS:
        if re.search(rf"\b{bug_type}\b", normalized):
            return bug_type, 1, bug_type

    return INVALID_PREDICTION, -1, None


def is_oom_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda oom" in message or "cuda out of memory" in message


def resize_half(image: Image.Image) -> Image.Image:
    width = max(1, image.width // 2)
    height = max(1, image.height // 2)
    return image.resize((width, height), Image.Resampling.BILINEAR)


def compute_metrics(true_classes: list[str], predicted_classes: list[str]) -> dict[str, Any]:
    if len(true_classes) != len(predicted_classes):
        raise ValueError("Mismatched ground-truth and prediction lengths.")

    per_class: dict[str, dict[str, float | int]] = {}
    for class_name in LABELS:
        true_positive = sum(1 for gold, pred in zip(true_classes, predicted_classes) if gold == class_name and pred == class_name)
        false_positive = sum(1 for gold, pred in zip(true_classes, predicted_classes) if gold != class_name and pred == class_name)
        false_negative = sum(1 for gold, pred in zip(true_classes, predicted_classes) if gold == class_name and pred != class_name)
        support = sum(1 for gold in true_classes if gold == class_name)

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        per_class[class_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(support),
        }

    accuracy = sum(1 for gold, pred in zip(true_classes, predicted_classes) if gold == pred) / len(true_classes) if true_classes else 0.0
    macro = {
        "precision": safe_mean([float(per_class[label]["precision"]) for label in LABELS]),
        "recall": safe_mean([float(per_class[label]["recall"]) for label in LABELS]),
        "f1": safe_mean([float(per_class[label]["f1"]) for label in LABELS]),
    }
    macro_bug_only = {
        "precision": safe_mean([float(per_class[label]["precision"]) for label in BUG_LABELS]),
        "recall": safe_mean([float(per_class[label]["recall"]) for label in BUG_LABELS]),
        "f1": safe_mean([float(per_class[label]["f1"]) for label in BUG_LABELS]),
    }

    predicted_label_order = LABELS + [INVALID_PREDICTION]
    confusion = [
        [sum(1 for gold, pred in zip(true_classes, predicted_classes) if gold == gold_label and pred == pred_label)
         for pred_label in predicted_label_order]
        for gold_label in LABELS
    ]

    return {
        "accuracy": float(accuracy),
        "macro": macro,
        "macro_bug_only": macro_bug_only,
        "per_class": per_class,
        "per_bug_type": {label: per_class[label] for label in BUG_LABELS},
        "confusion_matrix": {
            "true_labels": LABELS,
            "predicted_labels": predicted_label_order,
            "matrix": confusion,
        },
    }


class BaseBackend:
    def __init__(self, model_name: str, model_id: str, max_new_tokens: int, hf_token: str | None = None) -> None:
        self.model_name = model_name
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.hf_token = hf_token
        self.device_index = 0
        self.device: Any | None = None
        self.torch: Any | None = None

    def infer(self, image: Image.Image, prompt: str) -> str:
        raise NotImplementedError

    def get_library_versions(self) -> dict[str, str]:
        versions: dict[str, str] = {}
        if self.torch is not None:
            versions["torch"] = getattr(self.torch, "__version__", "unknown")
        return versions

    def cleanup(self) -> None:
        return None


class VllmVisionBackend(BaseBackend):
    def __init__(self, model_name: str, model_id: str, max_new_tokens: int, trust_remote_code: bool, hf_token: str | None = None) -> None:
        super().__init__(model_name=model_name, model_id=model_id, max_new_tokens=max_new_tokens, hf_token=hf_token)
        import torch
        from vllm import LLM, SamplingParams

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for vLLM inference.")

        if hf_token:
            os.environ.setdefault("HF_TOKEN", hf_token)
            os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", hf_token)

        self.torch = torch
        self.device_index = torch.cuda.current_device()
        self.device = torch.device(f"cuda:{self.device_index}")
        self.sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_new_tokens,
        )
        self.llm = LLM(
            model=model_id,
            trust_remote_code=trust_remote_code,
            limit_mm_per_prompt={"image": 1},
            max_num_seqs=1,
            gpu_memory_utilization=0.9,
        )

    def infer(self, image: Image.Image, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_pil", "image_pil": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        try:
            outputs = self.llm.chat(messages, sampling_params=self.sampling_params, use_tqdm=False)
        except TypeError:
            outputs = self.llm.chat(messages, sampling_params=self.sampling_params)
        return outputs[0].outputs[0].text.strip()

    def get_library_versions(self) -> dict[str, str]:
        versions = super().get_library_versions()
        versions["vllm"] = getattr(__import__("vllm"), "__version__", "unknown")
        return versions

    def cleanup(self) -> None:
        del self.llm
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


class LlavaTransformersBackend(BaseBackend):
    def __init__(self, model_name: str, model_id: str, max_new_tokens: int, hf_token: str | None = None) -> None:
        super().__init__(model_name=model_name, model_id=model_id, max_new_tokens=max_new_tokens, hf_token=hf_token)
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, LlavaNextForConditionalGeneration

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for LLaVA inference.")

        self.torch = torch
        self.device_index = torch.cuda.current_device()
        self.device = torch.device(f"cuda:{self.device_index}")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            token=hf_token,
        )

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
        if hasattr(self.processor, "apply_chat_template"):
            prompt_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_text = f"[INST] <image>\n{prompt} [/INST]"

        inputs = self.processor(
            text=[prompt_text],
            images=[image],
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self.device)
            if hasattr(value, "to")
            else value
            for key, value in inputs.items()
        }

        generated = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )
        trimmed = generated[:, inputs["input_ids"].shape[1] :]
        decoded = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0].strip()

    def get_library_versions(self) -> dict[str, str]:
        versions = super().get_library_versions()
        versions["transformers"] = getattr(__import__("transformers"), "__version__", "unknown")
        versions["bitsandbytes"] = getattr(__import__("bitsandbytes"), "__version__", "unknown")
        return versions

    def cleanup(self) -> None:
        del self.model
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def build_backend(args: argparse.Namespace) -> BaseBackend:
    spec = MODEL_SPECS[args.model]
    backend_type = spec["backend"]
    if backend_type == "vllm":
        return VllmVisionBackend(
            model_name=args.model,
            model_id=spec["model_id"],
            max_new_tokens=args.max_new_tokens,
            trust_remote_code=bool(spec["trust_remote_code"]),
            hf_token=args.hf_token,
        )
    if backend_type == "transformers":
        return LlavaTransformersBackend(
            model_name=args.model,
            model_id=spec["model_id"],
            max_new_tokens=args.max_new_tokens,
            hf_token=args.hf_token,
        )
    raise ValueError(f"Unsupported backend type: {backend_type}")


def run_inference_with_retry(
    backend: BaseBackend,
    image_path: Path,
    prompt: str,
    logger: logging.Logger,
) -> PredictionResult:
    with Image.open(image_path) as opened_image:
        image = opened_image.convert("RGB")

    torch = backend.torch
    attempt_images = [(image, False), (resize_half(image), True)]
    last_error: Exception | None = None

    for attempt_index, (attempt_image, used_retry) in enumerate(attempt_images):
        try:
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(backend.device_index)
                torch.cuda.synchronize(backend.device_index)

            start = time.perf_counter()
            raw_output = backend.infer(attempt_image, prompt)
            if torch is not None and torch.cuda.is_available():
                torch.cuda.synchronize(backend.device_index)
            latency_ms = (time.perf_counter() - start) * 1000.0
            peak_vram_mb = (
                float(torch.cuda.max_memory_allocated(backend.device_index) / (1024 ** 2))
                if torch is not None and torch.cuda.is_available()
                else 0.0
            )
            predicted_class, predicted_label, predicted_bug_type = parse_model_output(raw_output)
            return PredictionResult(
                raw_output=raw_output,
                predicted_class=predicted_class,
                predicted_label=predicted_label,
                predicted_bug_type=predicted_bug_type,
                latency_ms=float(latency_ms),
                peak_vram_mb=float(peak_vram_mb),
                used_retry=used_retry,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if is_oom_error(exc) and attempt_index == 0:
                logger.error("OOM on %s. Retrying once at half resolution.", image_path)
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            break

    error_text = f"{type(last_error).__name__}: {last_error}" if last_error is not None else "unknown error"
    return PredictionResult(
        raw_output="",
        predicted_class=INVALID_PREDICTION,
        predicted_label=-1,
        predicted_bug_type=None,
        latency_ms=0.0,
        peak_vram_mb=0.0,
        used_retry=bool(last_error and is_oom_error(last_error)),
        error=error_text,
    )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    logger = setup_logging(args.results_dir)
    metadata = load_metadata(args.metadata_path)
    test_df, split_strategy = build_test_split(metadata, test_size=args.test_size, seed=args.seed)
    if args.dry_run:
        test_df = test_df.head(10).copy()

    try:
        backend = build_backend(args)
    except Exception as exc:  # noqa: BLE001
        logger.error("Backend initialization failed: %s: %s", type(exc).__name__, exc)
        raise
    spec = MODEL_SPECS[args.model]

    prediction_rows: list[dict[str, Any]] = []
    true_classes: list[str] = []
    predicted_classes: list[str] = []
    latencies: list[float] = []
    peak_vram_values: list[float] = []
    error_count = 0

    try:
        with tqdm(test_df.itertuples(index=False), total=len(test_df), desc=f"Evaluating {args.model}", unit="image") as progress:
            for row in progress:
                image_path = Path(row.image_path)
                if not image_path.exists():
                    error_message = f"Image not found: {image_path}"
                    logger.error(error_message)
                    result = PredictionResult(
                        raw_output="",
                        predicted_class=INVALID_PREDICTION,
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
                    if result.error:
                        logger.error("Inference failed for %s: %s", image_path, result.error)

                true_classes.append(row.true_class)
                predicted_classes.append(result.predicted_class)
                latencies.append(float(result.latency_ms))
                peak_vram_values.append(float(result.peak_vram_mb))
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
                        "error": result.error,
                    }
                )
    finally:
        backend.cleanup()

    metrics = compute_metrics(true_classes=true_classes, predicted_classes=predicted_classes)
    predictions_path = args.results_dir / f"{args.model}_predictions.csv"
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)

    result_payload = {
        "model_name": args.model,
        "model_id": spec["model_id"],
        "backend": spec["backend"],
        "metadata_path": str(args.metadata_path.resolve()),
        "results_dir": str(args.results_dir.resolve()),
        "split_strategy": split_strategy,
        "dry_run": bool(args.dry_run),
        "num_samples": int(len(test_df)),
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
            "errors_log": str((args.results_dir / "errors.log").resolve()),
        },
        "library_versions": backend.get_library_versions(),
    }

    result_path = args.results_dir / f"{args.model}_baseline.json"
    result_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
    return result_payload


def main() -> None:
    args = parse_args()
    try:
        evaluate(args)
    except Exception:  # noqa: BLE001
        logger = setup_logging(args.results_dir)
        logger.exception("Evaluation failed.")
        raise


if __name__ == "__main__":
    main()

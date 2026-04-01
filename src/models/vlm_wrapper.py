"""Unified wrapper API over supported VLM backends."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

from evaluation.evaluate_baseline import (
    BaseBackend,
    LlavaTransformersBackend,
    MODEL_SPECS,
    VllmVisionBackend,
)


@dataclass
class VLMWrapper:
    model_name: str
    model_id: str
    backend: BaseBackend

    def infer(self, image, prompt: str) -> str:
        return self.backend.infer(image, prompt)

    def cleanup(self) -> None:
        self.backend.cleanup()

    def get_library_versions(self) -> dict[str, str]:
        return self.backend.get_library_versions()


def build_vlm_backend(
    model_name: str,
    max_new_tokens: int = 16,
    hf_token: str | None = None,
) -> BaseBackend:
    spec = MODEL_SPECS[model_name]
    if spec["backend"] == "vllm":
        return VllmVisionBackend(
            model_name=model_name,
            model_id=spec["model_id"],
            max_new_tokens=max_new_tokens,
            trust_remote_code=bool(spec["trust_remote_code"]),
            hf_token=hf_token,
        )
    if spec["backend"] == "transformers":
        return LlavaTransformersBackend(
            model_name=model_name,
            model_id=spec["model_id"],
            max_new_tokens=max_new_tokens,
            hf_token=hf_token,
        )
    raise ValueError(f"Unsupported backend type: {spec['backend']}")


def load_vlm(
    model_name: str,
    max_new_tokens: int = 16,
    hf_token: str | None = None,
) -> VLMWrapper:
    backend = build_vlm_backend(
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        hf_token=hf_token,
    )
    return VLMWrapper(
        model_name=model_name,
        model_id=MODEL_SPECS[model_name]["model_id"],
        backend=backend,
    )


def namespace_from_wrapper(model_name: str, max_new_tokens: int, hf_token: str | None) -> argparse.Namespace:
    return argparse.Namespace(
        model=model_name,
        max_new_tokens=max_new_tokens,
        hf_token=hf_token,
    )


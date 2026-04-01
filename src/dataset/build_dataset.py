#!/usr/bin/env python3
"""Build a GUI visual bug dataset from HuggingFaceM4/WebSight.

Default behavior balances the output around the requested target:
- 2,500 clean screenshots
- 500 buggy screenshots for each of B1-B5

Use ``--all-bugs`` to render all five bug variants for every sample instead.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import random
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from bs4.element import Tag
from datasets import load_dataset
from PIL import Image, ImageColor
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright
from skimage.metrics import structural_similarity
from tqdm import tqdm

try:
    from .bug_injectors import bug_injector_callables
except ImportError:  # pragma: no cover - script execution fallback
    from dataset.bug_injectors import bug_injector_callables


BUG_TYPES = ("B1", "B2", "B3", "B4", "B5")
BUG_INDEX = {bug_type: index for index, bug_type in enumerate(BUG_TYPES, start=1)}
METADATA_COLUMNS = ["sample_id", "image_path", "label", "bug_type", "vss_score"]
TEXT_TAGS = (
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "p",
    "span",
    "a",
    "button",
    "label",
    "li",
    "td",
    "div",
)
TAILWIND_COLOR_MAP = {
    "white": "#ffffff",
    "black": "#000000",
    "gray-50": "#f9fafb",
    "gray-100": "#f3f4f6",
    "gray-200": "#e5e7eb",
    "gray-300": "#d1d5db",
    "gray-400": "#9ca3af",
    "gray-500": "#6b7280",
    "gray-600": "#4b5563",
    "gray-700": "#374151",
    "gray-800": "#1f2937",
    "gray-900": "#111827",
    "red-500": "#ef4444",
    "red-600": "#dc2626",
    "yellow-400": "#facc15",
    "yellow-500": "#eab308",
    "green-500": "#22c55e",
    "green-600": "#16a34a",
    "blue-500": "#3b82f6",
    "blue-600": "#2563eb",
    "indigo-500": "#6366f1",
    "purple-500": "#a855f7",
    "pink-500": "#ec4899",
    "teal-200": "#99f6e4",
    "teal-500": "#14b8a6",
    "teal-600": "#0d9488",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="HuggingFaceM4/WebSight")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=2500)
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Target total image count. In balanced mode this includes clean and buggy images.",
    )
    parser.add_argument(
        "--samples-per-bug",
        type=int,
        default=500,
        help="Used in balanced mode only. Ignored when --all-bugs is enabled.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--viewport-width", type=int, default=1280)
    parser.add_argument("--viewport-height", type=int, default=800)
    parser.add_argument(
        "--all-bugs",
        action="store_true",
        help="Render all B1-B5 variants for each sample instead of a single balanced bug.",
    )
    return parser.parse_args()


def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("build_dataset")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def ensure_output_dirs(output_dir: Path) -> dict[str, Path]:
    screenshots_dir = output_dir / "screenshots"
    html_dir = output_dir / "html"
    output_dir.mkdir(parents=True, exist_ok=True)
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)
    return {
        "output": output_dir,
        "screenshots": screenshots_dir,
        "html": html_dir,
        "metadata": output_dir / "metadata.csv",
        "stats": output_dir / "dataset_stats.json",
        "checkpoint": output_dir / "checkpoint.json",
        "status": output_dir / "sample_status.jsonl",
        "logs": output_dir / "render_errors.log",
    }


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sample_id_for(index: int) -> str:
    return f"{index:05d}"


def relative_path_str(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def normalize_html(raw_html: str) -> str:
    html = raw_html.strip()
    if not html:
        raise ValueError("Sample HTML is empty.")
    if "<html" in html.lower():
        return html
    return f"<html><body>{html}</body></html>"


def extract_html(sample: dict[str, Any]) -> str:
    for key in ("text", "html", "content"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return normalize_html(value)
    raise ValueError(f"No HTML string found in sample keys: {sorted(sample.keys())}")


def load_processed_indices(status_path: Path) -> set[int]:
    processed: set[int] = set()
    if not status_path.exists():
        return processed

    with status_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            index = payload.get("index")
            if isinstance(index, int):
                processed.add(index)
    return processed


def normalize_bug_type(value: str | None) -> str:
    if value in (None, "", "None"):
        return "None"
    return value


def dedupe_metadata(metadata_path: Path) -> set[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    if not metadata_path.exists():
        return seen

    unique_rows: list[dict[str, Any]] = []
    duplicate_count = 0

    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (row["sample_id"], normalize_bug_type(row.get("bug_type")))
            if key in seen:
                duplicate_count += 1
                continue
            seen.add(key)
            unique_rows.append(row)

    if duplicate_count:
        with metadata_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=METADATA_COLUMNS)
            writer.writeheader()
            writer.writerows(unique_rows)

    return seen


def append_metadata_rows(
    metadata_path: Path,
    rows: list[dict[str, Any]],
    existing_keys: set[tuple[str, str]],
) -> int:
    rows_to_write: list[dict[str, Any]] = []
    for row in rows:
        key = (row["sample_id"], normalize_bug_type(row["bug_type"]))
        if key in existing_keys:
            continue
        existing_keys.add(key)
        rows_to_write.append(row)

    if not rows_to_write:
        return 0

    write_header = not metadata_path.exists() or metadata_path.stat().st_size == 0
    with metadata_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows_to_write)
    return len(rows_to_write)


def append_status(status_path: Path, payload: dict[str, Any]) -> None:
    with status_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
        }

    array = np.asarray(values, dtype=np.float32)
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std()),
    }


def write_dataset_stats(metadata_path: Path, stats_path: Path) -> dict[str, Any]:
    if not metadata_path.exists() or metadata_path.stat().st_size == 0:
        stats = {
            "counts": {"total_images": 0, "clean": 0, "bug_total": 0, **{bug: 0 for bug in BUG_TYPES}},
            "vss": {"overall": numeric_summary([]), "by_bug_type": {bug: numeric_summary([]) for bug in BUG_TYPES}},
        }
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        return stats

    dataframe = pd.read_csv(metadata_path)
    if dataframe.empty:
        stats = {
            "counts": {"total_images": 0, "clean": 0, "bug_total": 0, **{bug: 0 for bug in BUG_TYPES}},
            "vss": {"overall": numeric_summary([]), "by_bug_type": {bug: numeric_summary([]) for bug in BUG_TYPES}},
        }
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        return stats

    dataframe["bug_type"] = dataframe["bug_type"].fillna("None")
    dataframe["vss_score"] = pd.to_numeric(dataframe["vss_score"], errors="coerce")

    counts = {
        "total_images": int(len(dataframe)),
        "clean": int((dataframe["label"] == 0).sum()),
        "bug_total": int((dataframe["label"] == 1).sum()),
    }
    for bug_type in BUG_TYPES:
        counts[bug_type] = int((dataframe["bug_type"] == bug_type).sum())

    bug_rows = dataframe[dataframe["label"] == 1]
    vss_overall = numeric_summary(bug_rows["vss_score"].dropna().tolist())
    vss_by_bug_type = {
        bug_type: numeric_summary(
            bug_rows.loc[bug_rows["bug_type"] == bug_type, "vss_score"].dropna().tolist()
        )
        for bug_type in BUG_TYPES
    }

    stats = {"counts": counts, "vss": {"overall": vss_overall, "by_bug_type": vss_by_bug_type}}
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def write_checkpoint(
    checkpoint_path: Path,
    args: argparse.Namespace,
    processed_indices: set[int],
    stats: dict[str, Any],
) -> None:
    active_processed = sum(1 for index in processed_indices if index < args.max_samples)
    payload = {
        "timestamp": now_iso(),
        "dataset": args.dataset,
        "split": args.split,
        "max_samples": args.max_samples,
        "all_bugs": args.all_bugs,
        "samples_per_bug": args.samples_per_bug,
        "processed_samples": active_processed,
        "remaining_samples": max(args.max_samples - active_processed, 0),
        "counts": stats["counts"],
    }
    checkpoint_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def class_string(tag: Tag) -> str:
    classes = tag.get("class", [])
    if isinstance(classes, list):
        return " ".join(str(item) for item in classes)
    return str(classes)


def get_style_property(style: str, prop: str) -> str | None:
    match = re.search(rf"(?i)(?:^|;)\s*{re.escape(prop)}\s*:\s*([^;]+)", style)
    if not match:
        return None
    return match.group(1).strip()


def set_style_property(style: str, prop: str, value: str) -> str:
    pattern = re.compile(rf"(?i)(^|;)\s*{re.escape(prop)}\s*:\s*[^;]+")
    if pattern.search(style):
        updated = pattern.sub(lambda match: f"{match.group(1)} {prop}: {value}", style, count=1)
        return updated.strip().strip(";") + ";"

    chunks = [chunk.strip() for chunk in style.split(";") if chunk.strip()]
    chunks.append(f"{prop}: {value}")
    return "; ".join(chunks) + ";"


def apply_inline_styles(tag: Tag, styles: dict[str, str]) -> None:
    style = str(tag.get("style", ""))
    for prop, value in styles.items():
        style = set_style_property(style, prop, value)
    tag["style"] = style


def clean_text(tag: Tag) -> str:
    return re.sub(r"\s+", " ", tag.get_text(" ", strip=True)).strip()


def find_first_text_container(soup: BeautifulSoup, min_length: int = 12) -> Tag | None:
    for tag in soup.find_all(TEXT_TAGS):
        text = clean_text(tag)
        if len(text) < min_length:
            continue
        if tag.name == "div":
            child_text_tags = tag.find_all(TEXT_TAGS[:-1], recursive=False)
            if child_text_tags:
                continue
        return tag
    return None


def find_first_colored_text_container(soup: BeautifulSoup, min_length: int = 4) -> Tag | None:
    for tag in soup.find_all(TEXT_TAGS):
        text = clean_text(tag)
        if len(text) < min_length:
            continue
        style = str(tag.get("style", ""))
        if get_style_property(style, "color") or re.search(r"\btext-[a-z]", class_string(tag)):
            return tag
    return find_first_text_container(soup, min_length=min_length)


def pick_bug_types_for_sample(index: int, args: argparse.Namespace) -> list[str]:
    if args.all_bugs:
        return list(BUG_TYPES)
    bug_budget = args.samples_per_bug * len(BUG_TYPES)
    if index >= bug_budget:
        return []
    return [BUG_TYPES[index % len(BUG_TYPES)]]


def rng_for_sample(seed: int, sample_index: int, bug_type: str) -> random.Random:
    return random.Random(seed + sample_index * 101 + BUG_INDEX[bug_type] * 100_003)


def scale_css_dimension(value: str | None, factor: float) -> str | None:
    if not value:
        return None

    match = re.search(r"(-?\d+(?:\.\d+)?)(px|%|rem|em|vw|vh)", value.strip())
    if not match:
        return None

    amount = float(match.group(1))
    unit = match.group(2)
    scaled = amount * factor

    if unit == "%":
        scaled = max(10.0, min(scaled, 95.0))
    else:
        scaled = max(40.0, scaled)

    if scaled.is_integer():
        return f"{int(scaled)}{unit}"
    return f"{scaled:.1f}{unit}"


def parse_rgba_components(color: str) -> tuple[int, int, int] | None:
    match = re.fullmatch(
        r"rgba?\(\s*(\d{1,3})[\s,]+(\d{1,3})[\s,]+(\d{1,3})(?:[\s,]+[\d.]+)?\s*\)",
        color.strip(),
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    red, green, blue = (max(0, min(int(part), 255)) for part in match.groups())
    return red, green, blue


def parse_color(value: str | None) -> tuple[int, int, int] | None:
    if not value:
        return None

    normalized = value.strip().lower()
    if normalized in {"transparent", "inherit", "currentcolor"}:
        return None

    rgba = parse_rgba_components(normalized)
    if rgba is not None:
        return rgba

    try:
        return ImageColor.getrgb(normalized)
    except ValueError:
        return None


def get_color_from_style(style: str, prop: str) -> tuple[int, int, int] | None:
    return parse_color(get_style_property(style, prop))


def get_color_from_classes(tag: Tag, prefix: str) -> tuple[int, int, int] | None:
    for class_name in class_string(tag).split():
        if not class_name.startswith(prefix):
            continue
        tailwind_key = class_name[len(prefix) :]
        if tailwind_key in TAILWIND_COLOR_MAP:
            return parse_color(TAILWIND_COLOR_MAP[tailwind_key])
    return None


def get_background_color(tag: Tag | None) -> tuple[int, int, int]:
    current = tag
    while isinstance(current, Tag):
        style = str(current.get("style", ""))
        for prop in ("background-color", "background"):
            color = get_color_from_style(style, prop)
            if color is not None:
                return color
        color = get_color_from_classes(current, "bg-")
        if color is not None:
            return color
        current = current.parent if isinstance(current.parent, Tag) else None
    return 255, 255, 255


def get_text_color(tag: Tag | None) -> tuple[int, int, int]:
    current = tag
    while isinstance(current, Tag):
        style = str(current.get("style", ""))
        color = get_color_from_style(style, "color")
        if color is not None:
            return color
        color = get_color_from_classes(current, "text-")
        if color is not None:
            return color
        current = current.parent if isinstance(current.parent, Tag) else None
    return 0, 0, 0


def luminance(color: tuple[int, int, int]) -> float:
    red, green, blue = color
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def inject_b1_layout_overlap(html: str, rng: random.Random) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    target: Tag | None = None

    for div in soup.find_all("div"):
        style = str(div.get("style", ""))
        classes = class_string(div)
        has_margin = bool(
            re.search(r"(?i)\bmargin(?:-(top|right|bottom|left))?\s*:", style)
            or re.search(r"\b(?:m|mt|mr|mb|ml|mx|my)-(?:auto|px|\d+)\b", classes)
        )
        if has_margin:
            target = div
            break

    if target is None:
        target = soup.find("div")
    if target is None:
        return None

    negative_margin = -rng.randint(20, 80)
    classes = class_string(target)
    margin_prop = "margin-top" if re.search(r"\bmt-", classes) else "margin"
    apply_inline_styles(target, {margin_prop: f"{negative_margin}px"})
    return str(soup)


def inject_b2_text_overflow(html: str, rng: random.Random) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    target = find_first_text_container(soup, min_length=24)
    if target is None:
        return None

    factor = rng.uniform(0.30, 0.60)
    style = str(target.get("style", ""))
    original_width = get_style_property(style, "width") or get_style_property(style, "max-width")
    shrunk_width = scale_css_dimension(original_width, factor)
    if shrunk_width is None:
        shrunk_width = f"{int(round(factor * 100))}%"

    apply_inline_styles(
        target,
        {
            "display": "block",
            "width": shrunk_width,
            "max-width": shrunk_width,
        },
    )
    return str(soup)


def extract_z_index(tag: Tag, fallback: int) -> int:
    style = str(tag.get("style", ""))
    match = re.search(r"(?i)\bz-index\s*:\s*(-?\d+)", style)
    if match:
        return int(match.group(1))

    classes = class_string(tag)
    match = re.search(r"\bz-(\d+)\b", classes)
    if match:
        return int(match.group(1))
    match = re.search(r"\bz-\[(\d+)\]\b", classes)
    if match:
        return int(match.group(1))
    return fallback


def inject_b3_z_index_collision(html: str, rng: random.Random) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    positioned_elements: list[Tag] = []

    for tag in soup.find_all(True):
        style = str(tag.get("style", ""))
        classes = class_string(tag)
        if re.search(r"(?i)\bposition\s*:\s*(relative|absolute|fixed|sticky)\b", style):
            positioned_elements.append(tag)
            continue
        if re.search(r"\b(relative|absolute|fixed|sticky)\b", classes):
            positioned_elements.append(tag)

    if len(positioned_elements) < 2:
        fallback = soup.find_all(["header", "nav", "section", "main", "div"], limit=3)
        positioned_elements = [tag for tag in fallback if isinstance(tag, Tag)]
        for tag in positioned_elements:
            apply_inline_styles(tag, {"position": "relative"})

    if len(positioned_elements) < 2:
        return None

    z_values = [extract_z_index(tag, (index + 1) * 10) for index, tag in enumerate(positioned_elements)]
    reassigned = z_values[:]
    rng.shuffle(reassigned)
    if len(reassigned) > 1 and reassigned == z_values:
        reassigned = reassigned[1:] + reassigned[:1]

    for tag, z_value in zip(positioned_elements, reassigned):
        apply_inline_styles(tag, {"z-index": str(z_value)})

    if len(positioned_elements) >= 2:
        overlap_target = positioned_elements[1]
        apply_inline_styles(
            overlap_target,
            {
                "top": f"-{rng.randint(24, 64)}px",
                "margin-bottom": f"-{rng.randint(12, 36)}px",
            },
        )

    return str(soup)


def inject_b4_truncation(html: str, rng: random.Random) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    target = find_first_text_container(soup, min_length=8)
    if target is None:
        return None

    apply_inline_styles(
        target,
        {
            "display": "inline-block",
            "overflow": "hidden",
            "white-space": "nowrap",
            "text-overflow": "clip",
            "max-width": f"{rng.randint(48, 96)}px",
        },
    )
    return str(soup)


def inject_b5_color_contrast(html: str, rng: random.Random) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    target = find_first_colored_text_container(soup, min_length=4)
    if target is None:
        return None

    background = get_background_color(target)
    background_luminance = luminance(background)
    delta = rng.randint(24, 40)
    shifted = background_luminance - delta if background_luminance >= 128 else background_luminance + delta
    shifted = max(0, min(int(round(shifted)), 255))
    new_color = f"rgb({shifted}, {shifted}, {shifted})"

    targets = [target]
    parent = target.parent if isinstance(target.parent, Tag) else None
    if isinstance(parent, Tag):
        sibling_matches = [
            tag for tag in parent.find_all(target.name, recursive=False) if clean_text(tag)
        ]
        if len(sibling_matches) > 1:
            targets = sibling_matches[: min(3, len(sibling_matches))]

    for styled_target in targets:
        apply_inline_styles(styled_target, {"color": new_color})
    return str(soup)


BUG_INJECTORS: dict[str, Callable[[str, random.Random], str | None]] = bug_injector_callables()


async def render_html(
    page: Any,
    html: str,
    output_path: Path,
    timeout_ms: int,
) -> None:
    await page.goto("about:blank", wait_until="load", timeout=timeout_ms)
    await page.set_content(html, wait_until="load", timeout=timeout_ms)
    try:
        await page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 2_000))
    except PlaywrightTimeoutError:
        pass
    await page.wait_for_timeout(250)
    await page.screenshot(path=str(output_path), full_page=False, timeout=timeout_ms)


def compute_vss(clean_image_path: Path, buggy_image_path: Path) -> float:
    clean_image = np.asarray(Image.open(clean_image_path).convert("RGB"), dtype=np.uint8)
    buggy_image = np.asarray(Image.open(buggy_image_path).convert("RGB"), dtype=np.uint8)

    height = min(clean_image.shape[0], buggy_image.shape[0])
    width = min(clean_image.shape[1], buggy_image.shape[1])
    clean_image = clean_image[:height, :width]
    buggy_image = buggy_image[:height, :width]

    diff = np.abs(clean_image.astype(np.int16) - buggy_image.astype(np.int16))
    changed_mask = np.any(diff > 8, axis=-1)
    buggy_pixel_ratio = float(changed_mask.sum() / changed_mask.size)

    clean_gray = np.asarray(Image.fromarray(clean_image).convert("L"), dtype=np.uint8)
    buggy_gray = np.asarray(Image.fromarray(buggy_image).convert("L"), dtype=np.uint8)
    ssim_score = float(structural_similarity(clean_gray, buggy_gray, data_range=255))

    return 0.5 * buggy_pixel_ratio + 0.5 * (1.0 - ssim_score)


async def process_sample(
    sample: dict[str, Any],
    index: int,
    page: Any,
    args: argparse.Namespace,
    paths: dict[str, Path],
    metadata_keys: set[tuple[str, str]],
    logger: logging.Logger,
) -> dict[str, Any]:
    sample_id = sample_id_for(index)
    timeout_ms = int(args.timeout_seconds * 1000)
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    assigned_bug_types = pick_bug_types_for_sample(index, args)

    try:
        clean_html = extract_html(sample)
    except Exception as exc:  # noqa: BLE001
        message = f"{sample_id} | extraction failed | {type(exc).__name__}: {exc}"
        logger.error(message)
        return {
            "index": index,
            "sample_id": sample_id,
            "assigned_bug_types": assigned_bug_types,
            "clean_rendered": False,
            "rows_written": 0,
            "errors": [message],
            "processed_at": now_iso(),
        }

    clean_html_path = paths["html"] / f"{sample_id}_clean.html"
    clean_image_path = paths["screenshots"] / f"{sample_id}_clean.png"
    clean_html_path.write_text(clean_html, encoding="utf-8")

    clean_key = (sample_id, "None")
    clean_rendered = clean_key in metadata_keys and clean_image_path.exists()

    if not clean_rendered:
        try:
            await render_html(page, clean_html, clean_image_path, timeout_ms=timeout_ms)
            rows.append(
                {
                    "sample_id": sample_id,
                    "image_path": relative_path_str(clean_image_path),
                    "label": 0,
                    "bug_type": None,
                    "vss_score": None,
                }
            )
            clean_rendered = True
        except (PlaywrightTimeoutError, PlaywrightError, Exception) as exc:  # noqa: BLE001
            message = f"{sample_id} | clean render failed | {type(exc).__name__}: {exc}"
            logger.error(message)
            errors.append(message)
            return {
                "index": index,
                "sample_id": sample_id,
                "assigned_bug_types": assigned_bug_types,
                "clean_rendered": False,
                "rows_written": 0,
                "errors": errors,
                "processed_at": now_iso(),
            }

    for bug_type in assigned_bug_types:
        bug_html_path = paths["html"] / f"{sample_id}_{bug_type}.html"
        bug_image_path = paths["screenshots"] / f"{sample_id}_{bug_type}.png"
        bug_key = (sample_id, bug_type)

        if bug_key in metadata_keys and bug_image_path.exists():
            continue

        rng = rng_for_sample(args.seed, index, bug_type)
        injected_html = BUG_INJECTORS[bug_type](clean_html, rng)
        if not injected_html:
            message = f"{sample_id} | {bug_type} injection failed"
            logger.error(message)
            errors.append(message)
            continue

        bug_html_path.write_text(injected_html, encoding="utf-8")

        try:
            await render_html(page, injected_html, bug_image_path, timeout_ms=timeout_ms)
            vss_score = compute_vss(clean_image_path, bug_image_path)
            rows.append(
                {
                    "sample_id": sample_id,
                    "image_path": relative_path_str(bug_image_path),
                    "label": 1,
                    "bug_type": bug_type,
                    "vss_score": round(vss_score, 6),
                }
            )
        except (PlaywrightTimeoutError, PlaywrightError, Exception) as exc:  # noqa: BLE001
            message = f"{sample_id} | {bug_type} render failed | {type(exc).__name__}: {exc}"
            logger.error(message)
            errors.append(message)

    rows_written = append_metadata_rows(paths["metadata"], rows, metadata_keys)
    return {
        "index": index,
        "sample_id": sample_id,
        "assigned_bug_types": assigned_bug_types,
        "clean_rendered": clean_rendered,
        "rows_written": rows_written,
        "errors": errors,
        "processed_at": now_iso(),
    }


async def build_dataset(args: argparse.Namespace) -> None:
    paths = ensure_output_dirs(args.output_dir)
    logger = setup_logging(paths["logs"])
    processed_indices = load_processed_indices(paths["status"])
    metadata_keys = dedupe_metadata(paths["metadata"])

    total_existing = sum(1 for index in processed_indices if index < args.max_samples)
    progress = tqdm(total=args.max_samples, initial=total_existing, desc="Samples", unit="sample")

    dataset_stream = load_dataset(args.dataset, split=args.split, streaming=True)
    dataset_iterator = iter(dataset_stream)

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": args.viewport_width, "height": args.viewport_height},
            device_scale_factor=1,
        )
        page = await context.new_page()

        processed_since_checkpoint = 0
        try:
            for index in range(args.max_samples):
                try:
                    sample = next(dataset_iterator)
                except StopIteration:
                    break

                if index in processed_indices:
                    continue

                status_payload = await process_sample(
                    sample=sample,
                    index=index,
                    page=page,
                    args=args,
                    paths=paths,
                    metadata_keys=metadata_keys,
                    logger=logger,
                )
                append_status(paths["status"], status_payload)
                processed_indices.add(index)
                processed_since_checkpoint += 1
                progress.update(1)

                if processed_since_checkpoint >= 100:
                    stats = write_dataset_stats(paths["metadata"], paths["stats"])
                    write_checkpoint(paths["checkpoint"], args, processed_indices, stats)
                    processed_since_checkpoint = 0

            stats = write_dataset_stats(paths["metadata"], paths["stats"])
            write_checkpoint(paths["checkpoint"], args, processed_indices, stats)
        finally:
            await page.close()
            await context.close()
            await browser.close()
            close_iterator = getattr(dataset_iterator, "close", None)
            if callable(close_iterator):
                close_iterator()
            progress.close()


def validate_args(args: argparse.Namespace) -> None:
    if args.n_samples is not None:
        if args.n_samples <= 0:
            raise ValueError("--n_samples must be positive.")
        if args.all_bugs:
            args.max_samples = max(1, args.n_samples // (len(BUG_TYPES) + 1))
        else:
            bug_total = args.samples_per_bug * len(BUG_TYPES)
            clean_total = args.n_samples - bug_total
            if clean_total <= 0:
                raise ValueError(
                    "--n_samples is too small for the requested balanced bug budget. "
                    "Increase --n_samples or lower --samples-per-bug."
                )
            args.max_samples = clean_total

    if args.max_samples <= 0:
        raise ValueError("--max-samples must be positive.")
    if args.samples_per_bug <= 0:
        raise ValueError("--samples-per-bug must be positive.")
    if not args.all_bugs and args.samples_per_bug * len(BUG_TYPES) > args.max_samples:
        raise ValueError(
            "--samples-per-bug * 5 exceeds --max-samples. "
            "Use a smaller balanced target or pass --all-bugs."
        )


def main() -> None:
    args = parse_args()
    validate_args(args)
    asyncio.run(build_dataset(args))


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except Exception:  # noqa: BLE001
        exit_code = 1
        traceback.print_exc()
    finally:
        logging.shutdown()
        sys.stdout.flush()
        sys.stderr.flush()
        # The HuggingFace streaming stack can leave background threads alive after
        # all work has completed. Force process termination after flushing outputs.
        os._exit(exit_code)

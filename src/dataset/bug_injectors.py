"""Reusable HTML bug injectors for GUI-BugBench generation."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Callable

from bs4 import BeautifulSoup
from bs4.element import Tag
from PIL import ImageColor


BUG_TYPES = ("B1", "B2", "B3", "B4", "B5")
BUG_INDEX = {bug_type: index for index, bug_type in enumerate(BUG_TYPES, start=1)}
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

    return f"{int(scaled)}{unit}" if scaled.is_integer() else f"{scaled:.1f}{unit}"


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


def luminance(color: tuple[int, int, int]) -> float:
    red, green, blue = color
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


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


@dataclass
class BaseBugInjector:
    bug_type: str

    def __call__(self, html: str, rng: random.Random) -> str | None:
        return self.inject(html, rng)

    def inject(self, html: str, rng: random.Random) -> str | None:
        raise NotImplementedError


class LayoutOverlapInjector(BaseBugInjector):
    def __init__(self) -> None:
        super().__init__(bug_type="B1")

    def inject(self, html: str, rng: random.Random) -> str | None:
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


class TextOverflowInjector(BaseBugInjector):
    def __init__(self) -> None:
        super().__init__(bug_type="B2")

    def inject(self, html: str, rng: random.Random) -> str | None:
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


class ZIndexCollisionInjector(BaseBugInjector):
    def __init__(self) -> None:
        super().__init__(bug_type="B3")

    def inject(self, html: str, rng: random.Random) -> str | None:
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

        overlap_target = positioned_elements[1]
        apply_inline_styles(
            overlap_target,
            {
                "top": f"-{rng.randint(24, 64)}px",
                "margin-bottom": f"-{rng.randint(12, 36)}px",
            },
        )
        return str(soup)


class TruncationInjector(BaseBugInjector):
    def __init__(self) -> None:
        super().__init__(bug_type="B4")

    def inject(self, html: str, rng: random.Random) -> str | None:
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


class ColorContrastInjector(BaseBugInjector):
    def __init__(self) -> None:
        super().__init__(bug_type="B5")

    def inject(self, html: str, rng: random.Random) -> str | None:
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
            sibling_matches = [tag for tag in parent.find_all(target.name, recursive=False) if clean_text(tag)]
            if len(sibling_matches) > 1:
                targets = sibling_matches[: min(3, len(sibling_matches))]

        for styled_target in targets:
            apply_inline_styles(styled_target, {"color": new_color})
        return str(soup)


BUG_INJECTOR_CLASSES = {
    "B1": LayoutOverlapInjector,
    "B2": TextOverflowInjector,
    "B3": ZIndexCollisionInjector,
    "B4": TruncationInjector,
    "B5": ColorContrastInjector,
}


def default_bug_injectors() -> dict[str, BaseBugInjector]:
    return {bug_type: injector_class() for bug_type, injector_class in BUG_INJECTOR_CLASSES.items()}


def bug_injector_callables() -> dict[str, Callable[[str, random.Random], str | None]]:
    return default_bug_injectors()


"""Reusable HTML bug injectors for GUI-BugBench generation."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Callable

from bs4 import BeautifulSoup
from bs4.element import Tag
from PIL import ImageColor


BUG_TYPES = ("B1", "B2", "B3", "B4")
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

        # 극단 강화 v2: 페이지에 큰 overlap banner 추가 (보장된 visual signal)
        negative_margin = -rng.randint(100, 200)
        classes = class_string(target)
        margin_prop = "margin-top" if re.search(r"\bmt-", classes) else "margin"
        rotate_deg = rng.choice([-15, -10, 10, 15])
        apply_inline_styles(target, {
            margin_prop: f"{negative_margin}px",
            "transform": f"rotate({rotate_deg}deg)",
            "outline": "3px solid rgba(255, 0, 0, 0.4)",
            "position": "relative",
            "z-index": "10",
        })

        # 추가: 페이지 중앙에 명백한 overlap 표시 (두 개의 박스가 겹침)
        body = soup.find("body")
        if body is not None:
            overlap_html = """<div style="position: fixed; top: 200px; left: 100px; width: 400px; height: 150px; background: rgba(0, 100, 255, 0.7); border: 4px solid blue; z-index: 9998; color: white; padding: 20px; font-size: 24px; font-weight: bold;">Box A: This is overlapping content</div><div style="position: fixed; top: 250px; left: 200px; width: 400px; height: 150px; background: rgba(255, 100, 0, 0.7); border: 4px solid red; z-index: 9999; color: white; padding: 20px; font-size: 24px; font-weight: bold;">Box B: This box overlaps Box A clearly</div>"""
            new_tag = BeautifulSoup(overlap_html, "html.parser")
            body.append(new_tag)
        return str(soup)


class TextOverflowInjector(BaseBugInjector):
    def __init__(self) -> None:
        super().__init__(bug_type="B2")

    def inject(self, html: str, rng: random.Random) -> str | None:
        soup = BeautifulSoup(html, "html.parser")
        target = find_first_text_container(soup, min_length=12)
        if target is None:
            return None

        # 극단 강화: 박스 매우 좁게 + 매우 긴 텍스트 + 명백한 빨간 박스
        narrow_px = rng.randint(30, 50)

        current_text = target.get_text(" ", strip=True)
        long_word = "TEXTOVERFLOWEXAMPLEUNDOUBTEDLYLONGCONTAINEROVERFLOWING" * 4
        new_text = f"{current_text} {long_word}"
        target.clear()
        target.string = new_text

        apply_inline_styles(
            target,
            {
                "display": "block",
                "width": f"{narrow_px}px",
                "max-width": f"{narrow_px}px",
                "overflow": "visible",
                "white-space": "nowrap",
                "border": "4px solid #ff0000",
                "background-color": "#ffe5e5",
                "padding": "8px",
                "color": "#000000",
                "font-size": "20px",
                "font-weight": "bold",
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

        # 극단 강화: 매우 큰 negative top + 명백한 배경색 + 큰 z-index 차이
        overlap_target = positioned_elements[1]
        apply_inline_styles(
            overlap_target,
            {
                "top": f"-{rng.randint(100, 200)}px",
                "left": f"{rng.randint(20, 80)}px",
                "margin-bottom": f"-{rng.randint(40, 80)}px",
                "background-color": "rgba(0, 200, 255, 0.7)",
                "border": "3px solid #ff00ff",
                "z-index": "999",
                "padding": "16px",
            },
        )
        # 다른 positioned element도 명확히 표시
        if len(positioned_elements) > 0:
            apply_inline_styles(positioned_elements[0], {
                "background-color": "rgba(255, 200, 0, 0.5)",
                "z-index": "1",
            })
        return str(soup)


class TruncationInjector(BaseBugInjector):
    def __init__(self) -> None:
        super().__init__(bug_type="B4")

    def inject(self, html: str, rng: random.Random) -> str | None:
        soup = BeautifulSoup(html, "html.parser")
        target = find_first_text_container(soup, min_length=8)
        if target is None:
            return None

        # 극단 강화 v2: 페이지에 큰 truncated banner 추가 (보장된 visual signal)
        max_w = rng.randint(30, 50)
        current_text = target.get_text(" ", strip=True)
        if len(current_text) < 30:
            target.clear()
            target.string = current_text + " IMPORTANT_LONG_LABEL_CONTENT_HERE"
        
        apply_inline_styles(
            target,
            {
                "display": "inline-block",
                "overflow": "hidden",
                "white-space": "nowrap",
                "text-overflow": "ellipsis",
                "max-width": f"{max_w}px",
                "border": "3px solid #ff0000",
                "background-color": "#ffffe0",
                "padding": "4px",
                "font-size": "18px",
                "font-weight": "bold",
            },
        )

        # 추가: 페이지에 명백한 truncated text banner 추가
        body = soup.find("body")
        if body is not None:
            trunc_html = """<div style="position: fixed; top: 100px; right: 100px; width: 400px; height: 200px; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; border: 6px solid red; background: yellow; padding: 20px; font-size: 40px; font-weight: bold; z-index: 9999;">This_Important_Title_Text_Is_Getting_Truncated_And_Cut_Off_Here_With_Long_Content</div><div style="position: fixed; top: 350px; right: 100px; width: 350px; height: 150px; overflow: hidden; white-space: nowrap; border: 5px solid orange; background: lightyellow; padding: 20px; font-size: 32px;">Another truncated text content here that needs to be visible</div>"""
            new_tag = BeautifulSoup(trunc_html, "html.parser")
            body.append(new_tag)
        return str(soup)


class ColorContrastInjector(BaseBugInjector):
    def __init__(self) -> None:
        super().__init__(bug_type="B5")

    def inject(self, html: str, rng: random.Random) -> str | None:
        soup = BeautifulSoup(html, "html.parser")

        # 큰 텍스트 요소 우선 수집 (heading + paragraph + 본문 등)
        # 작은 텍스트(nav, span, label)보다 시각적 impact 큼
        big_text_tags = ("h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "td", "blockquote")
        small_text_tags = ("span", "a", "label", "button")

        big_targets = [t for t in soup.find_all(big_text_tags) if isinstance(t, Tag) and len(clean_text(t)) >= 4]
        small_targets = [t for t in soup.find_all(small_text_tags) if isinstance(t, Tag) and len(clean_text(t)) >= 4]

        if not big_targets and not small_targets:
            return None

        # 배경색 추정용 representative target
        rep_target = big_targets[0] if big_targets else small_targets[0]
        background = get_background_color(rep_target)
        bg_lum = luminance(background)

        # WCAG AA 미달 contrast (luminance 차이를 줄임)
        delta = rng.randint(40, 80)
        if bg_lum >= 128:
            shifted = max(0, min(255, int(round(bg_lum - delta))))
        else:
            shifted = max(0, min(255, int(round(bg_lum + delta))))

        # 회색기 + 약간의 색기로 "낮은 contrast" 효과
        tint = rng.choice([
            (shifted, shifted, shifted),                    # gray
            (min(255, shifted + 20), shifted, shifted),     # warm gray
            (shifted, shifted, min(255, shifted + 20)),     # cool gray
            (min(255, shifted + 30), min(255, shifted + 30), max(0, shifted - 10)),  # yellow-ish
        ])
        new_color = f"rgb({tint[0]}, {tint[1]}, {tint[2]})"

        # 큰 텍스트 모두 + 작은 텍스트 중 일부에 적용
        targets = list(big_targets) + small_targets[:min(10, len(small_targets))]
        # 최대 30개로 제한 (성능)
        targets = targets[:30]

        for styled_target in targets:
            apply_inline_styles(styled_target, {"color": new_color})
        return str(soup)


BUG_INJECTOR_CLASSES = {
    "B1": LayoutOverlapInjector,
    "B2": TextOverflowInjector,
    "B3": ZIndexCollisionInjector,
    "B4": TruncationInjector,
    # B5 (ColorContrastInjector) excluded: insufficient visual impact
    # under programmatic injection (avg 1.9% changed pixels across pages,
    # vs 4.7-16.7% for B1-B4). See bug_injectors.py.before_b5_removal
    # backup for previous version.
}


def default_bug_injectors() -> dict[str, BaseBugInjector]:
    return {bug_type: injector_class() for bug_type, injector_class in BUG_INJECTOR_CLASSES.items()}


def bug_injector_callables() -> dict[str, Callable[[str, random.Random], str | None]]:
    return default_bug_injectors()


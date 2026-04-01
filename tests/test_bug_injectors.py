from __future__ import annotations

import hashlib
import random
import re
import sys
from pathlib import Path

import numpy as np
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset.bug_injectors import default_bug_injectors


SAMPLE_HTML = """
<html>
  <body style="background-color: rgb(255, 255, 255);">
    <div style="margin: 24px; position: relative; z-index: 10; background: #f3f4f6;">
      <h1 style="color: rgb(30, 30, 30);">Orders Dashboard</h1>
      <p style="width: 360px; color: rgb(50, 50, 50);">
        Quarterly revenue summaries and fulfillment metrics for the operations team.
      </p>
      <div style="position: absolute; z-index: 20; top: 12px; left: 32px;">
        <span style="color: rgb(0, 0, 0);">Status</span>
      </div>
      <button style="margin-top: 12px; color: rgb(20, 20, 20);">Review exceptions</button>
    </div>
  </body>
</html>
"""


def html_signature_image(html: str) -> np.ndarray:
    soup = BeautifulSoup(html, "html.parser")
    image = Image.new("L", (260, 180), 255)
    draw = ImageDraw.Draw(image)

    y = 6
    for tag in soup.find_all(True)[:20]:
        text = re.sub(r"\s+", " ", tag.get_text(" ", strip=True)).strip()
        style = str(tag.get("style", ""))
        classes = " ".join(tag.get("class", []))
        digest = hashlib.sha256(f"{tag.name}|{text}|{style}|{classes}".encode("utf-8")).digest()
        for column, value in enumerate(digest[:24]):
            shade = 255 - value
            draw.rectangle((8 + column * 10, y, 15 + column * 10, y + 6), fill=shade)
        y += 8
    return np.asarray(image, dtype=np.uint8)


def test_each_injector_produces_valid_html_and_changes_visual_signature() -> None:
    clean_signature = html_signature_image(SAMPLE_HTML)

    for bug_type, injector in default_bug_injectors().items():
        buggy_html = injector.inject(SAMPLE_HTML, random.Random(123))
        assert buggy_html is not None, f"{bug_type} returned no HTML"

        soup = BeautifulSoup(buggy_html, "html.parser")
        assert soup.find() is not None, f"{bug_type} did not produce valid parseable HTML"

        buggy_signature = html_signature_image(buggy_html)
        score = structural_similarity(clean_signature, buggy_signature, data_range=255)
        assert score < 0.99, f"{bug_type} did not produce a meaningful visual change: SSIM={score:.4f}"


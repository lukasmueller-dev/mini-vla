# The model's-eye view renderer. Python port of paintSilhouette in
# js/src/scene.ts (the ONLY renderer that is part of the data pipeline — the
# styled `paintScene` display renderer is portfolio-render and stays JS-side).
#
# White background, the layout's colored blocks at their positions, a light/thin
# grey arm. Drawn at renderSize then averaged down to imgSize — drawn at target
# size directly the sub-pixel arm strokes alias away (the "draw big, average
# down" antialiasing the browser canvas relied on; here a supersampled PIL draw +
# a bilinear resize). Training samples and live rollout inference BOTH go through
# this exact renderer, so the policy never sees a distribution it wasn't trained
# on.
#
# Parity note: the browser demo retrains from scratch, so this needs FUNCTIONAL
# parity with the canvas (same geometry, same relative appearance), not
# byte-identical pixels.

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

from .config import CONFIG
from .geometry import BASE, BLOCK, L1, L2
from .task import COLORS, Layout

# Blocks render a touch larger in the model's-eye view (see silBlockScale).
_SIL_BLOCK_SCALE = CONFIG.render.silBlockScale


def _scene_map(size: int):
    """Isotropic y-up workspace → y-down pixel map, mirroring sceneMap()."""
    s = CONFIG.render.sceneScale * size
    floor_y = CONFIG.render.floorY * size

    def x_of(x: float) -> float:
        return size * 0.5 + (x - 0.5) * s

    def y_of(y: float) -> float:
        return floor_y - y * s

    return x_of, y_of, s, floor_y


def _round_line(draw: ImageDraw.ImageDraw, x0, y0, x1, y1, width, fill) -> None:
    """A line with round caps (canvas lineCap='round') — PIL lines are butt-capped,
    so cap each end with a filled disk of the line's radius."""
    draw.line([x0, y0, x1, y1], fill=fill, width=max(1, round(width)))
    r = width / 2
    for cx, cy in ((x0, y0), (x1, y1)):
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)


def render_silhouette(
    a1: float,
    a2: float,
    layout: Layout,
    carry: Optional[int] = None,
    render_size: Optional[int] = None,
    img_size: Optional[int] = None,
) -> np.ndarray:
    """Render one state; returns an (img_size, img_size, 3) uint8 RGB array.
    `carry` (a COLORS index) draws that block at the effector instead of its rest
    spot — the carry-phase state cue. The trainer inverts (1 − px/255) before
    feeding the model."""
    render_size = render_size if render_size is not None else CONFIG.trainer.renderSize
    img_size = img_size if img_size is not None else CONFIG.model.imgSize
    size = render_size
    x_of, y_of, s, floor_y = _scene_map(size)

    bx, by = x_of(BASE[0]), y_of(BASE[1])
    j1x = x_of(BASE[0] + math.cos(a1) * L1)
    j1y = y_of(BASE[1] + math.sin(a1) * L1)
    ex = x_of(BASE[0] + math.cos(a1) * L1 + math.cos(a1 + a2) * L2)
    ey = y_of(BASE[1] + math.sin(a1) * L1 + math.sin(a1 + a2) * L2)

    img = Image.new("RGB", (size, size), "#ffffff")
    draw = ImageDraw.Draw(img)

    # blocks at their rest spots — the carried one leaves its spot and is redrawn
    # at the gripper. Each block draws at its own randomized side length × the
    # model-view boost.
    for b in layout:
        if b.color == carry:
            continue
        box = b.size * _SIL_BLOCK_SCALE * s
        rest = (b.y or 0.0) * s
        x0 = x_of(b.x) - box / 2
        y0 = floor_y - rest - box
        draw.rectangle([x0, y0, x0 + box, y0 + box], fill=COLORS[b.color].hex)
    if carry is not None:
        carried = next((b for b in layout if b.color == carry), None)
        box = (carried.size if carried else BLOCK) * _SIL_BLOCK_SCALE * s
        draw.rectangle([ex - box / 2, ey - box / 2, ex + box / 2, ey + box / 2],
                       fill=COLORS[carry].hex)

    # The robot body — kept in-frame (real VLAs see their own mount) but LIGHT and
    # THIN so the two color blocks stay the most salient thing in the input.
    # base pedestal (shoulder joint down to the floor) + foot
    draw.rectangle([bx - size * 0.02, by, bx + size * 0.02, floor_y], fill="#bcbcbc")
    draw.rectangle(
        [bx - size * 0.06, floor_y - size * 0.03, bx + size * 0.06, floor_y],
        fill="#bcbcbc",
    )
    # links (two grey tones), round caps
    _round_line(draw, bx, by, j1x, j1y, size * 0.028, "#a8a8a8")
    _round_line(draw, j1x, j1y, ex, ey, size * 0.022, "#c2c2c2")
    # effector: a small light locator dot, not the loudest feature
    r = size * 0.03
    draw.ellipse([ex - r, ey - r, ex + r, ey + r], fill="#8f8f8f")

    if img_size != size:
        img = img.resize((img_size, img_size), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def to_model_input(rgb: np.ndarray) -> np.ndarray:
    """INVERTED intensities (background 0, content sparse positive) — fed raw the
    near-all-white image saturates the conv branch and the model collapses onto
    language-only predictions. Returns float32 in [0, 1]."""
    return 1.0 - rgb.astype(np.float32) / 255.0

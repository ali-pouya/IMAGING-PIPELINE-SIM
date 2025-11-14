"""
scene_generator.py — classic synthetic scenes for imaging tests (float32 [0..1])

WHAT THIS MODULE PROVIDES
-------------------------
Lightweight generators for standard test targets used in imaging pipelines:
  • Slanted edge           — for edge-spread (ESF) / line-spread (LSF) and MTF work
  • 1-D barcode strip      — alternating bars; useful for contrast/SNR intuition
  • Grayscale gradient     — tone/quantization sanity checks & histograms
  • Siemens star           — radial frequency sweep; quick visual MTF probe
  • Checkerboard           — strong contrast; aliasing/quantization demos

RETURNS
-------
All functions return a 2-D NumPy array, dtype float32, normalized to [0, 1].

REFERENCES (short list)
-----------------------
• ISO 12233:2017 — Electronic still picture imaging (SFR / Slanted-edge method).
• Burns, P. D. (2000). Slanted-edge MTF for digital camera and scanner analysis.
• Siemens star usage is widespread in camera QA; see vendor app notes (Imatest, etc.).

© 2023 Ali Pouya — Imaging Pipeline (classic edition)
"""

from __future__ import annotations
import numpy as np


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _normalize(img: np.ndarray) -> np.ndarray:
    """
    Convert to float32 and clamp to [0, 1].

    Rationale:
    • Keeps the pipeline numerically stable.
    • Downstream optics/sensor modules expect float32 in [0, 1].
    """
    img = img.astype(np.float32, copy=False)
    return np.clip(img, 0.0, 1.0)


# -----------------------------------------------------------------------------
# Scene generators
# -----------------------------------------------------------------------------
def generate_slanted_edge(
    size: int = 512,
    angle_deg: float = 5.0,
    threshold: float = 0.0,
) -> np.ndarray:
    """
    Slanted binary edge (classic ISO-12233-style test primitive).

    Parameters
    ----------
    size : int
        Square canvas size (pixels).
    angle_deg : float
        Edge angle (0 = vertical). Small angles (≈ 3–7°) help sub-pixel sampling.
    threshold : float
        Offset added before the half-plane cut; useful to shift the edge.

    Returns
    -------
    edge_img : (size, size) float32 array in [0, 1]

    Notes
    -----
    • The slant ensures the edge traverses multiple pixel phases, enabling
      accurate ESF/LSF estimation (ISO-12233 SFR method).
    """
    h = w = int(size)
    xv, yv = np.meshgrid(np.arange(w), np.arange(h))
    # Signed distance-like measure relative to a slanted line:
    ramp = (xv * np.cos(np.deg2rad(angle_deg)) + yv * np.sin(np.deg2rad(angle_deg))) - threshold
    edge = (ramp > w // 2).astype(np.float32)
    return _normalize(edge)


def generate_barcode_scene(
    width: int = 512,
    height: int = 128,
    stripe_width: int = 8,
) -> np.ndarray:
    """
    Horizontal strip of alternating vertical bars (1-D barcode surrogate).

    Parameters
    ----------
    width : int
        Number of columns (pixels).
    height : int
        Number of rows (pixels).
    stripe_width : int
        Width of each bar (pixels). Smaller => higher spatial frequency.

    Returns
    -------
    bars_img : (height, width) float32 array in [0, 1]
    """
    x = np.arange(int(width))
    # 0/1 bars with period = 2*stripe_width
    bars_01 = ((x // max(int(stripe_width), 1)) % 2).astype(np.float32)
    img = np.tile(bars_01, (int(height), 1))
    return _normalize(img)


def generate_gradient_scene(
    width: int = 512,
    height: int = 512,
    horizontal: bool = True,
) -> np.ndarray:
    """
    Unit-gradient for tone/quantization tests.

    Parameters
    ----------
    width, height : int
        Output size.
    horizontal : bool
        If True, gradient increases along +x; else along +y.

    Returns
    -------
    grad_img : (height, width) float32 array in [0, 1]
    """
    if horizontal:
        grad = np.tile(np.linspace(0, 1, int(width), dtype=np.float32), (int(height), 1))
    else:
        grad = np.tile(np.linspace(0, 1, int(height), dtype=np.float32)[:, None], (1, int(width)))
    return _normalize(grad)


def generate_siemens_star(
    size: int = 512,
    spokes: int = 55,
) -> np.ndarray:
    """
    Siemens star: alternating wedges radiating from center (radial frequency sweep).

    Parameters
    ----------
    size : int
        Square canvas size (pixels).
    spokes : int
        Number of black/white transitions around 360° (higher => finer detail).

    Returns
    -------
    star_img : (size, size) float32 array in [0, 1]

    Notes
    -----
    • Useful as a quick visual probe of MTF: where the wedges blur together,
      the system approaches its resolution limit.
    """
    h = w = int(size)
    y, x = np.indices((h, w))
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    theta = np.arctan2(y - cy, x - cx)  # [-π, π]
    # Sign of a cosine gives binary wedges; shift/scale to [0,1]
    star = 0.5 * (1.0 + np.sign(np.cos(float(spokes) * theta)))
    return _normalize(star.astype(np.float32))


def generate_checker(
    size: int = 512,
    square_px: int = 16,
    invert: bool = False,
) -> np.ndarray:
    """
    Checkerboard pattern (high contrast, aliasing/quantization demos).

    Parameters
    ----------
    size : int
        Square canvas size (pixels).
    square_px : int
        Tile size (pixels).
    invert : bool
        If True, swap black/white tiles.

    Returns
    -------
    checker_img : (size, size) float32 array in [0, 1]
    """
    h = w = int(size)
    y, x = np.indices((h, w))
    tiles = ((x // max(int(square_px), 1)) + (y // max(int(square_px), 1))) % 2
    img = 1.0 - tiles if invert else tiles
    return _normalize(img.astype(np.float32))


# -----------------------------------------------------------------------------
# Dispatcher (public API)
# -----------------------------------------------------------------------------
def generate_scene(kind: str, size: int, **kwargs) -> np.ndarray:
    """
    Dispatch scene generation by name.

    Parameters
    ----------
    kind : str
        One of: 'slanted_edge' | 'barcode' | 'gradient' | 'siemens_star' | 'checker'
        Also accepts lightweight aliases: 'edge', 'siemens', 'checkerboard'.
    size : int
        Base canvas size (pixels). Some generators (barcode) also use height.

    Returns
    -------
    img : float32 array in [0, 1]
    """
    k = (kind or "").lower().strip()

    if k in ("slanted_edge", "edge"):
        return generate_slanted_edge(
            size=int(size),
            angle_deg=float(kwargs.get("angle_deg", 5.0)),
            threshold=float(kwargs.get("threshold", 0.0)),
        )

    if k == "barcode":
        return generate_barcode_scene(
            width=int(size),
            height=int(kwargs.get("height", max(64, int(size) // 4))),
            stripe_width=int(kwargs.get("stripe_width", 8)),
        )

    if k == "gradient":
        return generate_gradient_scene(
            width=int(size),
            height=int(size),
            horizontal=bool(kwargs.get("horizontal", True)),
        )

    if k in ("siemens_star", "siemens"):
        return generate_siemens_star(
            size=int(size),
            spokes=int(kwargs.get("spokes", 55)),
        )

    if k in ("checker", "checkerboard"):
        return generate_checker(
            size=int(size),
            square_px=int(kwargs.get("square_px", 16)),
            invert=bool(kwargs.get("invert", False)),
        )

    # Fallback: gradient (safe for unknown names)
    return generate_gradient_scene(width=int(size), height=int(size))

'''
import numpy as np
import matplotlib.pyplot as plt

def generate_barcode_scene(width=512, height=128, stripe_width=8):
    """Generate a synthetic 1D barcode pattern."""
    x = np.arange(width)
    bars = ((x // stripe_width) % 2) * 255
    scene = np.tile(bars, (height, 1))
    return scene.astype(np.uint8)

def generate_gradient_scene(width=512, height=512):
    """Generate a horizontal grayscale gradient."""
    gradient = np.tile(np.linspace(0, 255, width), (height, 1))
    return gradient.astype(np.uint8)

def generate_slanted_edge(width=512, height=512, angle_deg=5):
    """Generate a slanted edge for MTF measurement."""
    xv, yv = np.meshgrid(np.arange(width), np.arange(height))
    edge = (xv * np.cos(np.deg2rad(angle_deg)) + yv * np.sin(np.deg2rad(angle_deg)))
    #return edge
    return (edge > width // 2).astype(np.uint8) * 255

def generate_siemens_star(size=512, spokes=55):
    """
    Generate a binary Siemens-star resolution target.
    The Siemens star is often used to visualize optical blur
    and evaluate MTF or PSF symmetry. It consists of alternating
    black and white wedges radiating from the center.

    Parameters
    ----------
    size : int
        Width and height of the square output image in pixels.
    spokes : int
        Number of black–white wedge pairs (so 36 gives 72 total wedges).

    Returns
    -------
    pattern : (size, size) ndarray of float
        Binary image (0 = black, 1 = white) containing the Siemens star.
    """
    # ------------------------------------------------------------
    # 1. Create a coordinate grid.
    # np.indices returns two 2-D arrays of pixel indices:
    #   y[i,j] gives the row number (vertical coordinate)
    #   x[i,j] gives the column number (horizontal coordinate)
    # We subtract size/2 to center the coordinate system at the image center
    # instead of the top-left corner.
    # ------------------------------------------------------------
    y, x = np.indices((size, size)) - size/2
    # ------------------------------------------------------------
    # 2. Compute the polar angle for each pixel.
    # arctan2(y, x) returns the angle (in radians) between the positive x-axis
    # and the line from the origin to point (x, y).
    # Range is (-pi, +pi].
    # ------------------------------------------------------------
    angle = np.arctan2(y, x)
    # ------------------------------------------------------------
    # 3. Create the alternating wedge pattern.
    # Multiply the angle by the desired number of spokes to control how many
    # times the sine function flips sign around the circle.
    #
    #   sin(spokes * angle) > 0  → white
    #   sin(spokes * angle) <= 0 → black
    #
    # The resulting boolean array is converted to float (1.0 / 0.0).
    # ------------------------------------------------------------
    pattern = (np.sin(spokes * angle) > 0).astype(float)
    # apply a circular mask to remove the square edges:
    r = np.sqrt(x**2 + y**2)
    mask = (r <= size/2)
    pattern[~mask] = 1
    #pattern[r > size/2] = 0.5  # gray background outside circle
    return pattern

def show_scene(scene, title="Scene"):
    plt.figure(figsize=(6, 3))
    plt.imshow(scene, cmap='gray', vmin=0, vmax=255)
    plt.title(title)
    plt.axis('off')
    plt.show()
'''
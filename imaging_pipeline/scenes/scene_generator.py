import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
from typing import Literal, Optional

__all__ = [
    "generate_scene",
    "generate_slanted_edge",
    "generate_barcode_scene",
    "generate_gradient_scene",
    "generate_siemens_star",
    "generate_checker_scene",
    "generate_custom_scene",
    "show_scene",
]

Kind = Literal[
    "slanted_edge",
    "barcode",
    "gradient",
    "siemens_star",
    "checker",
    "custom",
]

# --------- Helpers ---------
def _to_float01(img: np.ndarray) -> np.ndarray:
    """Ensure float32 in [0,1]. Accepts uint8-like or float arrays."""
    img = np.asarray(img)
    if img.dtype.kind in ("u", "i"):
        # Assume 0..255
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)
    # Clip for safety
    return np.clip(img, 0.0, 1.0)

def _centered_indices(h: int, w: int):
    """Return centered coordinate grids (y, x) with origin at image center."""
    y, x = np.indices((h, w))
    y = y - (h / 2.0)
    x = x - (w / 2.0)
    return y, x

def _to_gray01(img: np.ndarray) -> np.ndarray:
    """
    Convert RGB/RGBA or grayscale array to float32 in [0,1].
    Uses ITU-R BT.601 luma weights for RGB → gray.
    """
    img = np.asarray(img)
    if img.ndim == 2:
        return _to_float01(img)
    if img.ndim == 3:
        # If RGBA, drop alpha
        if img.shape[2] == 4:
            img = img[:, :, :3]
        # If already 0..1 floats, leave as-is; else scale 0..255 -> 0..1
        if img.dtype.kind in ("u", "i"):
            img = img.astype(np.float32) / 255.0
        # Luma
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        return np.clip(gray.astype(np.float32), 0.0, 1.0)
    raise ValueError("Unsupported image array shape for grayscale conversion.")

def _resize_nn(img: np.ndarray, out_size: int) -> np.ndarray:
    """
    Nearest-neighbor resize to (out_size, out_size) without extra deps.
    """
    h, w = img.shape[:2]
    ys = np.linspace(0, h - 1, out_size)
    xs = np.linspace(0, w - 1, out_size)
    yi = np.round(ys).astype(int)
    xi = np.round(xs).astype(int)
    return img[yi][:, xi]

# --------- Generators (each returns float32 in [0,1]) ---------
def generate_slanted_edge(size: int = 512, angle_deg: float = 5.0, threshold: float = 0.0) -> np.ndarray:
    """
    Slanted (tilted) binary edge for MTF measurement (ISO-12233 style input).
    The edge line is defined by: x*cosθ + y*sinθ = threshold
    Pixels with value > threshold are white (1), else black (0).
    """
    h = w = int(size)
    y, x = _centered_indices(h, w)
    t = np.deg2rad(angle_deg)
    edge_line = x * np.cos(t) + y * np.sin(t)
    img = (edge_line > threshold).astype(np.float32)
    return _to_float01(img)

def generate_barcode_scene(size: int = 512, stripe_width: int = 8, height_frac: float = 0.25) -> np.ndarray:
    """
    1D vertical barcode across width; repeated along rows (like a test strip).
    """
    w = int(size); h = int(size)
    band_h = max(1, int(round(h * height_frac)))
    x = np.arange(w)
    bars_255 = ((x // stripe_width) % 2) * 255  # 0/255 pattern across width
    band = np.tile(bars_255, (band_h, 1)).astype(np.uint8)
    img = np.zeros((h, w), dtype=np.uint8) + 255
    top = (h - band_h) // 2
    img[top:top + band_h, :] = band
    return _to_float01(img)

def generate_gradient_scene(size: int = 512, horizontal: bool = True) -> np.ndarray:
    """Grayscale gradient 0→1. Horizontal by default (left→right)."""
    w = h = int(size)
    if horizontal:
        grad = np.tile(np.linspace(0.0, 1.0, w, dtype=np.float32), (h, 1))
    else:
        grad = np.tile(np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None], (1, w))
    return _to_float01(grad)

def generate_siemens_star(size: int = 512, spokes: int = 55) -> np.ndarray:
    """Binary Siemens star (0/1) inside a circular aperture; background set to 1."""
    s = int(size)
    y, x = _centered_indices(s, s)
    angle = np.arctan2(y, x)  # (-pi, pi]
    pattern = (np.sin(spokes * angle) > 0).astype(np.float32)
    r = np.sqrt(x**2 + y**2)
    mask = (r <= s / 2.0)
    pattern[~mask] = 1.0  # white background outside circle
    return _to_float01(pattern)

def generate_checker_scene(size: int = 512, square_px: int = 16, invert: bool = False) -> np.ndarray:
    """
    Binary checkerboard. Good for aliasing/quantization tests.
    - size: output is size x size
    - square_px: size of each tile in pixels
    - invert: swap black/white
    """
    s = int(size)
    # Build index grids and compute tile indices
    y, x = np.indices((s, s))
    tiles = ((y // square_px) + (x // square_px)) % 2
    img = tiles.astype(np.float32)
    if invert:
        img = 1.0 - img
    return _to_float01(img)

def generate_custom_scene(
    path: Optional[str] = None,
    array: Optional[np.ndarray] = None,
    size: int = 512,
    keep_aspect: bool = True,
) -> np.ndarray:
    """
    Load a custom image and convert to grayscale float in [0,1].
    - Provide either 'path' to an image file or 'array' (H x W x [C]).
    - Resizes to (size, size) using nearest-neighbor (dependency-free).
    - If keep_aspect=True, centers the resized image on a white canvas.
    """
    if (path is None) and (array is None):
        raise ValueError("generate_custom_scene requires 'path' or 'array'.")

    if path is not None:
        img = mpimg.imread(path)  # returns float [0..1] or uint8
    else:
        img = array

    gray = _to_gray01(img)

    # Resize (nearest neighbor)
    h, w = gray.shape
    if keep_aspect:
        # scale to fit within size x size, preserving aspect ratio
        scale = min(size / h, size / w)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        resized = _resize_nn(gray, out_size=new_h)
        # Resize to (new_h, new_w): do row then column
        # First resize rows to new_h, then columns to new_w
        resized = resized
        # If new_w != new_h, adjust columns separately
        if new_w != new_h:
            # resize columns on the current array
            ys = np.arange(new_h)
            xs = np.linspace(0, gray.shape[1]-1, new_w)
            xi = np.round(xs).astype(int)
            resized = resized[:, xi]
        # place on white canvas
        canvas = np.ones((size, size), dtype=np.float32)
        top = (size - resized.shape[0]) // 2
        left = (size - resized.shape[1]) // 2
        canvas[top:top+resized.shape[0], left:left+resized.shape[1]] = resized
        return canvas
    else:
        return _resize_nn(gray, out_size=size)

# --------- Dispatcher ---------
def generate_scene(kind: Kind = "slanted_edge", size: int = 512, **kwargs) -> np.ndarray:
    """
    Public API used by main.py.
    Returns float32 image in [0,1].

    Parameters
    ----------
    kind : {"slanted_edge","barcode","gradient","siemens_star","checker","custom"}
    size : int
    kwargs : passed to the specific generator.
    """
    if kind == "slanted_edge":
        return generate_slanted_edge(size=size, **{k: v for k, v in kwargs.items()
                                                   if k in ("angle_deg", "threshold")})
    if kind == "barcode":
        return generate_barcode_scene(size=size, **{k: v for k, v in kwargs.items()
                                                    if k in ("stripe_width", "height_frac")})
    if kind == "gradient":
        return generate_gradient_scene(size=size, **{k: v for k, v in kwargs.items()
                                                     if k in ("horizontal",)})
    if kind == "siemens_star":
        return generate_siemens_star(size=size, **{k: v for k, v in kwargs.items()
                                                   if k in ("spokes",)})
    if kind == "checker":
        return generate_checker_scene(size=size, **{k: v for k, v in kwargs.items()
                                                    if k in ("square_px", "invert")})
    if kind == "custom":
        # For CLI use, you’d call generate_scene(kind="custom", path="...") in Python.
        # (main.py does not parse 'path' yet.)
        return generate_custom_scene(
            path=kwargs.get("path", None),
            array=kwargs.get("array", None),
            size=size,
            keep_aspect=kwargs.get("keep_aspect", True),
        )
    raise ValueError("Unknown scene kind "
                     f"'{kind}'. Valid: slanted_edge, barcode, gradient, siemens_star, checker, custom")

# --------- Quick viewer (expects 0..1) ---------
def show_scene(scene: np.ndarray, title: str = "Scene"):
    plt.figure(figsize=(5, 5))
    plt.imshow(scene, cmap="gray", vmin=0.0, vmax=1.0)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

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
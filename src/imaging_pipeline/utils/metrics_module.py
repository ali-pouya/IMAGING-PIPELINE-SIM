"""
metrics_module.py — small metrics helpers for the classic imaging pipeline

WHAT THIS MODULE PROVIDES
-------------------------
• compute_snr(img_noisy, img_ref)
    A simple, frame-wise signal-to-noise estimate using the reference
    pre-noise image as "signal" and the difference as "noise".
    Useful for quick comparisons across settings.

• mtf_from_fft(img)
    Orientation-averaged magnitude of the 2-D FFT (radial average).
    This is *not* ISO 12233 SFR (slanted-edge) MTF, but a fast, intuitive
    spectrum profile: flat near DC, rolling off at high spatial frequency.

• plot_histogram(img)
    Basic histogram of an image in [0, 1] range. Good for checking clipping,
    black level offsets, and distribution shape.

LEARNING NOTES
--------------
• True MTF measurement for cameras typically follows ISO 12233 SFR (slanted
  edge). That method estimates ESF → LSF → MTF, compensates for pixel sampling,
  and has precise rules for windowing and frequency scaling.
• The orientation-averaged FFT magnitude here is a compact *sanity check* of
  spectral content, not a standards-compliant MTF.

REFERENCES (short list)
-----------------------
• ISO 12233:2017 — Electronic still picture imaging: SFR/Resolution (slanted edge).
• Imatest / sfrplus docs — practical SFR implementation details.
• Oppenheim & Schafer — Discrete-Time Signal Processing (FFT/windowing basics).
• Goodman — Introduction to Fourier Optics (OTF/MTF background).

© 2025 Ava — Imaging Pipeline (classic edition)
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Simple SNR (frame-level)
# -----------------------------------------------------------------------------
def compute_snr(img_noisy: np.ndarray, img_ref: np.ndarray) -> float:
    """
    Compute a frame-level SNR in dB using the reference pre-noise image.

    Definition
    ----------
    SNR = 20 * log10( ||ref||_2 / ||ref - noisy||_2 )

    Parameters
    ----------
    img_noisy : ndarray
        The observed (e.g., post-sensor) image, preferably normalized to [0, 1].
    img_ref : ndarray
        The "clean" reference (e.g., optics output before noise), same shape.

    Returns
    -------
    snr_db : float
        Signal-to-noise ratio in decibels.

    Notes
    -----
    • This is not patch-based; it treats the whole frame as signal.
    • If you want photon-transfer-style SNR, use electrons and small uniform
      patches; see SensorParams + estimate_snr_patch in sensor_model.py.
    """
    ref = img_ref.astype(np.float32, copy=False)
    y = img_noisy.astype(np.float32, copy=False)

    num = np.linalg.norm(ref.ravel())
    den = np.linalg.norm((ref - y).ravel()) + 1e-12  # avoid divide-by-zero

    return float(20.0 * np.log10(num / den))


# -----------------------------------------------------------------------------
# Orientation-averaged FFT "MTF" (radial spectrum profile)
# -----------------------------------------------------------------------------
def mtf_from_fft(img: np.ndarray, nbins: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute an orientation-averaged magnitude spectrum profile.

    This function:
      1) Computes the 2-D FFT magnitude.
      2) Builds concentric annuli around the centered DC.
      3) Averages magnitude within each annulus to get a radial profile.
      4) Normalizes by the maximum to produce a curve in [0, 1].

    Parameters
    ----------
    img : ndarray
        2-D grayscale image (float recommended).
    nbins : int | None
        Number of radial bins. If None, uses min(H, W)//2.

    Returns
    -------
    f_norm : ndarray
        Nominal "normalized spatial frequency" in cycles/pixel (0..~1);
        note Nyquist ≈ 0.5 cycles/pixel.
    mtf : ndarray
        Orientation-averaged spectrum magnitude, normalized to [0, 1].

    Caveats
    -------
    • This is *not* ISO SFR. Use it for quick checks and intuition building.
    • Windowing (e.g., Hann) can reduce ringing effects if you adapt this
      function for small images; omitted here for simplicity in the classic path.
    """
    y = img.astype(np.float32, copy=False)

    # 2-D FFT magnitude centered at DC
    F = np.fft.fftshift(np.fft.fft2(y))
    mag = np.abs(F)

    h, w = mag.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.indices((h, w))
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    # Normalize radius so that Nyquist ≈ 0.5 cycles/pixel
    r_norm = r / (min(h, w) / 2.0)

    if nbins is None:
        nbins = int(min(h, w) // 2)
    nbins = max(8, int(nbins))

    edges = np.linspace(0.0, 1.0, nbins + 1)
    mtf = np.zeros(nbins, dtype=np.float32)

    for i in range(nbins):
        m = (r_norm >= edges[i]) & (r_norm < edges[i + 1])
        if m.any():
            mtf[i] = mag[m].mean()

    # Normalize to [0, 1]
    peak = float(mtf.max())
    if peak > 0:
        mtf /= peak

    # Bin centers as our x-axis
    f_norm = 0.5 * (edges[1:] + edges[:-1])

    return f_norm.astype(np.float32), mtf.astype(np.float32)


# -----------------------------------------------------------------------------
# Histogram helper
# -----------------------------------------------------------------------------
def plot_histogram(img: np.ndarray, title: str = "Histogram", bins: int = 64) -> None:
    """
    Plot a simple histogram for an image normalized to [0, 1].

    Parameters
    ----------
    img : ndarray
        2-D grayscale image, ideally in [0, 1].
    title : str
        Figure title.
    bins : int
        Number of histogram bins.
    """
    plt.figure()
    plt.hist(img.ravel(), bins=int(bins), range=(0.0, 1.0))
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.tight_layout()

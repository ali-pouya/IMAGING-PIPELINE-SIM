"""
optics_model.py — Gaussian-PSF-based camera blur model (classic)

WHAT THIS MODULE DOES
---------------------
Implements a simple but physically-motivated optical blur stage:
  • Build an energy-normalized Gaussian PSF (∑PSF = 1) so average brightness
    is conserved through convolution.
  • Map diffraction (λ, f/#) to a Gaussian sigma (σ_diff) as a surrogate for
    the Airy core, then combine with defocus/aberration spreads in quadrature.
  • Convolve the image with the PSF using FFT (efficient for moderate kernels).

WHY A GAUSSIAN?
---------------
The true incoherent PSF of a circular pupil is the Airy pattern, but its
central lobe can be approximated well by a Gaussian for system-level sims and
intuition-building. That lets us add independent blur contributors by adding
their variances (σ^2) and keep the code compact and fast.

HOW KERNEL SIZE IS CHOSEN
-------------------------
σ captures spread; ±3σ encloses ≈99.7% of the Gaussian energy. We therefore
pick kernel_size ≈ ceil(6σ) + 1 (forced odd) to contain the tails without
wasting cycles.

REFERENCES (short list)
-----------------------
• Goodman, J. W. (2017). *Introduction to Fourier Optics* (4th ed.).
  (Fourier optics, OTF/PSF relations, Gaussian surrogates)
• Born, M., & Wolf, E. (1999). *Principles of Optics* (7th ed.).
  (Diffraction & Airy pattern fundamentals)
• ISO 12233:2017. *Electronic still picture imaging — SFR/Resolution*.
  (Practical MTF conventions & pixel-aperture effects)
• Smith, W. J. (2007). *Modern Optical Engineering* (4th ed.).
  (Rule-of-thumb system blur estimates from f/# and wavelength)

© 2023 Ali Pouya — Imaging Pipeline (classic edition)
"""

from __future__ import annotations
import numpy as np
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from typing import Tuple


# =============================================================================
# Gaussian PSF fundamentals
# =============================================================================
# A circularly symmetric 2-D Gaussian intensity:
#   G(x, y) = A * exp(-(x^2 + y^2) / (2 σ^2))
# We *sample* this continuous function on a finite grid and then L1-normalize
# the discrete kernel so that psf.sum() == 1. This preserves average brightness
# under convolution (energy conservation in the sampled sense).


def gaussian_psf(size: int, sigma: float) -> np.ndarray:
    """
    Construct an L1-normalized 2-D Gaussian PSF.

    Parameters
    ----------
    size : int
        Kernel side length. Use an odd number (e.g., 21). Good default:
        size ≈ ceil(6*sigma) + 1 to contain ~±3σ in each direction.
    sigma : float
        Standard deviation in *pixels* (intensity-domain σ).

    Returns
    -------
    psf : (size, size) float32 ndarray
        Discrete PSF with psf.sum() == 1.0.
    """
    # Centered integer coordinates: for size=21 ⇒ [-10 ... +10]
    ax = np.arange(-size // 2 + 1.0, size // 2 + 1.0)
    xx, yy = np.meshgrid(ax, ax, indexing="xy")

    # Unnormalized Gaussian sample (avoid σ→0 numeric blowup with a floor)
    s2 = max(sigma, 1e-12) ** 2
    psf = np.exp(-(xx**2 + yy**2) / (2.0 * s2))

    # L1 normalization (discrete energy conservation)
    S = float(psf.sum())
    if S <= 0.0 or not np.isfinite(S):
        # Degenerate: fall back to a delta kernel to avoid NaNs
        psf = np.zeros_like(psf, dtype=np.float32)
        psf[size // 2, size // 2] = 1.0
        return psf
    return (psf / S).astype(np.float32)


def apply_psf(image: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """
    Convolve an image with a PSF using FFT (same-shape output).

    Why FFT?
    --------
    For moderate-to-large kernels, FFT-based convolution is O(N log N) and
    typically faster than direct spatial convolution. Because our PSF is
    L1-normalized, the average image brightness is preserved.
    """
    return fftconvolve(image, psf, mode="same")


def sigma_from_fnum(
    wavelength: float = 550e-9,
    fnum: float = 8.0,
    pixel_pitch: float = 3.75e-6,
) -> float:
    """
    Map diffraction blur to a Gaussian σ in *pixels* via the Airy-core surrogate.

    Rule-of-thumb derivation
    ------------------------
    The Airy central lobe can be approximated by a Gaussian with an intensity
    waist w0 ≈ 0.84 λ N. Equating exp(-2 r^2 / w0^2) with exp(-r^2 / (2σ^2))
    gives w0 = 2σ ⇒ σ ≈ 0.42 λ N. Convert to pixels by dividing by pixel_pitch.
    """
    sigma_m = 0.42 * wavelength * fnum
    return float(sigma_m / max(pixel_pitch, 1e-12))


def defocus_sigma_quadrature(
    sigma_diff: float,
    sigma_defocus: float = 0.0,
    sigma_aberr: float = 0.0,
) -> float:
    """
    Combine approximately independent blur contributors in quadrature:

        σ_eff = sqrt( σ_diff^2 + σ_defocus^2 + σ_aberr^2 + ... )

    Rationale
    ---------
    For independent Gaussian-like spreads, variances add. This is a pragmatic,
    system-level model to capture multiple small contributors succinctly.
    """
    return float(np.sqrt(sigma_diff**2 + sigma_defocus**2 + sigma_aberr**2))


def _ensure_odd(n: int) -> int:
    """Return n if odd, else n+1 (centers the impulse)."""
    return n if (n % 2 == 1) else (n + 1)


def apply_optics_blur(
    image: np.ndarray,
    *,
    # Back-compat hook: only 'gaussian' is implemented at the moment.
    model: str | None = None,
    # Physics mapping inputs (used if `sigma` is not explicitly provided)
    fnum: float = 8.0,
    wavelength: float = 550e-9,
    pixel_pitch: float = 3.75e-6,
    # Additional spreads (in *pixels*) that add in quadrature
    defocus_sigma: float = 0.0,
    aberr_sigma: float = 0.0,
    # Kernel sizing and outputs
    kernel_size: int | None = None,
    return_psf: bool = False,
    # Direct override: bypass physics mapping and use this σ (pixels)
    sigma: float | None = None,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """
    Blur a 2-D grayscale image using a Gaussian PSF.

    Parameters
    ----------
    image : ndarray
        2-D grayscale image (float preferred; uint ok but will be converted).
    model : str | None
        Back-compat: accepts "gaussian" (other strings raise ValueError).
    fnum, wavelength, pixel_pitch : float
        Used to derive σ_diff if `sigma` is not provided directly.
    defocus_sigma, aberr_sigma : float
        Additional blur contributors (pixels), combined in quadrature.
    kernel_size : int | None
        If None, we choose ceil(6σ)+1 to contain ~±3σ. Forced odd.
    return_psf : bool
        If True, return (blurred, psf) instead of just blurred.
    sigma : float | None
        Direct Gaussian σ override in pixels (skips diffraction mapping).

    Returns
    -------
    blurred : ndarray
        Blurred image (float32).
    psf : ndarray (optional)
        The PSF used (float32), if return_psf=True.

    Notes
    -----
    • Outputs are float32; caller may clip/quantize as needed.
    • This function is the classic extension point: add other PSFs and branch
      on `model` for richer aberration studies in the future.
    """
    if image.ndim != 2:
        raise ValueError("apply_optics_blur expects a 2-D grayscale image.")
    if model is not None and model.lower() != "gaussian":
        raise ValueError(f"Unsupported blur model: {model!r}. Only 'gaussian' is available.")

    # 1) Determine σ (pixels)
    if sigma is None:
        sigma_diff = sigma_from_fnum(wavelength=wavelength, fnum=fnum, pixel_pitch=pixel_pitch)
        sigma_eff = defocus_sigma_quadrature(sigma_diff, sigma_defocus=defocus_sigma, sigma_aberr=aberr_sigma)
    else:
        sigma_eff = float(sigma)

    # Degenerate: no blur
    if sigma_eff <= 0.0:
        if return_psf:
            delta = np.zeros((3, 3), dtype=np.float32); delta[1, 1] = 1.0
            return image.astype(np.float32, copy=False), delta
        return image.astype(np.float32, copy=False)

    # 2) Kernel sizing (contain ≈99.7% energy within ±3σ)
    if kernel_size is None:
        kernel_size = int(np.ceil(6.0 * sigma_eff) + 1)
    kernel_size = _ensure_odd(max(3, kernel_size))

    # 3) Build normalized PSF and convolve
    psf = gaussian_psf(kernel_size, sigma_eff)
    img_f = image.astype(np.float32, copy=False)
    blurred = apply_psf(img_f, psf)

    return (blurred, psf) if return_psf else blurred


# -----------------------------------------------------------------------------
# Optional quick-look visual (module self-test)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal slanted edge
    h = w = 256
    xv, yv = np.meshgrid(np.arange(w), np.arange(h))
    edge = (xv * np.cos(np.deg2rad(20)) + yv * np.sin(np.deg2rad(20)))
    img = (edge > w // 2).astype(np.float32)

    blurred, psf = apply_optics_blur(
        img,
        model="gaussian",
        fnum=8.0, wavelength=550e-9, pixel_pitch=3.75e-6,
        defocus_sigma=0.6, aberr_sigma=0.3,
        return_psf=True,
    )

    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    axs[0].imshow(img, cmap="gray", vmin=0, vmax=1); axs[0].set_title("Input");  axs[0].axis("off")
    axs[1].imshow(blurred, cmap="gray", vmin=0, vmax=1); axs[1].set_title("Blurred"); axs[1].axis("off")
    im = axs[2].imshow(psf, cmap="inferno"); axs[2].set_title(f"PSF (sum={psf.sum():.4f})"); axs[2].axis("off")
    fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.show()


'''
import numpy as np
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt

# -----------------------------
# Gaussian PSF fundamentals
# -----------------------------
# A circularly symmetric 2-D Gaussian intensity profile:
#   G(x,y) = A * exp(-(x^2 + y^2) / (2*sigma^2))
# In continuous space, to conserve energy (integral = 1), A = 1/(2*pi*sigma^2).
# On a discrete grid with finite size, we sample and then L1-normalize so sum(psf) == 1.
# This ensures convolution preserves average image brightness.

def gaussian_psf(size: int, sigma: float) -> np.ndarray:
    """
    Construct a discrete, L1-normalized 2-D Gaussian PSF.

    Parameters
    ----------
    size : int
        Kernel side length (odd recommended, e.g. 21). Choose so that ~3*sigma falls inside
        the half-width to capture >~99% of Gaussian energy (size ≳ 6*sigma + 1).
    sigma : float
        Standard deviation in *pixels* (intensity-domain sigma).

    Returns
    -------
    psf : (size, size) ndarray
        Normalized PSF kernel such that psf.sum() == 1.0 (energy-conserving for convolution).
    """
    # Build centered coordinates. For size=21, ax = [-10, ..., +10].
    # Centering makes (x^2 + y^2) radial distance straightforward.
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    # Evaluate the unnormalized 2-D Gaussian. This implements:
    #   exp(- (x^2 + y^2) / (2*sigma^2) )
    # We omit the continuous-space amplitude 1/(2*pi*sigma^2) here because we will
    # L1-normalize the discrete kernel next (which is better under truncation).
    psf = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))

    # --- CRUCIAL STEP: L1 normalization ---
    # Divide by the discrete sum so convolution preserves average brightness.
    # This compensates for both discrete sampling and finite-size truncation.
    psf_sum = psf.sum()
    if psf_sum == 0:
        # Degenerate case (e.g., sigma too tiny for given size) – fallback to a delta.
        psf = np.zeros_like(psf)
        psf[size // 2, size // 2] = 1.0
        return psf

    psf /= psf_sum
    return psf


def apply_psf(image: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """
    Convolve an image with a PSF using FFT-based convolution.

    Why FFT?
    --------
    For larger kernels, FFT convolution is O(N log N) and typically faster than direct
    spatial convolution. Since our PSF is already normalized (sum=1), this preserves
    the image's average intensity.

    Parameters
    ----------
    image : ndarray
        2-D grayscale image (float or uint8; if uint8, convert to float before).
    psf : ndarray
        Normalized PSF kernel (sum == 1.0).

    Returns
    -------
    blurred : ndarray
        Blurred image (same shape as input).
    """
    return fftconvolve(image, psf, mode='same')


def sigma_from_fnum(wavelength: float = 550e-9,
                    fnum: float = 8.0,
                    pixel_pitch: float = 3.75e-6) -> float:
    """
    Estimate Gaussian intensity sigma (in pixels) by matching the Airy core
    with a Gaussian approximation.

    Derivation (rule-of-thumb):
    ---------------------------
    - Airy core can be approximated by a Gaussian with waist w0 ~ 0.84 * lambda * N (intensity).
    - Using the common mapping exp(-2 r^2 / w0^2) ≡ exp(- r^2 / (2*sigma^2)) gives w0 = 2*sigma.
    - Therefore sigma ≈ 0.42 * lambda * N.
    - Convert to pixels: divide by pixel_pitch.

    Parameters
    ----------
    wavelength : float
        Wavelength in meters (e.g., 550e-9 for green).
    fnum : float
        f-number N (f/#).
    pixel_pitch : float
        Pixel size in meters.

    Returns
    -------
    sigma_pixels : float
        Diffraction-limited Gaussian sigma in *pixels*.
    """
    sigma_meters = 0.42 * wavelength * fnum
    return sigma_meters / pixel_pitch


def defocus_sigma_quadrature(sigma_diff: float,
                             sigma_defocus: float = 0.0,
                             sigma_aberr: float = 0.0) -> float:
    """
    Combine multiple blur contributors in quadrature:
        sigma_eff = sqrt(sigma_diff^2 + sigma_defocus^2 + sigma_aberr^2 + ...)

    Why quadrature?
    ---------------
    Independent, approximately Gaussian-like blurs (diffraction, mild defocus spread,
    small aberrations, motion, etc.) combine as variances add:
        sigma^2_total = sum_i sigma_i^2.

    Parameters
    ----------
    sigma_diff : float
        Diffraction-limited sigma (pixels), e.g., from sigma_from_fnum().
    sigma_defocus : float
        Additional defocus contribution (pixels). You can map a circle-of-confusion
        diameter 'c' to a Gaussian sigma via sigma ≈ c/2 for a rough surrogate.
    sigma_aberr : float
        Lumped aberration contribution (pixels).

    Returns
    -------
    sigma_eff : float
        Effective sigma in pixels.
    """
    return np.sqrt(sigma_diff**2 + sigma_defocus**2 + sigma_aberr**2)


def show_psf(psf: np.ndarray) -> None:
    """Quick visualization helper for the PSF."""
    plt.imshow(psf, cmap='inferno')
    plt.title(f'PSF (sum={psf.sum():.6f})')
    plt.colorbar()
    plt.tight_layout()
    plt.show()
'''
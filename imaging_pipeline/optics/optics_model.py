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

import numpy as np
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from typing import Tuple

# =============================================================================
# optics_model.py comments added
# Gaussian-PSF-based camera blur model + convenience wrapper
# =============================================================================

# -----------------------------
# Gaussian PSF fundamentals
# -----------------------------
# A circularly symmetric 2-D Gaussian intensity profile:
#   G(x,y) = A * exp(-(x^2 + y^2) / (2*sigma^2))
# Continuous-space energy conservation (∬ G dx dy = 1) uses A = 1/(2*pi*sigma^2).
# On a discrete, finite grid, we sample and then L1-normalize so psf.sum() == 1.
# This ensures convolution preserves average image brightness.


def gaussian_psf(size: int, sigma: float) -> np.ndarray:
    """
    Construct a discrete, L1-normalized 2-D Gaussian PSF.

    Parameters
    ----------
    size : int
        Kernel side length (odd recommended, e.g. 21). Choose so that ~3*sigma falls inside
        the half-width to capture ~99% of the Gaussian energy (size ≳ 6*sigma + 1).
    sigma : float
        Standard deviation in *pixels* (intensity-domain sigma).

    Returns
    -------
    psf : (size, size) ndarray
        Normalized PSF kernel such that psf.sum() == 1.0 (energy-conserving for convolution).
    """
    # Centered coordinates. For size=21, ax = [-10, ..., +10].
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    # Unnormalized 2-D Gaussian (we'll L1-normalize after sampling).
    psf = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))

    # --- CRUCIAL: L1 normalization ---
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
    spatial convolution. Since our PSF is normalized (sum=1), this preserves the
    image's average intensity.

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

    Rule-of-thumb mapping
    ---------------------
    - Airy core ~ Gaussian with waist w0 ~ 0.84 * lambda * N (intensity).
    - Using exp(-2 r^2 / w0^2) ≡ exp(- r^2 / (2*sigma^2)) gives w0 = 2*sigma.
    - Therefore sigma ≈ 0.42 * lambda * N.
    - Convert to pixels by dividing by pixel_pitch.

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

    Rationale
    ---------
    Independent, approximately Gaussian-like blurs (diffraction, mild defocus spread,
    small aberrations, motion, etc.) combine as variances add:
        sigma^2_total = sum_i sigma_i^2.

    Parameters
    ----------
    sigma_diff : float
        Diffraction-limited sigma (pixels), e.g., from sigma_from_fnum().
    sigma_defocus : float
        Additional defocus contribution (pixels). A circle-of-confusion diameter 'c'
        maps roughly to sigma ≈ c/2 for a crude surrogate.
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


# -----------------------------
# High-level blur helper
# -----------------------------
def _ensure_odd(n: int) -> int:
    """Make kernel side length odd (centered impulse)."""
    return n if (n % 2 == 1) else (n + 1)


def apply_optics_blur(
    image: np.ndarray,
    *,
    fnum: float = 8.0,
    wavelength: float = 550e-9,
    pixel_pitch: float = 3.75e-6,
    defocus_sigma: float = 0.0,
    aberr_sigma: float = 0.0,
    kernel_size: int | None = None,
    clip: bool = True,
    return_psf: bool = False
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """
    Blur an image using a Gaussian PSF derived from simple optics parameters.

    Steps
    -----
    1) Compute diffraction-limited sigma (pixels) from (λ, f/#, pixel_pitch).
    2) Combine with extra blur sources in quadrature (defocus, aberrations).
    3) Build an L1-normalized Gaussian PSF (auto-size if not given).
    4) FFT-convolve and return the blurred image (and PSF if requested).

    Parameters
    ----------
    image : ndarray
        2-D grayscale image. uint8 is fine; it will be processed as float32.
    fnum : float
        Lens f-number (f/#).
    wavelength : float
        Wavelength in meters (e.g., 550e-9 for green).
    pixel_pitch : float
        Pixel size in meters.
    defocus_sigma : float
        Additional defocus contribution (in pixels).
    aberr_sigma : float
        Lumped aberration contribution (in pixels).
    kernel_size : int | None
        PSF side length. If None, uses ceil(6*sigma_eff)+1 (then forced odd).
    clip : bool
        If True and input was integer type, clip output to the dtype range.
    return_psf : bool
        If True, return (blurred, psf) instead of just blurred.

    Returns
    -------
    blurred : ndarray
        Blurred image, same shape as input.
    psf : ndarray (optional)
        The PSF used (if return_psf=True).
    """
    if image.ndim != 2:
        raise ValueError("apply_optics_blur expects a 2-D grayscale image.")

    # 1) Diffraction-limited blur (pixels)
    sigma_diff = sigma_from_fnum(
        wavelength=wavelength, fnum=fnum, pixel_pitch=pixel_pitch
    )

    # 2) Combine blur terms (pixels)
    sigma_eff = defocus_sigma_quadrature(
        sigma_diff, sigma_defocus=defocus_sigma, sigma_aberr=aberr_sigma
    )

    # If sigma is ~0 (e.g., absurdly small λ or N), short-circuit
    if sigma_eff <= 0:
        if return_psf:
            delta = np.zeros((3, 3), dtype=np.float32)
            delta[1, 1] = 1.0
            return image.copy(), delta
        return image.copy()

    # 3) Kernel sizing (capture ~99% energy within ±3σ)
    if kernel_size is None:
        kernel_size = int(np.ceil(6.0 * sigma_eff) + 1)
    kernel_size = _ensure_odd(max(3, kernel_size))

    psf = gaussian_psf(kernel_size, sigma_eff)

    # 4) Convolution
    orig_dtype = image.dtype
    img_f = image.astype(np.float32, copy=False)
    blurred_f = apply_psf(img_f, psf)

    # Match original dtype if it was integer
    if np.issubdtype(orig_dtype, np.integer):
        if clip:
            info = np.iinfo(orig_dtype)
            blurred_f = np.clip(blurred_f, info.min, info.max)
        blurred = blurred_f.round().astype(orig_dtype)
    else:
        blurred = blurred_f

    return (blurred, psf) if return_psf else blurred


# Optional: simple self-test (runs only if executed directly)
if __name__ == "__main__":
    # Create a toy slanted edge
    h, w = 256, 256
    xv, yv = np.meshgrid(np.arange(w), np.arange(h))
    edge = (xv * np.cos(np.deg2rad(20)) + yv * np.sin(np.deg2rad(20)))
    img = (edge > w // 2).astype(np.uint8) * 255

    # Apply blur with modest defocus
    blurred, psf = apply_optics_blur(
        img,
        fnum=8.0,
        wavelength=550e-9,
        pixel_pitch=3.75e-6,
        defocus_sigma=0.6,
        aberr_sigma=0.3,
        return_psf=True,
    )

    # Visualize quick sanity check
    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    axs[0].imshow(img, cmap="gray")
    axs[0].set_title("Input")
    axs[0].axis("off")

    axs[1].imshow(blurred, cmap="gray")
    axs[1].set_title("Blurred")
    axs[1].axis("off")

    im = axs[2].imshow(psf, cmap="inferno")
    axs[2].set_title(f"PSF (sum={psf.sum():.4f})")
    axs[2].axis("off")
    fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

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
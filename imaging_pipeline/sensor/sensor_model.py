# sensor_model.py
# -----------------------------------------------------------------------------
# Monochrome sensor pipeline with optional Pixel-Aperture MTF:
#   irradiance (W/m^2 on pixel grid) --[Pixel MTF]--> electrons (with noise)
#   --> DN (ADC).
#
# Key physics (single effective wavelength λ):
#   photons/pixel = (E * A * t) / (hc/λ)
#   electrons mean μ_e = photons * QE
#
# Noise model:
#   • Shot noise: Poisson(μ_e)  → var ≈ μ_e  → σ ≈ √μ  → SNR = μ/σ ≈ √μ
#     => On a log–log plot, SNR vs mean has slope 0.5 in the photon-limited regime.
#   • Dark current: Poisson(dark_rate * t); DSNU adds pixel-wise Gaussian offset.
#   • Read noise: additive Gaussian with RMS σ_read (independent of signal).
#   • PRNU: multiplicative fixed-pattern gain variation (Gaussian on gain).
#
# Why σ ≈ √μ matters:
#   In the photon-limited regime (read noise small, no clipping), each pixel’s
#   electrons follow Poisson statistics, so var ≈ mean. That gives the classic
#   √μ scaling of noise and √μ growth of SNR with illumination/exposure time.
#   Practically: every 4× increase in mean signal gives ~2× SNR improvement.
#
# Pixel-Aperture MTF theory (rectangular pixel aperture):
#   The pixel integrates irradiance over its sensitive area. For a rectangular
#   aperture of width a_x and a_y (in *pixel-pitch units*), the continuous OTF
#   is separable and equals:
#       OTF(fx, fy) = sinc(a_x * fx) * sinc(a_y * fy),  where sinc(u)=sin(pi*u)/(pi*u)
#   When a_x = a_y = 1.0 (aperture spans one pixel), this introduces the well-known
#   sinc roll-off limiting contrast near Nyquist (0.5 cyc/pixel).
#
# Why apply Pixel MTF *before* noise?
#   Pixel integration is deterministic optical averaging. All stochastic effects
#   (shot/dark/read) come *after* light has been integrated, so we filter the
#   irradiance first, then generate electrons + noises.
#
# Implementation notes:
#   • Keep arrays float32 after noise steps to save memory; use uint16 for DN.
#   • We use a fixed numpy RNG seed for deterministic unit tests (overrideable).
#   • All inputs are assumed already sampled on the sensor pixel grid (H×W).
#
# Extensions to add later in this order:
#   • Pixel-aperture MTF variant in spatial domain (separable box) for cross-checks.
#   • CFA + demosaic (keep this monochrome path as a fast baseline).
#   • Rolling shutter timing and temperature-dependent dark current.
# -----------------------------------------------------------------------------

from dataclasses import dataclass
import numpy as np
from typing import Tuple

# Physical constants
H = 6.626_070_15e-34      # Planck constant [J·s]
C = 2.997_924_58e8        # Speed of light [m/s]

@dataclass
class SensorParams:
    # Geometry & exposure
    pixel_size_m: float                 # pixel pitch (square pixel assumed)
    exposure_time_s: float              # integration time

    # Spectral simplification
    wavelength_nm: float = 550.0        # effective wavelength for photon energy

    # Quantum efficiency and fixed-pattern terms
    qe: float = 0.6                     # quantum efficiency (0..1)
    prnu_sigma: float = 0.0             # pixel response non-uniformity (fractional stdev)

    # Dark current and offsets
    dark_current_e_per_s: float = 0.1   # mean dark electrons per second
    dsnu_e_rms: float = 0.0             # dark-signal non-uniformity (additive electrons RMS)

    # Readout & well capacity
    read_noise_e_rms: float = 1.5       # read noise (Gaussian, RMS electrons)
    full_well_e: float = 20000.0        # saturation capacity in electrons

    # ADC model (linear)
    conversion_gain_e_per_dn: float = 2.0  # e⁻ per DN (LSB size in electrons)
    bit_depth: int = 12
    black_level_dn: int = 64

    # Pixel-aperture MTF (rectangular)
    enable_pixel_mtf: bool = True       # apply sinc MTF in frequency domain
    fill_factor_x: float = 1.0          # aperture width along x in pixel units (0..1]
    fill_factor_y: float = 1.0          # aperture height along y in pixel units (0..1]

    # Reproducibility
    seed: int | None = 1234

# ------------------------------ Pixel MTF -------------------------------------
def apply_pixel_aperture_mtf(irradiance: np.ndarray, p: SensorParams) -> np.ndarray:
    """
    Apply the separable pixel-aperture MTF in the frequency domain.

    Theory: rectangular aperture integration (a_x, a_y in pixel units) =>
            MTF(fx,fy) = sinc(a_x * fx) * sinc(a_y * fy),
            where fx, fy are in cycles/pixel and numpy.sinc(x)=sin(pi*x)/(pi*x).

    Notes:
      • Uses rFFT for efficiency; result is strictly real.
      • If fill_factor is 1.0, this matches 'one-pixel box integration'.
      • If <1.0, the optical aperture is smaller than the pixel pitch, yielding
        less low-pass blur (weaker roll-off).
    """
    if not p.enable_pixel_mtf:
        return irradiance.astype(np.float32, copy=False)

    Ht, Wt = irradiance.shape
    ax = float(np.clip(p.fill_factor_x, 1e-6, 1.0))  # clamp to (0,1]
    ay = float(np.clip(p.fill_factor_y, 1e-6, 1.0))

    # Frequency grids in cycles/pixel. For rFFT, use rfftfreq along x (columns).
    fy = np.fft.fftfreq(Ht, d=1.0)          # length Ht, symmetric
    fx = np.fft.rfftfreq(Wt, d=1.0)         # length Wt//2+1, non-negative

    # Separable MTF = sinc(ax*fx) * sinc(ay*fy)
    mtf_x = np.sinc(ax * fx)[None, :]       # shape (1, Wt//2+1)
    mtf_y = np.sinc(ay * fy)[:, None]       # shape (Ht, 1)
    mtf = (mtf_y * mtf_x).astype(np.float32)

    # FFT multiply
    F = np.fft.rfft2(irradiance.astype(np.float32, copy=False))
    F_filtered = F * mtf
    out = np.fft.irfft2(F_filtered, s=irradiance.shape)

    # Numerical guard (small negatives from round-off)
    return np.maximum(out, 0.0).astype(np.float32)

# ------------------------ Irradiance -> Photons -------------------------------
def irradiance_to_photons(irradiance_W_m2: np.ndarray, p: SensorParams) -> np.ndarray:
    """
    Convert irradiance (W/m^2) to incident photons per pixel over the exposure.

    photons = (E * A * t) / (hc/λ)  where A=pixel area, t=exposure time.
    """
    lam = p.wavelength_nm * 1e-9
    pixel_area = p.pixel_size_m ** 2
    energy_J = irradiance_W_m2 * pixel_area * p.exposure_time_s
    photons = energy_J / (H * C / lam)
    return photons

# --------------------------- Full Sensor Pipeline -----------------------------
def apply_sensor_pipeline(irradiance_W_m2: np.ndarray, p: SensorParams) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full sensor pipeline:
      0) (Optional) Pixel-aperture MTF on irradiance (sinc in frequency domain)
      1) Irradiance -> photons -> mean electrons via QE
      2) PRNU (multiplicative fixed pattern on mean)
      3) Shot noise (Poisson)
      4) Dark current (Poisson) + DSNU (Gaussian)
      5) Read noise (Gaussian)
      6) Full-well clip
      7) ADC quantization to DN

    Returns
    -------
    electrons : np.ndarray (float32)
        Noisy, clipped electrons per pixel before ADC.
    dn : np.ndarray (uint16 or uint32)
        Digitized pixel values after linear ADC and black level.
    """
    rng = np.random.default_rng(p.seed)

    # --- 0) Pixel-aperture MTF (deterministic optical averaging) --------------
    if p.enable_pixel_mtf:
        irradiance_W_m2 = apply_pixel_aperture_mtf(irradiance_W_m2, p)

    # --- 1) photons & QE (mean electrons) ------------------------------------
    photons = irradiance_to_photons(irradiance_W_m2, p)
    # Electrons mean before noise; clamp negatives away from Poisson:
    signal_mean_e = np.clip(photons * p.qe, 0.0, None)

    # --- 2) PRNU: multiplicative gain variation (fixed pattern) --------------
    # Model as per-pixel Gaussian fractional variation. This broadens histograms
    # even without shot noise and appears as "fixed grain" across frames.
    if p.prnu_sigma > 0.0:
        prnu = rng.normal(loc=0.0, scale=p.prnu_sigma, size=signal_mean_e.shape)
        signal_mean_e = signal_mean_e * (1.0 + prnu)

    # --- 3) Shot noise: Poisson around the (possibly PRNU-perturbed) mean ----
    # For large means, Poisson ≈ Normal(mean, sqrt(mean)); here we sample Poisson
    # to remain accurate in low-light. This is the source of σ ≈ √μ behavior.
    signal_e = rng.poisson(signal_mean_e).astype(np.float32)

    # --- 4) Dark current + DSNU ----------------------------------------------
    # Dark shot noise is also Poisson around dark_mean = dark_rate * t.
    dark_mean = max(p.dark_current_e_per_s * p.exposure_time_s, 0.0)
    if dark_mean > 0:
        dark_e = rng.poisson(dark_mean, size=signal_e.shape).astype(np.float32)
    else:
        dark_e = np.zeros_like(signal_e, dtype=np.float32)

    # DSNU adds an *additive* pixel-wise offset (e.g., due to leakage variation).
    if p.dsnu_e_rms > 0.0:
        dark_e += rng.normal(loc=0.0, scale=p.dsnu_e_rms, size=signal_e.shape).astype(np.float32)

    electrons = signal_e + dark_e

    # --- 5) Read noise (Gaussian, independent of signal) ---------------------
    # In very low light, read noise dominates and SNR no longer follows √μ.
    if p.read_noise_e_rms > 0.0:
        electrons += rng.normal(loc=0.0, scale=p.read_noise_e_rms, size=electrons.shape).astype(np.float32)

    # --- 6) Full-well clipping -----------------------------------------------
    # Saturation flattens highlights and collapses variance; useful to check with
    # histograms when tuning exposure.
    electrons = np.clip(electrons, 0.0, p.full_well_e).astype(np.float32)

    # --- 7) ADC: linear quantization with black level -------------------------
    # DN = floor(electrons / e_per_dn) + black level; clamp to ADC range.
    dn = np.floor_divide(electrons, max(p.conversion_gain_e_per_dn, 1e-12)).astype(np.int64)
    dn += int(p.black_level_dn)
    max_dn = (1 << p.bit_depth) - 1
    dn = np.clip(dn, 0, max_dn)
    dn = dn.astype(np.uint16 if p.bit_depth <= 16 else np.uint32)

    return electrons, dn

# ------------------------------- Diagnostics ----------------------------------
def estimate_snr_patch(electrons: np.ndarray, patch_slice) -> Tuple[float, float, float]:
    """
    Quick SNR check on a uniform patch of the image.

    Returns (mean, std, snr=mean/std) to help verify:
      • Shot-noise regime: std ≈ sqrt(mean) → snr ≈ sqrt(mean).
      • Read-noise floor: std ~ σ_read when mean is very small.
    """
    patch = electrons[patch_slice].astype(np.float64)
    mu = float(np.mean(patch))
    sigma = float(np.std(patch, ddof=1))
    snr = mu / sigma if sigma > 0 else np.inf
    return mu, sigma, snr
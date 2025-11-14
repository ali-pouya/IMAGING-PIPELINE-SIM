"""
sensor_model.py — monochrome sensor pipeline with optional pixel-aperture MTF.

WHAT THIS MODULE DOES
---------------------
Models a simplified yet physically grounded imaging sensor:
  0) (Optional) Pixel-aperture MTF: rectangular pixel integrates light -> sinc MTF
  1) Irradiance [W/m^2] -> Photons using pixel area and exposure time
  2) Quantum efficiency maps photons -> mean electrons (μ_e)
  3) Fixed pattern terms: PRNU (multiplicative) and DSNU (additive on dark)
  4) Stochastic noise: shot (Poisson), dark (Poisson), read (Gaussian)
  5) Full-well clipping (saturation)
  6) ADC: linear conversion to DN with black level and bit depth

LEARNING NOTES
--------------
• Shot noise ⇒ var ≈ mean (Poisson), so σ ≈ √μ and SNR ≈ √μ (photon-limited).
• Read noise dominates at very low light (SNR flattens).
• Pixel-aperture (rectangular) MTF yields a separable sinc roll-off vs frequency.
  At Nyquist (0.5 cyc/pix), contrast naturally drops even without lens blur.

REFERENCES (short list)
-----------------------
• Janesick, J. R. (2007). Photon Transfer: DN → λ. SPIE Press.
  (Conversion gain, shot/read noise, full-well, SNR behavior)
• Holst, G. C. (2011). CMOS/CCD Sensors and Camera Systems (2e). SPIE Press.
  (PRNU/DSNU, dark current, read noise, linear ADC)
• Goodman, J. W. (2017). Introduction to Fourier Optics (4e).
  (Apertures → OTF/MTF; rectangular integration → sinc)
• ISO 12233:2017. Electronic still picture imaging — SFR/Resolution.
  (Pixel-aperture considerations in practical MTF)
  © 2023 Ali Pouya — Imaging Pipeline (classic edition)
"""


from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np

# Physical constants
H = 6.626_070_15e-34      # Planck [J·s]
C = 2.997_924_58e8        # Speed of light [m/s]


# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
@dataclass
class SensorParams:
    """
    Grouped sensor parameters (monochrome, single effective wavelength).

    Geometry & exposure
    -------------------
    pixel_size_m : pixel pitch in meters (square pixel assumed)
    exposure_time_s : integration time (seconds)

    Spectral simplification
    -----------------------
    wavelength_nm : effective wavelength in nanometers (for photon energy)

    Photo-electron conversion & fixed pattern
    ----------------------------------------
    qe : quantum efficiency (0..1)
    prnu_sigma : fractional STDEV of multiplicative pixel gain variation

    Dark current & offsets
    ----------------------
    dark_current_e_per_s : mean dark electrons per second
    dsnu_e_rms : additive dark-signal non-uniformity (RMS electrons)

    Readout, well, ADC
    ------------------
    read_noise_e_rms : read noise RMS (electrons)
    full_well_e : saturation capacity (electrons)
    conversion_gain_e_per_dn : electrons per DN (LSB size)
    bit_depth : ADC bits (8/10/12/16 typical)
    black_level_dn : offset added prior to clipping (DN)

    Pixel-aperture MTF
    ------------------
    enable_pixel_mtf : if True, apply separable sinc MTF to irradiance
    fill_factor_x/y : rectangular aperture size as a fraction of pixel pitch (0..1]

    Reproducibility
    ---------------
    seed : RNG seed (int or None)
    """
    # Geometry & exposure
    pixel_size_m: float
    exposure_time_s: float

    # Spectral simplification
    wavelength_nm: float = 550.0

    # Conversion & fixed-pattern
    qe: float = 0.6
    prnu_sigma: float = 0.0

    # Dark current & offsets
    dark_current_e_per_s: float = 0.1
    dsnu_e_rms: float = 0.0

    # Readout, well, ADC
    read_noise_e_rms: float = 1.5
    full_well_e: float = 20000.0
    conversion_gain_e_per_dn: float = 2.0
    bit_depth: int = 12
    black_level_dn: int = 64

    # Pixel-aperture MTF
    enable_pixel_mtf: bool = True
    fill_factor_x: float = 1.0
    fill_factor_y: float = 1.0

    # Reproducibility
    seed: int | None = 1234


# -----------------------------------------------------------------------------
# Pixel-aperture MTF (rectangular aperture => separable sinc)
# -----------------------------------------------------------------------------
def apply_pixel_aperture_mtf(irradiance: np.ndarray, p: SensorParams) -> np.ndarray:
    """
    Apply rectangular pixel-aperture MTF in frequency domain.

    Theory: integration over a rectangle a_x×a_y (in pixel-pitch units) yields
        MTF(fx, fy) = sinc(a_x * fx) * sinc(a_y * fy),
    where fx, fy are cycles/pixel and numpy.sinc(u) = sin(pi*u)/(pi*u).

    Implementation details
    ----------------------
    • Uses rFFT along x to exploit Hermitian symmetry (real image).
    • Returns a strictly real irradiance after inverse rFFT.
    • Clamps tiny negative numerical noise to 0.

    Parameters
    ----------
    irradiance : (H, W) float array, W/m^2 on the pixel grid
    p : SensorParams

    Returns
    -------
    filtered irradiance : float32 array, same shape
    """
    if not p.enable_pixel_mtf:
        return irradiance.astype(np.float32, copy=False)

    Ht, Wt = irradiance.shape
    ax = float(np.clip(p.fill_factor_x, 1e-6, 1.0))
    ay = float(np.clip(p.fill_factor_y, 1e-6, 1.0))

    fy = np.fft.fftfreq(Ht, d=1.0)      # length Ht, signed
    fx = np.fft.rfftfreq(Wt, d=1.0)     # length Wt//2+1, non-negative

    mtf = (np.sinc(ay * fy)[:, None] * np.sinc(ax * fx)[None, :]).astype(np.float32)

    F = np.fft.rfft2(irradiance.astype(np.float32, copy=False))
    out = np.fft.irfft2(F * mtf, s=irradiance.shape)

    return np.maximum(out, 0.0).astype(np.float32)


# -----------------------------------------------------------------------------
# Irradiance → Photons
# -----------------------------------------------------------------------------
def irradiance_to_photons(irradiance_W_m2: np.ndarray, p: SensorParams) -> np.ndarray:
    """
    Convert irradiance [W/m^2] to incident photons per pixel over the exposure:
        photons = (E * A * t) / (hc/λ)

    where:
      E = irradiance_W_m2
      A = pixel_size_m^2
      t = exposure_time_s
      hc/λ = photon energy (J)
    """
    lam = p.wavelength_nm * 1e-9
    pixel_area = p.pixel_size_m ** 2
    energy_J = irradiance_W_m2 * pixel_area * p.exposure_time_s
    return energy_J / (H * C / lam)


# -----------------------------------------------------------------------------
# Full sensor pipeline
# -----------------------------------------------------------------------------
def apply_sensor_pipeline(irradiance_W_m2: np.ndarray, p: SensorParams) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full monochrome sensor pipeline.

    Steps
    -----
    0) Pixel-aperture MTF (deterministic optical averaging; optional)
    1) Irradiance → photons → mean electrons via QE (clamped >= 0)
    2) PRNU: multiplicative per-pixel gain variation on the mean (optional)
    3) Shot noise: Poisson around the (possibly PRNU-perturbed) mean
    4) Dark current: Poisson(dark_rate * t) + DSNU (Gaussian)
    5) Read noise: additive Gaussian with RMS read_noise_e_rms
    6) Full-well clipping
    7) ADC: linear quantization with black level and bit-depth clamping

    Returns
    -------
    electrons : float32 (pre-ADC, after noise & clipping)
    dn : uint (post-ADC, bit-depth dependent type)
    """
    rng = np.random.default_rng(p.seed)

    # 0) Pixel-aperture MTF (sinc roll-off near Nyquist)
    if p.enable_pixel_mtf:
        irradiance_W_m2 = apply_pixel_aperture_mtf(irradiance_W_m2, p)

    # 1) Photons → mean electrons
    photons = irradiance_to_photons(irradiance_W_m2, p)
    mean_e = np.clip(photons * p.qe, 0.0, None)

    # 2) PRNU (multiplicative fixed pattern on the mean)
    if p.prnu_sigma > 0.0:
        mean_e = mean_e * (1.0 + rng.normal(0.0, p.prnu_sigma, size=mean_e.shape))

    # 3) Shot noise (Poisson around mean)
    signal_e = rng.poisson(mean_e).astype(np.float32)

    # 4) Dark current + DSNU (additive)
    dark_mean = max(p.dark_current_e_per_s * p.exposure_time_s, 0.0)
    dark_e = rng.poisson(dark_mean, size=signal_e.shape).astype(np.float32) if dark_mean > 0 else np.zeros_like(signal_e)
    if p.dsnu_e_rms > 0.0:
        dark_e += rng.normal(0.0, p.dsnu_e_rms, size=signal_e.shape).astype(np.float32)

    electrons = signal_e + dark_e

    # 5) Read noise (independent Gaussian)
    if p.read_noise_e_rms > 0.0:
        electrons += rng.normal(0.0, p.read_noise_e_rms, size=electrons.shape).astype(np.float32)

    # 6) Full-well clipping
    electrons = np.clip(electrons, 0.0, p.full_well_e).astype(np.float32)

    # 7) ADC: DN = floor(e / e_per_dn) + black level; clamp to ADC range
    e_per_dn = max(p.conversion_gain_e_per_dn, 1e-12)
    dn = np.floor_divide(electrons, e_per_dn).astype(np.int64)
    dn += int(p.black_level_dn)
    dn = np.clip(dn, 0, (1 << p.bit_depth) - 1)
    dn = dn.astype(np.uint16 if p.bit_depth <= 16 else np.uint32)

    return electrons, dn


# -----------------------------------------------------------------------------
# Diagnostics helper (useful for unit demos)
# -----------------------------------------------------------------------------
def estimate_snr_patch(electrons: np.ndarray, patch_slice) -> Tuple[float, float, float]:
    """
    Estimate (mean, std, SNR) on a uniform patch of the electrons image.

    For a shot-noise-limited region:
      std ≈ sqrt(mean)  and  SNR ≈ sqrt(mean).

    Returns
    -------
    mean, std, snr
    """
    patch = electrons[patch_slice].astype(np.float64)
    mu = float(np.mean(patch))
    sigma = float(np.std(patch, ddof=1))
    snr = mu / sigma if sigma > 0 else np.inf
    return mu, sigma, snr


# -----------------------------------------------------------------------------
# Back-compat wrapper (classic API)
# -----------------------------------------------------------------------------
def add_sensor_effects(
    irradiance_W_m2: np.ndarray,
    *,
    # legacy names & defaults used in older main.py versions
    pixel_pitch_um: float = 3.75,
    exposure_time_s: float = 0.01,
    qe: float = 0.6,
    read_noise_e: float = 1.5,
    full_well_e: float = 20000.0,
    conversion_gain_e_per_dn: float = 2.0,
    bit_depth: int = 12,
    black_level_dn: int = 64,
    enable_pixel_mtf: bool = True,
    fill_factor: float = 1.0,
    seed: int = 1234,
    wavelength_nm: float = 550.0,
    prnu_sigma: float = 0.0,
    dark_current_e_per_s: float = 0.1,
    dsnu_e_rms: float = 0.0,
    # accepted but ignored as a toggle (shot noise is always modeled)
    shot_noise: bool = True,
    # allow silent swallowing of any extra legacy kwargs
    **_ignored,
):
    """
    Backward-compatible entry point that maps legacy kwargs to SensorParams.
    Intended to keep classic main.py working unchanged.
    """
    p = SensorParams(
        pixel_size_m=pixel_pitch_um * 1e-6,
        exposure_time_s=exposure_time_s,
        wavelength_nm=wavelength_nm,
        qe=qe,
        prnu_sigma=prnu_sigma,
        dark_current_e_per_s=dark_current_e_per_s,
        dsnu_e_rms=dsnu_e_rms,
        read_noise_e_rms=read_noise_e,
        full_well_e=full_well_e,
        conversion_gain_e_per_dn=conversion_gain_e_per_dn,
        bit_depth=bit_depth,
        black_level_dn=black_level_dn,
        enable_pixel_mtf=enable_pixel_mtf,
        fill_factor_x=fill_factor,
        fill_factor_y=fill_factor,
        seed=seed,
    )
    return apply_sensor_pipeline(irradiance_W_m2, p)

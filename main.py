""""
End-to-end imaging pipeline:
scene → optics → sensor → metrics

Module contracts (refresher):
- imaging_pipeline.scenes.scene_generator.generate_scene(kind: str, size: int) -> float32 image [0..1]
- imaging_pipeline.optics.optics_model.apply_optics_blur(image, **kwargs) -> float32 image
  (e.g., defocus or Gaussian PSF; kwargs can be sigma=... or defocus_waves=... or your richer params)
- imaging_pipeline.sensor.sensor_model:
    - SensorParams dataclass (pixel geometry, exposure, noise, ADC, pixel MTF)
    - add_sensor_effects(irradiance_W_m2: np.ndarray, params: SensorParams) -> (electrons, dn)

This file provides:
- run_pipeline(): full scene→optics→sensor→metrics run + plots/saves
- Quick tests for each stage:
    test_scene()
    test_optics()
    test_sensor()
    test_metrics()
  Run via:  python main.py --test <scene|optics|sensor|metrics|all>
"""

import argparse
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

# --- Pipeline modules ---
from imaging_pipeline.scenes.scene_generator import generate_scene
from imaging_pipeline.optics.optics_model import apply_optics_blur
from imaging_pipeline.sensor.sensor_model import (
    SensorParams,
    apply_sensor_pipeline,   # used by add_sensor_effects wrapper
    estimate_snr_patch
)
from imaging_pipeline.utils.metrics_module import compute_snr, mtf_from_fft, plot_histogram


# -----------------------------------------------------------------------------
# Helpers & Adapters
# -----------------------------------------------------------------------------

def _build_sensor_params(sensor_kwargs: Dict[str, Any]) -> SensorParams:
    """
    Backward-compatible adapter from older kwargs (pixel_pitch_um, read_noise_e, fill_factor)
    to the new SensorParams dataclass.

    Accepted keys (examples):
      - pixel_pitch_um: 3.75        -> pixel_size_m
      - exposure_time_s: 0.01       -> same
      - wavelength_nm: 550.0        -> same
      - qe: 0.6                     -> same
      - prnu_sigma: 0.01            -> same
      - dark_current_e_per_s: 0.05  -> same
      - dsnu_e_rms: 0.5             -> same
      - read_noise_e or read_noise_e_rms: 1.5  -> read_noise_e_rms
      - full_well_e: 20000          -> same
      - conversion_gain_e_per_dn: 2.0 -> same
      - bit_depth: 12               -> same
      - black_level_dn: 64          -> same
      - enable_pixel_mtf: True      -> same
      - fill_factor or fill_factor_x/y: 1.0 -> fill_factor_x & fill_factor_y
      - seed: 1234                  -> same
    """
    k = dict(sensor_kwargs) if sensor_kwargs else {}

    # Map legacy names
    pixel_size_m = k.pop("pixel_size_m", None)
    if pixel_size_m is None:
        pitch_um = k.pop("pixel_pitch_um", 3.75)
        pixel_size_m = float(pitch_um) * 1e-6

    read_noise = k.pop("read_noise_e_rms", None)
    if read_noise is None:
        read_noise = k.pop("read_noise_e", 1.5)

    # Fill factor
    ff_x = k.pop("fill_factor_x", None)
    ff_y = k.pop("fill_factor_y", None)
    ff = k.pop("fill_factor", None)
    if ff is not None:
        ff_x = ff_x if ff_x is not None else ff
        ff_y = ff_y if ff_y is not None else ff
    if ff_x is None: ff_x = 1.0
    if ff_y is None: ff_y = 1.0

    # Construct SensorParams with sane defaults, then override with remaining keys if any
    params = SensorParams(
        pixel_size_m=pixel_size_m,
        exposure_time_s=k.pop("exposure_time_s", 0.01),
        wavelength_nm=k.pop("wavelength_nm", 550.0),
        qe=k.pop("qe", 0.6),
        prnu_sigma=k.pop("prnu_sigma", 0.0),
        dark_current_e_per_s=k.pop("dark_current_e_per_s", 0.1),
        dsnu_e_rms=k.pop("dsnu_e_rms", 0.0),
        read_noise_e_rms=read_noise,
        full_well_e=k.pop("full_well_e", 20000.0),
        conversion_gain_e_per_dn=k.pop("conversion_gain_e_per_dn", 2.0),
        bit_depth=k.pop("bit_depth", 12),
        black_level_dn=k.pop("black_level_dn", 64),
        enable_pixel_mtf=k.pop("enable_pixel_mtf", True),
        fill_factor_x=ff_x,
        fill_factor_y=ff_y,
        seed=k.pop("seed", 1234),
    )

    # Any unexpected keys? keep quiet or raise; here we ignore to stay flexible.
    return params


def add_sensor_effects(irradiance_W_m2: np.ndarray, sensor_params: SensorParams) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wrapper to convert optics-plane irradiance (already on the pixel grid)
    into noisy electrons and digitized DN using the sensor model.

    Returns:
      electrons (float32), dn (uint)
    """
    return apply_sensor_pipeline(irradiance_W_m2, sensor_params)


# -----------------------------------------------------------------------------
# Full Pipeline
# -----------------------------------------------------------------------------

def run_pipeline(
    scene_kind: str = "barcode",
    size: int = 512,
    scene_kwargs: Dict[str, Any] | None = None,
    optics_kwargs: Dict[str, Any] | None = None,
    sensor_kwargs: Dict[str, Any] | None = None,
    outdir: str | Path = "outputs"
):
    """
    Full end-to-end run:
      scene → optics → sensor → metrics → plots & .npy saves

    Usage:
      python main.py            # with defaults
      python main.py --scene checker --size 256 --sigma 0.8 --bit_depth 12

    Notes:
      • The optics stage here calls a richer apply_optics_blur signature —
        update the kwargs to match your current optics_model.
      • The sensor stage uses SensorParams with pixel-aperture MTF available.
    """
    # --- Defaults --------------------------------------------------------------
    optics_kwargs = optics_kwargs or {
        # Example hybrid blur: defocus + aberration sigma on pixel grid
        "fnum": 8.0,
        "wavelength": 550e-9,
        "pixel_pitch": 3.75e-6,
        "defocus_sigma": 0.6,
        "aberr_sigma": 0.3,
        "kernel_size": None,
        "return_psf": True,
    }

    sensor_kwargs = sensor_kwargs or {
        "pixel_pitch_um": 3.75,
        "exposure_time_s": 0.01,
        "qe": 0.6,
        "read_noise_e": 1.5,
        "full_well_e": 20000,
        "conversion_gain_e_per_dn": 2.0,
        "bit_depth": 12,
        "black_level_dn": 64,
        "enable_pixel_mtf": True,
        "fill_factor": 1.0,
        "seed": 1234,
    }

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- 1) Scene --------------------------------------------------------------
    # Theory: "scene" is reflectance/brightness; we keep it normalized [0..1].
    scene_kwargs = scene_kwargs or {}
    scene = generate_scene(kind=scene_kind, size=size, **scene_kwargs).astype(np.float32)
    scene = np.clip(scene, 0.0, 1.0)

    # --- 2) Optics (PSF/OTF convolution → irradiance on sensor grid) ----------
    # NOTE: The exact signature depends on your optics_model; adjust as needed.
    blurred, psf = apply_optics_blur(
        scene,
        **optics_kwargs
    )
    img_optics = np.clip(blurred.astype(np.float32), 0.0, 1.0)

    # --- 3) Sensor (integration, noise, quantization) -------------------------
    params = _build_sensor_params(sensor_kwargs)
    electrons, dn = add_sensor_effects(img_optics, params)

    # --- 4) Metrics ------------------------------------------------------------
    # For SNR, compare noisy/quantized image (normalized back to 0..1) vs pre-noise (img_optics)
    max_level = float((1 << params.bit_depth) - 1)
    img_sensor_norm = dn.astype(np.float32) / max_level
    snr_db = compute_snr(img_sensor_norm, img_optics)
    f_norm, mtf = mtf_from_fft(img_sensor_norm)

    # --- Plots / Saves ---------------------------------------------------------
    # Montage
    fig1 = plt.figure(figsize=(12, 4))
    for i, (title, im) in enumerate([("Scene", scene),
                                     ("After Optics", img_optics),
                                     (f"Sensor (quantized {params.bit_depth}-bit)", img_sensor_norm)], start=1):
        ax = fig1.add_subplot(1, 3, i)
        ax.imshow(im, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    fig1.suptitle(f"Pipeline Overview — SNR ≈ {snr_db:.1f} dB")
    fig1.tight_layout()
    fig1.savefig(outdir / "pipeline_overview.png", dpi=150)

    # Histogram
    plot_histogram(img_sensor_norm, title="Sensor Output Histogram")
    plt.savefig(outdir / "histogram.png", dpi=150)

    # MTF (orientation-averaged from FFT magnitude)
    plt.figure()
    plt.plot(f_norm, mtf)
    plt.xlabel("Normalized spatial frequency (cycles/pixel, Nyquist=0.5)")
    plt.ylabel("MTF")
    plt.title("FFT-based MTF (orientation-averaged)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "mtf_fft.png", dpi=150)

    # Save arrays
    np.save(outdir / "scene.npy", scene)
    np.save(outdir / "after_optics.npy", img_optics)
    np.save(outdir / "sensor_electrons.npy", electrons.astype(np.float32))
    np.save(outdir / "sensor_dn_norm.npy", img_sensor_norm.astype(np.float32))

    print(f"[OK] Saved outputs to: {outdir.resolve()}")
    print(f"SNR ≈ {snr_db:.2f} dB | DN range [{dn.min()} .. {dn.max()}] @ {params.bit_depth} bits")


# -----------------------------------------------------------------------------
# Quick Tests (each unit focuses on a stage)
# -----------------------------------------------------------------------------
# Run all tests:          python main.py --test all
# Run just scene test:    python main.py --test scene
# Run just optics test:   python main.py --test optics
# Run just sensor test:   python main.py --test sensor
# Run just metrics test:  python main.py --test metrics
# -----------------------------------------------------------------------------

def test_scene(size: int = 256, outdir: str | Path = "outputs_test_scene"):
    """
    Purpose: visually sanity-check generated scenes (barcode/checker/edges/etc.)
    Expectation: images within [0..1], correct geometry and contrast.
    """
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    kinds = ["slanted_edge", "barcode", "gradient", "siemens_star", "checker", "custom"]
    fig = plt.figure(figsize=(12, 6))
    idx = 1
    for k in kinds:
        # Skip 'custom' when no path provided
        if k == "custom":
            continue
        im = generate_scene(kind=k, size=size).astype(np.float32)
        ax = fig.add_subplot(2, (len(kinds)+1)//2, idx); idx += 1
        ax.imshow(np.clip(im, 0, 1), cmap="gray", vmin=0, vmax=1)
        ax.set_title(k)
        ax.axis("off")
    fig.suptitle("Scene Generator — Quick Gallery")
    fig.tight_layout()
    fig.savefig(outdir / "scene_gallery.png", dpi=150)
    print("[TEST scene] saved:", (outdir / "scene_gallery.png").resolve())


def test_optics(size: int = 256, outdir: str | Path = "outputs_test_optics"):
    """
    Purpose: verify optics PSF/blur behaves as expected.
    Test 1: impulse → PSF image
    Test 2: slanted edge → edge spread function + visible blur broadening
    """
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Impulse response → PSF
    impulse = np.zeros((size, size), dtype=np.float32)
    impulse[size//2, size//2] = 1.0
    blurred_impulse, psf = apply_optics_blur(
        impulse,
        fnum=8.0, wavelength=550e-9, pixel_pitch=3.75e-6,
        defocus_sigma=0.8, aberr_sigma=0.3, kernel_size=None, return_psf=True
    )
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1); plt.imshow(blurred_impulse, cmap="magma"); plt.title("Impulse → Image")
    plt.axis("off")
    plt.subplot(1, 2, 2); plt.imshow(psf, cmap="magma"); plt.title("Kernel (PSF)")
    plt.axis("off")
    plt.tight_layout(); plt.savefig(outdir / "optics_impulse_psf.png", dpi=150)

    # Slanted edge
    edge = generate_scene("slanted_edge", size)
    blurred_edge, _ = apply_optics_blur(
        edge,
        fnum=8.0, wavelength=550e-9, pixel_pitch=3.75e-6,
        defocus_sigma=0.8, aberr_sigma=0.3, kernel_size=None, return_psf=True
    )
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1); plt.imshow(edge, cmap="gray", vmin=0, vmax=1); plt.title("Input: Slanted Edge"); plt.axis("off")
    plt.subplot(1,2,2); plt.imshow(blurred_edge, cmap="gray", vmin=0, vmax=1); plt.title("After Optics"); plt.axis("off")
    plt.tight_layout(); plt.savefig(outdir / "optics_slanted_edge.png", dpi=150)

    print("[TEST optics] saved gallery to:", outdir.resolve())


def test_sensor(size: int = 256, outdir: str | Path = "outputs_test_sensor"):
    """
    Purpose: validate sensor noise physics and pixel-aperture MTF.
    Checks:
      • SNR scaling ~ √μ in photon-limited regime (irradiance sweep)
      • Read-noise floor at very low light
      • Pixel MTF roll-off near Nyquist (toggle enable_pixel_mtf / fill_factor)
    """
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Uniform irradiance frames at different levels
    levels = [2.5e-7, 5e-7, 1e-6, 2e-6, 4e-6]  # arbitrary units (W/m^2)
    params = SensorParams(
        pixel_size_m=3.75e-6, exposure_time_s=0.01,
        qe=0.6, read_noise_e_rms=1.5, full_well_e=20000, conversion_gain_e_per_dn=2.0,
        bit_depth=12, black_level_dn=64, enable_pixel_mtf=True, fill_factor_x=1.0, fill_factor_y=1.0, seed=1234
    )

    snr_vals = []
    for L in levels:
        irr = np.ones((size, size), dtype=np.float32) * L
        electrons, dn = add_sensor_effects(irr, params)
        mu, sigma, snr = estimate_snr_patch(electrons, np.s_[size//2-16:size//2+16, size//2-16:size//2+16])
        snr_vals.append((L, mu, sigma, snr))

    # Plot SNR vs mean electrons (log-log): expect slope ~0.5 when photon-limited
    means = np.array([m for _, m, _, _ in snr_vals])
    snrs = np.array([s for _, _, _, s in snr_vals])
    plt.figure()
    plt.loglog(means, snrs, marker="o")
    plt.xlabel("Mean electrons (μ)"); plt.ylabel("SNR (μ/σ)")
    plt.title("Shot-Noise Regime: SNR ∝ √μ (slope ~ 0.5 on log-log)")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout(); plt.savefig(outdir / "sensor_snr_scaling.png", dpi=150)

    # Pixel-MTF demo: Siemens star through pixel aperture on/off
    star = generate_scene("siemens_star", size)
    params_on = params
    params_off = SensorParams(**{**params.__dict__, "enable_pixel_mtf": False})
    _, dn_on  = add_sensor_effects(star, params_on)
    _, dn_off = add_sensor_effects(star, params_off)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.imshow(dn_off.astype(np.float32)/((1<<params.bit_depth)-1), cmap="gray", vmin=0, vmax=1)
    plt.title("Pixel MTF OFF"); plt.axis("off")
    plt.subplot(1,2,2); plt.imshow(dn_on.astype(np.float32)/((1<<params.bit_depth)-1), cmap="gray", vmin=0, vmax=1)
    plt.title("Pixel MTF ON"); plt.axis("off")
    plt.tight_layout(); plt.savefig(outdir / "sensor_pixel_mtf_on_off.png", dpi=150)

    print("[TEST sensor] saved SNR scaling & pixel MTF demos to:", outdir.resolve())


def test_metrics(size: int = 256, outdir: str | Path = "outputs_test_metrics"):
    """
    Purpose: ensure metrics run and produce plausible curves.
    We use a slanted edge through optics + sensor and compute FFT-based MTF.
    """
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    edge = generate_scene("slanted_edge", size)
    blurred, _ = apply_optics_blur(
        edge,
        fnum=8.0, wavelength=550e-9, pixel_pitch=3.75e-6,
        defocus_sigma=0.6, aberr_sigma=0.3, kernel_size=None, return_psf=True
    )
    params = SensorParams(
        pixel_size_m=3.75e-6, exposure_time_s=0.01, qe=0.6, read_noise_e_rms=1.5,
        full_well_e=20000, conversion_gain_e_per_dn=2.0, bit_depth=12,
        black_level_dn=64, enable_pixel_mtf=True, fill_factor_x=1.0, fill_factor_y=1.0, seed=5678
    )
    _, dn = add_sensor_effects(np.clip(blurred.astype(np.float32), 0, 1), params)
    img_sensor_norm = dn.astype(np.float32) / ((1 << params.bit_depth) - 1)

    f_norm, mtf = mtf_from_fft(img_sensor_norm)
    plt.figure(); plt.plot(f_norm, mtf); plt.xlabel("Normalized f (cyc/pix)"); plt.ylabel("MTF")
    plt.title("FFT-based MTF (edge scene)"); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(outdir / "metrics_mtf.png", dpi=150)
    print("[TEST metrics] saved MTF curve to:", outdir.resolve())


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Imaging pipeline: scene → optics → sensor → metrics")
    p.add_argument("--scene", default="barcode",
                   help="scene kind: slanted_edge, barcode, gradient, siemens_star, checker, custom")
    p.add_argument("--size", type=int, default=512, help="scene size (pixels)")
    p.add_argument("--sigma", type=float, default=1.2, help="optics blur sigma (if using Gaussian-only path)")
    p.add_argument("--bit_depth", type=int, default=12, help="sensor quantization bits")
    p.add_argument("--outdir", default="outputs", help="output directory")
    p.add_argument("--test", default=None, choices=["scene","optics","sensor","metrics","all"], help="run quick tests instead of full pipeline")

    # --- Scene-specific options ---
    # slanted_edge
    p.add_argument("--slanted_angle_deg", type=float, default=5.0, help="slanted_edge: edge angle in degrees")
    p.add_argument("--slanted_threshold", type=float, default=0.0, help="slanted_edge: threshold (edge position)")
    # siemens_star
    p.add_argument("--siemens_spokes", type=int, default=55, help="siemens_star: number of spokes")
    # checker
    p.add_argument("--checker_square_px", type=int, default=16, help="checker: square tile size in pixels")
    p.add_argument("--checker_invert", action="store_true", help="checker: invert black/white tiles")
    # custom
    p.add_argument("--custom_path", type=str, default=None, help="custom: path to image file (PNG/JPG/etc.)")
    p.add_argument("--custom_keep_aspect", action="store_true", help="custom: keep aspect ratio on square canvas")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.test:
        if args.test in ("scene", "all"):
            # Run just the scene generator gallery:
            #   python main.py --test scene
            test_scene(size=min(args.size, 384))
        if args.test in ("optics", "all"):
            # Run optics impulse & edge demos:
            #   python main.py --test optics
            test_optics(size=min(args.size, 384))
        if args.test in ("sensor", "all"):
            # Run sensor SNR-scaling & pixel-MTF demos:
            #   python main.py --test sensor
            test_sensor(size=min(args.size, 384))
        if args.test in ("metrics", "all"):
            # Run metrics MTF computation demo:
            #   python main.py --test metrics
            test_metrics(size=min(args.size, 384))
    else:
        # Full end-to-end pipeline:
        #   python main.py --scene barcode --size 512 --bit_depth 12 --outdir outputs

        # --- Assemble scene_kwargs from CLI ---
        scene_kwargs: Dict[str, Any] = {}
        scene_kind = args.scene

        if scene_kind == "slanted_edge":
            scene_kwargs.update({
                "angle_deg": args.slanted_angle_deg,
                "threshold": args.slanted_threshold,
            })
        elif scene_kind == "siemens_star":
            scene_kwargs.update({"spokes": args.siemens_spokes})
        elif scene_kind == "checker":
            scene_kwargs.update({
                "square_px": args.checker_square_px,
                "invert": bool(args.checker_invert),
            })
        elif scene_kind == "custom":
            # If no path provided, ignore 'custom' by falling back to 'gradient'
            if not args.custom_path:
                print("[WARN] --scene custom requested but --custom_path not provided. Falling back to 'gradient'.")
                scene_kind = "gradient"
            else:
                scene_kwargs.update({
                    "path": args.custom_path,
                    "keep_aspect": bool(args.custom_keep_aspect),
                })

        run_pipeline(
            scene_kind=scene_kind,
            size=args.size,
            scene_kwargs=scene_kwargs,
            optics_kwargs={
                "fnum": 8.0,
                "wavelength": 550e-9,
                "pixel_pitch": 3.75e-6,
                "defocus_sigma": args.sigma,  # wired to CLI
                "aberr_sigma": 0.3,
                "kernel_size": None,
                "return_psf": True,
            },
            sensor_kwargs={"bit_depth": args.bit_depth},
            outdir=args.outdir
        )

#Siemens Star — Nominal Diffraction-Limited Case
#python main.py --scene siemens --size 512 --bit_depth 12 --sigma 0.6
#→ Verifies circular symmetry and MTF falloff at high frequency.

#Barcode — Realistic Imaging Example
#python main.py --scene barcode --size 512 --bit_depth 12 --sigma 0.8
#→ Great for checking how EDoF blur affects line contrast and SNR.

#Checkerboard — Sharp Contrast Scene
#python main.py --scene checker --size 256 --bit_depth 10 --sigma 0.5
#→ Good for testing quantization and tone response.

#Slanted Edge — For MTF Validation
#python main.py --scene slanted_edge --size 512 --bit_depth 12 --sigma 0.7
#→ Useful for comparing FFT-based and ISO 12233 edge-MTF methods.

#Siemens Star — Severe Defocus Stress Test
#python main.py --scene siemens --size 512 --bit_depth 12 --sigma 1.5
#→ Simulates strong defocus and checks PSF normalization.

#Siemens Star — Low-Noise, 16-bit Sensor
#python main.py --scene siemens --size 512 --bit_depth 16 --sigma 0.6
#→ Useful for dynamic-range and SNR sweep checks.


##Quick Tests (unit tests)
##These skip the full pipeline and test each subsystem individually.
##Scene generator gallery
#python main.py --test scene

##Optics impulse and edge response
#python main.py --test optics

##Sensor noise and pixel MTF behavior
#python main.py --test sensor

##FFT-based MTF computation check
#python main.py --test metrics

##Run all four tests in one go
#python main.py --test all

#Advanced Custom Examples
#Override output folder and higher resolution
#python main.py --scene siemens --size 1024 --sigma 0.5 --outdir results/hires

##Use a lower f-number (simulate shallower DOF)
##(add an extra key manually in optics kwargs inside main, or expose --fnum later)
##For now, edit the optics_kwargs in code like this:
##"fnum": 4.0,
##Then run:
#python main.py --scene checker --size 512 --sigma 0.6

##Sweep bit depth quickly (useful for plotting quantization effects)
#python main.py --scene siemens --bit_depth 8
#python main.py --scene siemens --bit_depth 10
#python main.py --scene siemens --bit_depth 12



'''
# %% Setup
import numpy as np, matplotlib.pyplot as plt
from imaging_pipeline.scenes.scene_generator import *
from imaging_pipeline.optics.optics_model import *

#def main():

# %% Generate synthetic scene
#show_scene(barcode, "Synthetic Barcode Scene")
barcode = generate_barcode_scene()
#show_scene(gradient, "Grayscale Gradient Scene")
gradient = generate_gradient_scene()
#show_scene(edge, "Slanted Edge Scene")
edge = generate_slanted_edge()

siemens = generate_siemens_star()

scene = siemens.astype(np.float32)
plt.imshow(scene, cmap="gray")
plt.title("Scene")
plt.show()

# %%  (2) Compute diffraction-limited sigma (in pixels) from optics + sensor
sigma_diff = sigma_from_fnum(wavelength=550e-9, fnum=8.0, pixel_pitch=3.75e-6)
print(f"sigma_diff = {sigma_diff:.4f}")
# Optional to add defocus/aberration as extra sigmas (pixels)
sigma_eff = defocus_sigma_quadrature(sigma_diff, sigma_defocus=0.7, sigma_aberr=0.3)
print(f"sigma_eff = {sigma_eff:.4f}")

# %% Build PSF and blur image
# (3) kernel size chosen to capture Gaussian tails; ~6*sigma + 1 is a good start
# size = int(np.ceil(6 * sigma_eff)) | 1  # ensure odd
size = int(np.ceil(6 * sigma_eff))
size = size + (size % 2 == 0)

psf = gaussian_psf(size, sigma_eff) # L1-normalized (sum==1)
# (4) Convolve (FFT) – average brightness preserved because psf.sum()==1
blurred = apply_psf(scene, psf)
# (5) Visualize
show_psf(psf)
plt.figure()
plt.subplot(1, 2, 1); plt.imshow(scene, cmap='gray');   plt.title('Original'); plt.axis('off')
plt.subplot(1, 2, 2); plt.imshow(blurred, cmap='gray'); plt.title('Blurred');  plt.axis('off')
plt.tight_layout(); plt.show()




#if __name__ == "__main__":
#    main()
# %%
'''
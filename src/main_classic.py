"""
#python -m src.main_classic        # quick visual sanity
#python -m src.main_full --test all

main_classic.py — Minimal, classic imaging pipeline: scene → optics → sensor

WHAT THIS FILE DOES
-------------------
1) Generates a synthetic scene (float32 in [0, 1])
2) Applies an optics blur using a Gaussian PSF surrogate (energy-normalized)
3) Runs the sensor model (photons → electrons with noise → DN)
4) Shows a 3-panel figure: Scene | After Optics | Sensor DN
5) Saves the figure and key arrays to ./outputs/

USAGE (run from repo root)
--------------------------
  python -m src.main_classic
  python -m src.main_classic --scene siemens_star --size 512 --sigma 0.8 --bit_depth 12

NOTES
-----
• This is the lightweight “classic” entry point for demonstrations and teaching.
• The detailed math and references live inside the modules under imaging_pipeline/.
• For a fuller CLI/test harness, keep a second file (e.g., src/main_full.py).

© 2025 Ali Pouya — Imaging Pipeline Classic
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Local package imports (relative to src/)
from imaging_pipeline.scenes.scene_generator import generate_scene
from imaging_pipeline.optics.optics_model import apply_optics_blur
from imaging_pipeline.sensor.sensor_model import add_sensor_effects
from imaging_pipeline.utils.metrics_module import compute_snr


def run_once(scene_kind: str, size: int, sigma: float, bit_depth: int, outdir: str = "outputs") -> None:
    """
    Execute one pass of the classic pipeline and visualize/save results.

    Parameters
    ----------
    scene_kind : str
        'slanted_edge' | 'barcode' | 'gradient' | 'siemens_star' | 'checker'
    size : int
        Square canvas size for the scene.
    sigma : float
        Gaussian sigma in *pixels* for the optics blur (classic path).
    bit_depth : int
        Sensor ADC bit depth (8/10/12/16 typical).
    outdir : str
        Directory to save outputs (figure + .npy arrays).
    """
    outpath = Path(outdir)
    outpath.mkdir(parents=True, exist_ok=True)

    # 1) Scene (normalized [0..1])
    scene = generate_scene(kind=scene_kind, size=size).astype(np.float32)

    # 2) Optics (energy-normalized Gaussian PSF; preserves average brightness)
    blurred, psf = apply_optics_blur(
        scene,
        model="gaussian",         # back-compat; only 'gaussian' implemented
        sigma=float(sigma),       # direct sigma override in pixels
        return_psf=True,
    )
    blurred = np.clip(blurred, 0.0, 1.0).astype(np.float32)

    # 3) Sensor (photons → electrons with noise → DN)
    electrons, dn = add_sensor_effects(
        irradiance_W_m2=blurred,
        pixel_pitch_um=3.75,
        exposure_time_s=0.01,
        qe=0.6,
        read_noise_e=1.5,
        full_well_e=20000,
        conversion_gain_e_per_dn=2.0,
        bit_depth=int(bit_depth),
        black_level_dn=64,
        enable_pixel_mtf=True,
        fill_factor=1.0,
        seed=1234,
        shot_noise=True,  # accepted by legacy wrapper; shot noise is always modeled
    )

    # Normalize DN for display and compute a quick global SNR vs pre-noise
    max_dn = float((1 << bit_depth) - 1)
    sensor_norm = dn.astype(np.float32) / max_dn
    snr_db = compute_snr(sensor_norm, blurred)

    # --- 3-up visualization: Scene | After Optics | Sensor DN ---
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(scene, cmap="gray", vmin=0, vmax=1);      axs[0].set_title("Scene");                    axs[0].axis("off")
    axs[1].imshow(blurred, cmap="gray", vmin=0, vmax=1);    axs[1].set_title("After Optics");             axs[1].axis("off")
    axs[2].imshow(sensor_norm, cmap="gray", vmin=0, vmax=1);axs[2].set_title(f"Sensor DN (SNR≈{snr_db:.1f} dB)"); axs[2].axis("off")
    fig.tight_layout()

    # Save figure & arrays
    fig.savefig(outpath / "pipeline_overview.png", dpi=150)
    np.save(outpath / "scene.npy", scene)
    np.save(outpath / "after_optics.npy", blurred)
    np.save(outpath / "sensor_electrons.npy", electrons.astype(np.float32))
    np.save(outpath / "sensor_dn.npy", dn.astype(np.uint16 if bit_depth <= 16 else np.uint32))

    # Show on screen last (so saves are ensured even if the window is closed)
    plt.show()

    print(f"[OK] Saved outputs to: {outpath.resolve()}")
    print(f"SNR ≈ {snr_db:.2f} dB | DN range [{int(dn.min())} .. {int(dn.max())}] @ {bit_depth} bits")


def parse_args() -> argparse.Namespace:
    """Small CLI for the classic baseline."""
    p = argparse.ArgumentParser(description="Classic Imaging Pipeline (scene → optics → sensor)")
    p.add_argument("--scene", default="siemens_star",
                   help="slanted_edge | barcode | gradient | siemens_star | checker")
    p.add_argument("--size", type=int, default=512, help="scene canvas size (pixels)")
    p.add_argument("--sigma", type=float, default=1.2, help="optics Gaussian sigma (pixels)")
    p.add_argument("--bit_depth", type=int, default=12, help="sensor ADC bit depth")
    p.add_argument("--outdir", default="outputs", help="directory to save outputs")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_once(args.scene, args.size, args.sigma, args.bit_depth, args.outdir)


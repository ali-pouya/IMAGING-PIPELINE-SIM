"""
Classic Imaging Pipeline (clean baseline)
scene → optics → sensor

WHY THIS FILE
-------------
This is the miaimal, point to build on. It wires together:
  1) Scene generation (normalized float32 [0..1])
  2) Optics blur via a Gaussian PSF surrogate (energy‐normalized kernel)
  3) Sensor pipeline (photons → electrons with noise → DN)

RUN FROM REPO ROOT
------------------
  python -m src.main
  python -m src.main --scene siemens_star --size 512 --sigma 0.8 --bit_depth 12

OUTPUT
------
A 3-panel figure (Scene / After Optics / Sensor DN) with a quick SNR estimate.

NOTES ON SIMPLIFICATIONS
------------------------
• The optics stage uses a Gaussian surrogate for the diffraction core and adds
  defocus/aberration via variance addition; see optics_model.py for references.
• The sensor stage models pixel-aperture MTF (rectangular), Poisson shot noise,
  dark current + DSNU, and read noise. ADC is linear with a black level.

EXTENDING LATER
---------------
• Add other PSFs (defocus disk, astigmatism, coma kernels, etc.) and expose a
  --model switch. Keep this file light; put math in the modules.
"""

from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Package-local imports (relative to 'src')
from imaging_pipeline.scenes.scene_generator import generate_scene
from imaging_pipeline.optics.optics_model import apply_optics_blur
from imaging_pipeline.sensor.sensor_model import add_sensor_effects
from imaging_pipeline.utils.metrics_module import compute_snr


def run_once(scene_kind: str, size: int, sigma: float, bit_depth: int) -> None:
    """
    Execute one scene→optics→sensor pass and show a 3-panel visualization.

    Parameters
    ----------
    scene_kind : str
        One of: slanted_edge | barcode | gradient | siemens_star | checker
    size : int
        Square canvas size for the scene.
    sigma : float
        Gaussian sigma in *pixels* used by the optics stage (classic mode).
    bit_depth : int
        Sensor ADC bit depth (8/10/12/16 typical).
    """
    # 1) Scene: normalized reflectance [0..1]
    scene = generate_scene(kind=scene_kind, size=size).astype(np.float32)

    # 2) Optics: energy-normalized PSF so average brightness is preserved
    blurred, _ = apply_optics_blur(
        scene,
        model="gaussian",      # accepted for back-compat; only 'gaussian' implemented
        sigma=float(sigma),    # classic path: direct sigma override (pixels)
        return_psf=True
    )
    blurred = np.clip(blurred, 0.0, 1.0).astype(np.float32)

    # 3) Sensor: photons→electrons (with noise)→DN; returns electrons, dn
    _, dn = add_sensor_effects(
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
        shot_noise=True,  # accepted for legacy API; shot noise is always modeled
    )

    # Normalize DN to [0..1] for plotting & compute a quick SNR vs pre-noise image
    max_dn = float((1 << bit_depth) - 1)
    sensor_norm = dn.astype(np.float32) / max_dn
    snr_db = compute_snr(sensor_norm, blurred)

    # --- 3-up visualization: Scene / After Optics / Sensor DN ---
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(scene, cmap="gray", vmin=0, vmax=1);  axs[0].set_title("Scene");            axs[0].axis("off")
    axs[1].imshow(blurred, cmap="gray", vmin=0, vmax=1);axs[1].set_title("After Optics");     axs[1].axis("off")
    axs[2].imshow(sensor_norm, cmap="gray", vmin=0, vmax=1)
    axs[2].set_title(f"Sensor DN (SNR≈{snr_db:.1f} dB)"); axs[2].axis("off")
    fig.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    """CLI for the classic baseline."""
    p = argparse.ArgumentParser(description="Classic Imaging Pipeline (scene → optics → sensor)")
    p.add_argument("--scene", default="siemens_star",
                   help="slanted_edge | barcode | gradient | siemens_star | checker")
    p.add_argument("--size", type=int, default=512, help="scene canvas size (pixels)")
    p.add_argument("--sigma", type=float, default=1.2, help="optics Gaussian sigma (pixels)")
    p.add_argument("--bit_depth", type=int, default=12, help="sensor ADC bit depth")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_once(args.scene, args.size, args.sigma, args.bit_depth)


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
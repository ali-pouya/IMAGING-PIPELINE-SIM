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

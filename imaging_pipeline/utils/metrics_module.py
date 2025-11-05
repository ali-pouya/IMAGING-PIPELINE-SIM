# imaging_pipeline/utils/metrics_module.py
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift

def compute_snr(noisy: np.ndarray, clean: np.ndarray) -> float:
    """
    Global SNR = 20*log10( RMS(signal) / RMS(noise) )
    Assumes 'clean' is the pre-noise image aligned to 'noisy'.
    """
    noisy = np.asarray(noisy, dtype=np.float64)
    clean = np.asarray(clean, dtype=np.float64)
    noise = noisy - clean
    sig_rms = np.sqrt(np.mean(clean**2) + 1e-12)
    noise_rms = np.sqrt(np.mean(noise**2) + 1e-12)
    return 20.0 * np.log10(sig_rms / noise_rms)
'''
def mtf_from_fft(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Crude, orientation-averaged MTF via radial average of |FFT|.
    This func returns (freq_norm, mtf), where freq_norm is normalized spatial frequency [0..0.5].
    """
    img = np.asarray(image, dtype=np.float64)
    img = img - img.mean()  # remove DC bias before FFT magnitude normalize
    F = np.abs(fftshift(fft2(img)))
    F /= F.max() + 1e-12

    h, w = F.shape
    cy, cx = h//2, w//2
    yy, xx = np.indices(F.shape)
    rr = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    rmax = min(cy, cx)
    rbins = np.arange(0, rmax + 1)
    # radial average
    mtf = np.zeros_like(rbins, dtype=np.float64)
    counts = np.zeros_like(rbins, dtype=np.float64)
    r_int = rr.astype(int)
    np.add.at(mtf, r_int, F)
    np.add.at(counts, r_int, 1)
    mtf = np.divide(mtf, counts, out=np.zeros_like(mtf), where=counts > 0)
    # Normalize frequency to Nyquist (0..0.5 cycles/pixel)
    freq_norm = (rbins / rmax) * 0.5
    # Normalize MTF to 1 at zero frequency
    if mtf[0] > 0:
        mtf = mtf / mtf[0]
    return freq_norm, mtf
'''
def mtf_from_fft(img: np.ndarray):
    """
    Orientation-averaged MTF via radial averaging of |FFT(img)|.
    Returns (f_norm, mtf) with f_norm in [0, 0.5] cycles/pixel (Nyquist).
    """
    img = np.asarray(img, dtype=np.float32, order="C")
    H, W = img.shape

    # rFFT: non-negative fx; signed fy
    F = np.fft.rfft2(img)
    Mag = np.abs(F)

    fy = np.fft.fftfreq(H, d=1.0)       # [-0.5, 0.5)
    fx = np.fft.rfftfreq(W, d=1.0)      # [0, 0.5]
    FX, FY = np.meshgrid(fx, fy, indexing="xy")
    R = np.sqrt(FX**2 + FY**2)

    # Keep ≤ Nyquist
    mask = (R <= 0.5 + 1e-12)
    Rm = R[mask]
    Mm = Mag[mask]

    n_bins = fx.size                     # W//2 + 1
    r_idx = np.floor(Rm * (n_bins - 1) / 0.5).astype(np.int64)
    r_idx = np.clip(r_idx, 0, n_bins - 1)

    mtf_sum = np.bincount(r_idx, weights=Mm, minlength=n_bins)
    counts  = np.bincount(r_idx, minlength=n_bins)
    mtf = np.divide(mtf_sum, counts, out=np.zeros_like(mtf_sum, dtype=np.float64), where=counts>0)

    # Normalize by DC (bin 0). If DC is zero (rare), fall back to max=1.
    dc = mtf[0]
    if dc > 0:
        mtf = mtf / dc
    else:
        m = mtf.max()
        if m > 0: mtf = mtf / m

    f_norm = fx.astype(np.float32)
    return f_norm, mtf.astype(np.float32)


def plot_histogram(img: np.ndarray, title: str = "Histogram", bins: int = 256):
    """
    Safe histogram for images in either [0..1] or integer DN.
    """
    x = np.asarray(img).ravel()
    plt.figure(figsize=(6,4))
    plt.hist(x, bins=bins, density=False)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.tight_layout()
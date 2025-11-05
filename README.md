# Imaging Pipeline (scene → optics → sensor → metrics)

Simulates an end-to-end imaging chain and produces visual outputs and basic metrics.

## Public APIs
- `imaging_pipeline.scenes.scene_generator.generate_scene(kind: str, size: int, **kwargs) -> np.float32 image [0..1]`
- `imaging_pipeline.optics.optics_model.apply_optics_blur(image, **kwargs) -> blurred[, psf]`
- `imaging_pipeline.sensor.sensor_model`  
  - `SensorParams` (pixel geometry, exposure, noise, ADC, pixel MTF)  
  - `add_sensor_effects(irradiance_W_m2: np.ndarray, params: SensorParams) -> (electrons, dn)`

## Quick Start (full pipeline)
From the project root (where `main.py` lives):
```bash
python main.py --scene siemens_star --size 512 --bit_depth 12 --sigma 0.6
```

**Outputs** (default `outputs/`):
- `pipeline_overview.png` – scene, post-optics image, sensor output (montage)  
- `histogram.png` – histogram of normalized DN  
- `mtf_fft.png` – orientation-averaged MTF (FFT magnitude)  
- `.npy` arrays — `scene.npy`, `after_optics.npy`, `sensor_electrons.npy`, `sensor_dn_norm.npy`

> Tip: redirect with `--outdir <folder>`.

## Common Runs (scene → optics → sensor → metrics)

| # | Command | Purpose / Visual Behavior | Key Output Files |
|:-:|:--|:--|:--|
| 1 | `python main.py --scene siemens_star --size 512 --bit_depth 12 --sigma 0.6` | Baseline diffraction-limited case; circular Siemens star; good for MTF validation. | `pipeline_overview.png`, `mtf_fft.png`, `histogram.png` |
| 2 | `python main.py --scene barcode --size 512 --bit_depth 12 --sigma 0.8` | Realistic barcode imaging; good for DOF & SNR behavior. | same as above |
| 3 | `python main.py --scene checker --size 256 --bit_depth 10 --sigma 0.5` | High-contrast tiles; useful for quantization & dynamic-range checks. | same as above |
| 4 | `python main.py --scene slanted_edge --size 512 --bit_depth 12 --sigma 0.7` | Edge-based MTF validation. | `pipeline_overview.png`, `mtf_fft.png` |
| 5 | `python main.py --scene siemens_star --size 512 --bit_depth 16 --sigma 0.6` | High dynamic range (16-bit); observe smoother quantization. | `histogram.png`, `sensor_dn_norm.npy` |
| 6 | `python main.py --scene siemens_star --size 1024 --sigma 0.5 --outdir results/hires` | High-resolution star; good for aliasing/debug. | files under `results/hires/` |
| 7 | `python main.py --scene siemens_star --bit_depth 8` (then 10, 12) | Bit-depth sweep: compare histogram resolution. | compare `histogram.png` |

## Scene-Specific CLI Flags
- **`slanted_edge`**:  
  `--slanted_angle_deg <float>` (default 5.0), `--slanted_threshold <float>` (default 0.0)
- **`siemens_star`**:  
  `--siemens_spokes <int>` (default 55)
- **`checker`**:  
  `--checker_square_px <int>` (default 16), `--checker_invert`
- **`custom`**:  
  `--custom_path <file>` (PNG/JPG/…), optional `--custom_keep_aspect`  
  If `--scene custom` is used **without** `--custom_path`, the pipeline **falls back** to `gradient` and prints a warning.

**Examples**
```bash
python main.py --scene checker --checker_square_px 24 --checker_invert --size 512 --sigma 0.6
python main.py --scene custom --custom_path "data/photo.jpg" --custom_keep_aspect --size 512
python main.py --scene slanted_edge --slanted_angle_deg 4.5 --slanted_threshold 0.0
python main.py --scene siemens_star --siemens_spokes 72 --sigma 0.7
```

## Stage Tests (unit-test mode)
```bash
python main.py --test scene     # scene gallery (slanted_edge, barcode, gradient, siemens_star, checker)
python main.py --test optics    # impulse→PSF + slanted edge blur
python main.py --test sensor    # SNR scaling & pixel MTF ON/OFF
python main.py --test metrics   # FFT-based MTF sanity check
python main.py --test all       # run all quick tests
```

## Adding Scenes (checkerboard & custom input)
- **Checkerboard** — `generate_checker_scene(size=512, square_px=16)` → binary tiles for aliasing/quantization tests.  
- **Custom Input** — `generate_custom_scene(path="path/to/image.png", size=512)` → load, grayscale to `[0,1]`, resize (nearest-neighbor). Also accepts a NumPy array via `array=<np.ndarray>`.

See `imaging_pipeline/scenes/scene_generator.py` for details.

## Notes
- `--sigma` is wired to the optics `defocus_sigma`.  
- To avoid an apparent “jump” at f=0 in MTF plots from the DC spike, subtract the mean before FFT (already applied in `test_metrics()`).  
- Figures are closed after saving to prevent Matplotlib handle buildup.

---

© Ali Pouya Imaging Pipeline — educational & experimental use.

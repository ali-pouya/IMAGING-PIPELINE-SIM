<p align="center">
  <img src="assets/img/pipeline_banner.png" alt="Imaging Pipeline Simulator — scene → optics → sensor → metrics">
</p>
<p align="center"><em>Scene → Optics → Sensor → Metrics — end-to-end modeling for imaging systems</em></p>

# 📷 Imaging Pipeline Simulator
**scene → optics → sensor → metrics**

This project simulates a **complete imaging chain**, modeling how light from a scene passes through optics and is captured by a sensor — then analyzed via imaging metrics such as **MTF (Modulation Transfer Function)**.  
It’s both a **learning platform** and a **validation sandbox** for imaging engineers, combining physical modeling and numerical analysis in Python.

---

## 🧭 Overview
The simulator reproduces the key transformations in a digital camera system:

| Stage | Module | Description |
|:--|:--|:--|
| **Scene** | `scene_generator` | Generates test patterns (Siemens star, checkerboard, barcode, slanted edge, custom) |
| **Optics** | `optics_model` | Applies blur/PSF based on Gaussian or defocus parameters |
| **Sensor** | `sensor_model` | Converts irradiance to electrons, applies noise, quantization, and pixel MTF |
| **Metrics** | `metrics_module` | Computes FFT-based and edge-based MTFs, contrast curves, and basic SNR plots |

> This codebase was developed as a foundation for **optical metrology, imaging system validation, and educational visualization** of system-level imaging concepts.

---

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/ali-pouya/IMAGING-PIPELINE-SIM.git
cd IMAGING-PIPELINE-SIM

# Create & activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -e .

# Run the simulator
python src/main.py --scene siemens_star --size 512 --bit_depth 12 --sigma 0.6
```

**Outputs** (in `outputs/`):
- `pipeline_overview.png` — Scene → Post-optics → Sensor stages side-by-side  
- `histogram.png` — Normalized DN histogram  
- `mtf_fft.png` — Orientation-averaged MTF  
- `.npy` arrays — Intermediate irradiance and sensor data  

---

## 🧩 Example Runs

| # | Command | Purpose | Output |
|:-:|:--|:--|:--|
| 1 | `python src/main.py --scene siemens_star --sigma 0.6` | Baseline diffraction-limited test | `pipeline_overview.png`, `mtf_fft.png` |
| 2 | `python src/main.py --scene barcode --sigma 0.8` | Realistic barcode imaging (DOF/SNR test) | `pipeline_overview.png` |
| 3 | `python src/main.py --scene checker --size 512 --bit_depth 10 --sigma 0.5` | Dynamic range & quantization | `histogram.png` |
| 4 | `python src/main.py --scene slanted_edge --sigma 0.7` | Edge-based MTF validation | `mtf_fft.png` |

---

## 🧠 Scene-Specific Options
| Scene | Key CLI Flags |
|:--|:--|
| **slanted_edge** | `--slanted_angle_deg <float>` (default 5.0) |
| **siemens_star** | `--siemens_spokes <int>` (default 55) |
| **checker** | `--checker_square_px <int>` (default 16)` ` `--checker_invert` |
| **custom** | `--custom_path <file>` (PNG/JPG), optional `--custom_keep_aspect` |

---

## 🔬 Technical Highlights
- Modular architecture (`scenes`, `optics`, `sensor`, `metrics`)
- Configurable PSF (Gaussian / defocus)
- Sensor modeling with shot + read noise
- Slanted-edge and FFT-based MTF
- Realistic quantization and bit-depth sweeps
- NumPy, OpenCV, and Matplotlib backbone
- Future UI planned via **Streamlit**

---

## 📂 Project Structure
```
src/
 └── imaging_pipeline/
      ├── scenes/           # Pattern generators (star, edge, checker, barcode)
      ├── optics/           # PSF convolution and defocus models
      ├── sensor/           # Pixel, noise, ADC simulation
      ├── utils/metrics_module.py
      └── __init__.py
src/main.py                 # Command-line entry point
requirements.txt
pyproject.toml
```

---

## 📈 Applications
- Educational visualization of optical imaging theory  
- Algorithm testing (e.g. deblurring, autofocus, HDR reconstruction)  
- Camera/lens simulation for barcode, ophthalmic, and industrial imaging systems  
- Rapid prototyping for **EDoF** or **autofocus** module evaluation  

---

## 📜 License
Distributed under the **MIT License** — free for academic and experimental use.

---

## 👤 Author
**Ali Pouya**  
Optical Engineer — Optics & Metrology System Design  
- GitHub: [@ali-pouya](https://github.com/ali-pouya)  
- Project: *Imaging Pipeline Simulator (Classic Edition)*  

> *Built for exploration, education, and optical engineering insight.*

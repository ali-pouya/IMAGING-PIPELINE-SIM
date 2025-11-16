<p align="center" style="margin-top:20px;">
  <img src="assets/img/pipeline_banner.png" width="70%" style="border-radius:12px; box-shadow:0 2px 6px rgba(0,0,0,0.15);">
</p>

<h1 align="center"> Imaging Pipeline Simulator</h1>
<p align="center"><em>Scene → Optics → Sensor → Metrics</em></p>

<hr style="border:0.5px solid #ccc; margin:30px 0;">

## 📑 Quick Navigation
- [1. Summary](#summary)
- [2. Pipeline Overview & System Architecture](#pipeline-overview-system-architecture)
- [3. Module Architecture](#module-architecture)
- [4. Scene Modeling Theory](#scene-modeling-theory)
- [5. Optics Modeling Theory](#optics-modeling-theory)
- [6. Sensor Modeling Theory](#sensor-modeling-theory)
- [7. Metrics and Analysis Theory](#metrics-and-analysis-theory)
- [8. CLI Usage, Reference Experiments, and Workflows](#reference-use)
- [9. Extensibility](#extensibility-advanced-development)
- [Intended Uses](#intended-uses)
- [License](#license)
- [Author](#author)

<hr style="border:0.5px solid #ccc; margin:30px 0;">

<h1 id="summary" align="center">📘 1. Summary</h1>

The *Imaging Pipeline Simulator* is an end-to-end model of digital image formation.  
It was originally built to visualize and compare imaging behavior between two OnSemi CMOS sensors with different sensor parameters—and has since grown into a general-purpose simulation toolbox.

At its core, the pipeline models the full chain:

**Scene → Optics → Sensor → Sampling → Metrics**

Each stage is represented explicitly, enabling controlled, deterministic experiments where every contributing factor (blur, noise, pixel geometry, quantization, and sampling) can be isolated and studied independently. This simulator provides an environment where you can:

- **Trace irradiance formation with full physical transparency**  
  (ideal scenes, continuous irradiance, energy-normalized PSFs)

- **Model optical degradation using interpretable PSFs**  
  Gaussian as a baseline, with extension paths to Airy, defocus, and Zernike-derived aberrations

- **Simulate realistic sensor physics**  
  photon statistics, shot noise, read noise, pixel-aperture MTF, full-well behavior, conversion gain, and quantization

- **Analyze resolution and spectral behavior**  
  through ISO-style slanted-edge MTF, FFT-based falloff, aliasing exposure, and system-MTF composition

The goal is not to mimic camera pipelines from industry OEMs, but to provide a **transparent, mathematical reference model**.  
This was designed to understand and visualize **why** an imaging system behaves the way it does—before adding complexity such as color pipelines, demosaicing, tone-mapping, or sharpening.

#### **Intended uses include:**

- comparing imaging with different sensors  
- studying spatial-resolution limits under controlled blur  
- testing aliasing behavior  
- validating algorithms against known ground truth  
- visualizing image-formation physics  

<hr style="border:0.5px solid #ccc; margin:30px 0;">

<h1 id="pipeline-overview-system-architecture" align="center">🧭 2. Pipeline Overview &amp; System Architecture</h1>

The imaging pipeline follows a deterministic sequence of transformations that reflect the signal-formation pathway of real digital imaging systems.
The process is organized into **four conceptual subsystems**:  
<div align="center">

### **Scene → Optics → Sensor → Metrics**
</div>
Each subsystem operates on well-defined physical quantities.The design promotes traceability from theoretical models to numerical implementation.



## **2.1 High-Level Data Flow**

<div align="center">

```text
┌────────────────┐   ┌──────────────────────┐   ┌─────────────────────────┐   ┌───────────────────┐
│   SCENE MODEL  │-->│    OPTICAL SYSTEM    │-->│      IMAGE SENSOR       │-->│  METRIC ANALYSIS  │
│(irradiance map)│   │ (PSF convolution: h) │   │ (electrons → DN output) │   │(SNR, MTF, spectra)│
└────────────────┘   └──────────────────────┘   └─────────────────────────┘   └───────────────────┘
```

</div>

Let:

- $S(x, y)$: scene irradiance in normalized units  
- $h(x, y)$: point spread function  
- $I_{\mathrm{opt}}(x, y) = (S * h)(x, y)$: optically blurred irradiance  
- $N_e(x, y)$: electron map  
- $DN(x, y)$: quantized digital output  


## **2.2 Mathematical Formulation of the Pipeline**


$$I_{\mathrm{opt}}(x,y) = (S * h)(x,y)$$

$$N_e = I_{\mathrm{opt}} \cdot A_{\mathrm{pix}} \cdot t_{\mathrm{exp}} \cdot QE$$

Shot noise:

$$N_e^{\prime} \sim \mathrm{Poisson}(N_e)$$

Read noise:

$$N_e^{\mathrm{noisy}} = N_e^{\prime} + \mathcal{N}(0, \sigma_r^2)$$

Quantization:

$$DN = \mathrm{clip}\left(\left\lfloor \frac{N_e^{\mathrm{noisy}}}{CG} \right\rceil + BL,\ 0,\ 2^B - 1\right)$$

<br>

## **2.3 Repository Architecture**

```text
src/
│
├── main.py                 # Primary pipeline demonstration
├── main_classic.py         # Minimal teaching version
├── main_full.py            # Batch/testing version
│
└── imaging_pipeline/
      ├── scenes/
      │     └── scene_generator.py
      │
      ├── optics/
      │     └── optics_model.py
      │
      ├── sensor/
      │     └── sensor_model.py
      │
      └── utils/
            └── metrics_module.py
```


## **2.4 Execution Flow in `main.py`**

The script performs:

| Step | Operation | Output |
|------|-----------|--------|
| 1 | Scene construction | irradiance map |
| 2 | Optical degradation | blurred irradiance |
| 3 | Sensor modeling | electrons + DN |
| 4 | Metric computation | SNR, diagnostic data |
| 5 | Export states | `.npy` tensors |



<hr style="border:0.5px solid #ccc; margin:30px 0;">

<h1 id="module-architecture" align="center">🧩 3. Module Architecture</h1>

The simulator contains distinct modules directly corresponding to physical image-formation stages.

## **3.1 `main.py` — Primary Execution Path**

```text
Scene → Optics → Sensor → Metrics
```

Operations:

- deterministic scene generation  
- Gaussian PSF convolution  
- sensor noise + quantization  
- global SNR measurement  
- intermediate state export  



## **3.2 `main_classic.py` — Minimal Version**


Simplifies:

- core pipeline  
- no batch mode  
- minimal parameters  



## **3.4 Scene Module — `scenes/scene_generator.py`**

Scene types: (range `[0,1]`, `float32`, and deterministic)

| Scene | Purpose |
|--------|---------|
| Siemens star | radial frequency coverage, shows aliasing radially |
| Slanted edge | produces ESF → LSF → ISO MTF |
| Checkerboard | stress test harmonic preservation |
| Barcode | 1D high-freq pattern (blur sensitivity) |
| Gradient | tonal + ADC tests |
| Custom | arbitrary irradiance |



## **3.5 Optics Module — `optics/optics_model.py`**

Applies PSF via **spatial-domain convolution**. Spatial convolution avoids FFT wrap-around artifacts.

Models supported:

- Gaussian PSF (implemented)  
- Defocus disk  
- Airy diffraction  
- Zernike aberrated  
- Polychromatic weighted  

Gaussian PSF:

$$
h(x,y)=\frac{1}{2\pi\sigma^2}\exp\left(-\frac{x^2+y^2}{2\sigma^2}\right)
$$

MTF:

$$\mathrm{MTF}_{\mathrm{gauss}}(f)=\exp[-2(\pi\sigma f)^2]$$



## **3.6 Sensor Module — `sensor/sensor_model.py`**


####  Sensor Processing Stages

| Stage | Formula / Operation | Purpose |
|-------|----------------------|---------|
| Electron generation | $N_e = I_{\mathrm{opt}} A_{\mathrm{pix}} t_{\mathrm{exp}} QE$ | Convert irradiance to electrons |
| Shot noise | $N_e' \sim \mathrm{Poisson}(N_e)$ | Photon arrival randomness |
| Read noise | $N_e^{\mathrm{noisy}} = N_e' + \mathcal{N}(0,\sigma_r^2)$ | Electronic noise floor |
| Pixel-aperture MTF | Spatial averaging (box filter) | Models finite pixel size |
| Quantization | $DN =\mathrm{clip}\left(\left\lfloor\frac{N_e^{\mathrm{noisy}}}{CG}\right\rceil+ BL,\ 0,\ 2^B - 1\right)$ | ADC conversion |
| Saturation | clamp to FWC | Prevents overflow |


## **3.7 Metrics Module — `utils/metrics_module.py`**

Two families of metrics:

| Metric | Method | Use |
|--------|--------|-----|
| SNR | global variance-ratio | noise evaluation |
| FFT-MTF | radial FFT magnitude | spectral attenuation |
| Edge-MTF | ESF → LSF → MTF | resolution measurement |

> **Note**  
> Edge-based MTF aligns with ISO 12233 methodology.



<hr style="border:0.5px solid #ccc; margin:30px 0;">

<h1 id="scene-modeling-theory" align="center">🎨 4. Scene Modeling Theory</h1>

Analytic scenes provide controlled spatial frequencies and deterministic reproducibility.

## **4.1 Purpose of Analytic Scenes**

| Purpose | Explanation |
|--------|-------------|
| Controlled spatial-frequency content | Predictable propagation through optics + sensor |
| Deterministic behavior | Same inputs → same outputs |
| Convolution compatibility | Sharp edges, periodic patterns, ramps |
| Alignment with test targets | Siemens star, ISO edge, checkerboard, barcode |

**Scene irradiance interpretation**

In this simulator, \(S(x,y)\) is defined as the **ideal, blur-free, noise-free, distortion-free irradiance** at the sensor plane.  
Physically, it corresponds to the image that would be formed if the optics were perfect (no aberrations, diffraction, or defocus) and the sensor introduced no noise or quantization.  
This separation makes it possible to attribute all subsequent resolution loss and noise strictly to the optics, sensor, and sampling chain.



## **4.2 Scene Types and Definitions**

### **4.2.1 Siemens Star**

$$S(x,y)=\frac{1}{2}\left[1+\mathrm{sign}(\cos(N\theta))\right]$$

<div align="center">

where $\theta = \mathrm{atan2}(y,x)$ and $N$ is the number of radial spokes.

</div>

| Property | Meaning |
|----------|---------|
| Radial frequency gradient | frequencies increase inward |
| High aliasing sensitivity | reveals sampling issues |
| Blur isotropy test | useful for verifying symmetry |



### **4.2.2 Slanted Edge**

Binary transition rotated by angle $\theta$:

$$
S(x,y)=
\begin{cases}
1, & x\cos\theta + y\sin\theta > 0 \\
0, & x\cos\theta + y\sin\theta \le 0
\end{cases}
$$

| Property | Meaning |
|----------|---------|
| Subpixel sampling | used for ESF oversampling |
| ISO-12233 compatibility | industry-standard |
| Sharp transition | ideal for LSF/MTF extraction |



### **4.2.3 Checkerboard**

$$
S(x,y)=
\begin{cases}
1, & \bigl(\lfloor x/p \rfloor + \lfloor y/p \rfloor\bigr) \bmod 2 = 0 \\
0, & \text{otherwise}
\end{cases}
$$

<div align="center">

where $p$ is the block period.

</div>

| Property | Meaning |
|----------|---------|
| Harmonics | strong odd/even components |
| DR + quantization tests | reveals banding |
| Aliasing visibility | clear folding patterns |


### **4.2.4 Barcode Pattern**

| Property | Meaning |
|----------|---------|
| 1D high-frequency | sensitive to blur |
| DOF indicator | blur changes readability |
| Extreme aspect ratio | stresses PSF model |


### **4.2.5 Gradient Patterns**

A linear irradiance ramp such as: $S(x,y)=\frac{x}{W}$ or $S(x,y)=\frac{x+y}{H+W}$.

| Property | Meaning |
|----------|---------|
| Low-frequency ramp | tone + histogram tests |
| Quantization reveals banding | ADC artifacts |
| Useful for noise visualization | smooth backgrounds |


### **4.2.6 Custom Scenes**

> **Note**  
> Any grayscale image can be mapped to $[0,1]$ and used as a scene.


## **4.3 Frequency-Domain Properties of Scenes**

| Scene | Dominant Content | Primary Use |
|--------|------------------|-------------|
| Siemens star | radially increasing harmonics | resolution & aliasing |
| Slanted edge | single sharp boundary | MTF extraction |
| Checkerboard | harmonic-rich | quantization + DR |
| Barcode | narrowband 1D | DOF & blur |
| Gradient | low-frequency ramp | tone & ADC |


## **4.4 Interface Requirements**

| Requirement | Description |
|-------------|-------------|
| Output type | `float32` |
| Range | $[0,1]$ |
| Deterministic | yes |
| Convolution-ready | edges continuous |
| Pixel-aligned | spatially consistent |


<hr style="border:0.5px solid #ccc; margin:30px 0;">

<h1 id="optics-modeling-theory" align="center">🔬 5. Optics Modeling Theory</h1>

Optical effects are simulated via convolution with a point spread function (PSF).  

## **5.1 Optical Transformation Framework**

$$I_{\mathrm{opt}}(x,y) = (S * h)(x,y)$$

**Linear, shift-invariant assumption**

The optics are modeled as a linear, shift-invariant (LSI) system over the simulated field. This implies that:

- blur behaves the same at every location in the image, and  
- the effect of the lens can be written as the **superposition of many blurred points**.

This approximation is not exact for wide-angle, strongly off-axis, or heavily aberrated systems, but it is accurate for **small fields of view** and **center-of-field synthetic experiments**, and is standard practice in image-formation modeling.

**Physical contributors to PSF shape**

In real imaging systems, the PSF aggregates multiple physical effects, including:

- diffraction from the finite aperture  
- defocus and circle-of-confusion blur  
- low- and high-order aberrations (spherical, coma, astigmatism, trefoil, etc.)  
- manufacturing tolerances and alignment errors  
- sensor microlenses and cover glass  
- wavelength-dependent behavior and chromatic dispersion  
- motion blur, which can be treated as a temporal PSF

| Principle | Meaning |
|-----------|---------|
| Linear shift-invariant | constant PSF across field |
| Energy normalized | $\iint h(x,y)\,dx\,dy = 1$ |
| Spatial convolution | avoids FFT wrap-around artifacts |

**Why PSFs must be energy-normalized**

If the PSF is not normalized to unit integral, convolution will artificially brighten or darken the image.  
Non-unit-energy PSFs introduce:

- exposure drift when changing blur parameters,  
- brightness inconsistency between experiments, and  
- unphysical gain or loss of radiant energy.

Enforcing

$$
\iint h(x,y)\,dx\,dy = 1
$$

ensures irradiance conservation and keeps comparisons between different blur models physically meaningful.


## **5.2 Gaussian PSF (Implemented Model)**

The simulator implements an energy-normalized Gaussian PSF:

$$h_{\text{gauss}}(x, y)= \frac{1}{2\pi\sigma^{2}}\exp\left(-\frac{x^{2} + y^{2}}{2\sigma^{2}}\right)$$

where  
- $\sigma$ is expressed in pixel units,  
- kernel radius is selected to approximate infinite support.  

Gaussian blur functions as a surrogate for aggregated optical effects such as small defocus, minor manufacturing deviations, and residual aberrations.

### **Gaussian MTF**

The corresponding modulation transfer function is:

$$\mathrm{MTF}_{\mathrm{gauss}}(f)=\exp\!\left( -2(\pi\sigma f)^2 \right)$$

A useful rule of thumb links the Gaussian width \(\sigma\) (in pixels) to the spatial frequency at which contrast drops to 50% (MTF50):

$$
f_{50} \approx \frac{0.32}{\sigma} \quad [\text{cycles per pixel}]
$$

Smaller \(\sigma\) values correspond to sharper imagery (higher MTF50), while larger \(\sigma\) values model increased blur.

| Property | Meaning |
|----------|---------|
| Closed-form MTF | easy validation |
| Approx. optical blur | surrogate for real aberrations |
| Controlled blur strength | via $\sigma$ |


## **5.3 Defocus PSF (Circle of Confusion) — Extensible**

Geometric defocus produces a uniformly illuminated disk:

$$
h_{\mathrm{defocus}}(r)=
\begin{cases}
\dfrac{1}{\pi R^2}, & r \le R \\
0, & r > R
\end{cases}
$$

where the radius $R$ relates to defocus distance and f-number.

### **Defocus MTF**

$$\mathrm{MTF}_{\mathrm{defocus}}(\nu)=\dfrac{2}{\pi}\left[\arccos(\nu)-\nu\sqrt{1-\nu^2}\right]$$

with $\nu = f / f_{\mathrm{cutoff}}$.


## **5.4 Airy PSF (Diffraction-Limited) — Extensible**

For a circular aperture, diffraction produces an Airy pattern:

$$h_{\mathrm{airy}}(r)=\left[\frac{2 J_1(kr)}{kr}\right]^2$$

where  
- $J_1$ is the Bessel function of the first kind,  
- $k = \dfrac{\pi D}{\lambda f}$.

### **Airy MTF**

Diffraction-limited MTF takes the form:

$$\mathrm{MTF}_{\mathrm{diff}}(u)=\dfrac{2}{\pi}\left[\arccos(u)-u\sqrt{1-u^2}\right]$$

where the cutoff frequency is:
$$f_{\mathrm{cutoff}}=\frac{1}{\lambda N}$$

with f-number $N$.


## **5.5 Aberrated PSFs — Zernike Wavefronts**

Wavefront:

$$W(\rho,\theta)=\sum_k a_k Z_k(\rho,\theta)$$

Pupil:

$$
P(\rho,\theta)=A(\rho)\exp\left(i\,\frac{2\pi}{\lambda}\,W(\rho,\theta)\right)
$$

where $A(\rho)$ describes aperture geometry.

The PSF follows from the Fourier transform relationship:

PSF:

$$
h(x,y)=\left|\mathcal{F}\left(P(\rho,\theta)\right)\right|^{2}
$$

This formulation supports modeling of coma, astigmatism, spherical aberration, trefoil, and higher-order wavefront errors.


## **5.6 Polychromatic PSFs**

A wavelength-weighted PSF may be constructed as:

$$h_{\mathrm{poly}}(x,y)=\sum_{\lambda}w(\lambda)\,h_{\lambda}(x,y)$$

where the weighting function reflects illumination spectrum and sensor quantum efficiency.


## **5.7 Optics Implementation Summary**

| Component | Responsibility |
|-----------|----------------|
| PSF generator | Gaussian/defocus/Airy/Zernike |
| Normalization | $\iint h=1$ |
| Convolution | spatial-domain |
| Optional PSF export | for diagnostics |

## **5.8 Handling Undersampled PSFs**

When the PSF is significantly narrower than the pixel pitch (for example, Gaussian blur with \(\sigma \lesssim 0.5\) pixels), a naïve convolution on the sensor grid becomes numerically unreliable:

- the ESF becomes quantized rather than smooth,  
- the corresponding LSF develops spiky structure, and  
- the recovered MTF can exhibit aliasing or over-optimistic resolution.

To preserve the correct sampling order, the simulator uses an **upsample → blur → downsample** strategy in these regimes:

1. upsample the irradiance to a finer grid,  
2. apply the continuous-space PSF blur on the fine grid, then  
3. downsample back to the sensor pixel pitch.

This mirrors the physical process (continuous blur followed by discrete sampling) and yields stable, physically meaningful MTF estimates even when the optical blur is tighter than one pixel.



<hr style="border:0.5px solid #ccc; margin:30px 0;">

<h1 id="sensor-modeling-theory" align="center">🖥️ 6. Sensor Modeling Theory</h1>

Sensor behavior is modeled through:  
- photon → electron conversion  
- Poisson + Gaussian noise  
- pixel-aperture MTF  
- quantization  
- sampling  


## **6.1 Irradiance-to-Electron Conversion**

$$N_e(x,y)=I_{\mathrm{opt}} A_{\mathrm{pix}} t_{\mathrm{exp}} QE$$

| Term | Meaning |
|------|---------|
| $A_{\mathrm{pix}}$ | pixel area |
| $t_{\mathrm{exp}}$ | exposure time |
| $QE$ | quantum efficiency |

This expression assumes uniform pixel response and wavelength-independent QE unless otherwise extended.


## **6.2 Shot Noise**

Photon arrival follows a Poisson process. The shot-noise–perturbed electron count is:

$$N_e' \sim \mathrm{Poisson}(N_e)$$

| Property | Meaning |
|----------|---------|
| signal-dependent noise | variance = mean |
| dominates mid/high light | Poisson behavior |


## **6.3 Read Noise**

Electronic contributions are modeled as additive Gaussian noise:

$$N_e^{\mathrm{noisy}}=N_e' + \mathcal{N}(0,\sigma_r^2)$$

| Noise | Meaning |
|--------|---------|
| Gaussian | independent of illumination |
| electronics-origin | dominates in low light and is independent of signal level |


## **6.4 Full-Well Capacity**

Electron counts are limited by pixel full-well capacity (FWC):

$$N_e^{\mathrm{sat}}(x,y)=\min(N_e^{\mathrm{noisy}},FWC)$$

FWC defines the maximum number of electrons the photodiode can hold before saturation occurs.


## **6.5 Pixel-Aperture MTF**

Each pixel integrates irradiance over its finite geometric extent, imposing a pixel-aperture modulation transfer function. For a rectangular aperture of width $p$:

$$\mathrm{MTF}_{\mathrm{pixel}}(f)=|\mathrm{sinc}(\pi f p)|$$

> **Note**  
> The simulator uses spatial-domain averaging, which is equivalent to convolving with a box kernel.


## **6.6 Quantization**

Electron counts are converted to digital numbers (DN) through:

$$DN(x,y) =\left\lfloor\frac{N_e^{\mathrm{sat}}(x,y)}{CG}\right\rceil+ BL$$

| Term | Meaning |
|------|---------|
| $CG$ | conversion gain |
| $BL$ | black level |
| $2^B-1$ | max DN, the output is clamped to the bit-depth interval |


## **6.7 Sampling + Nyquist**

After quantization, the spatial sampling imposed by the pixel grid restricts representable spatial frequencies to:

$$f_{\mathrm{Nyquist}} = \frac{1}{2p}$$


## **6.8 Global SNR Metric**

$$\mathrm{SNR}_{\mathrm{dB}}=20\log_{10}\left(\frac{\sigma_{\mathrm{signal}}}{\sigma_{\mathrm{noise}}}\right)$$

where  
- “signal” refers to the variance of the noise-free irradiance,  
- “noise” refers to the variance of the difference between noise-free and noisy outputs.

Although not a pixel-wise or frequency-dependent SNR measure, this metric provides a coarse assessment of noise behavior across the full image.

## **6.9 Sensor Summary**

| Stage | Description |
|--------|-------------|
| irradiance → electrons | physical conversion |
| Poisson noise | photon statistics |
| Gaussian noise | electronics |
| aperture integration | pixel MTF |
| quantization | ADC |
| sampling | Nyquist-limited |



<hr style="border:0.5px solid #ccc; margin:30px 0;">

<h1 id="metrics-and-analysis-theory" align="center">📈 7. Metrics and Analysis Theory</h1>

The metrics module analyzes spatial resolution, spectral behavior, and noise performance.

## **7.1 Global SNR**

A global SNR estimate is computed by comparing a noise-free irradiance reference to the corresponding noisy output:

$$\mathrm{SNR}_{\mathrm{dB}}=20 \log_{10}\left(\frac{\sigma_{\mathrm{signal}}}{\sigma_{\mathrm{noise}}}\right)$$

| Term | Meaning |
|------|---------|
| $\sigma_{\mathrm{signal}}$ | std of noise-free irradiance |
| $\sigma_{\mathrm{noise}}$ | std of difference between noisy and clean outputs |


## **7.2 Spatial-Resolution Metrics**

Two frameworks:

| Method | Description | Use |
|--------|-------------|-----|
| FFT-based | radial average of magnitude spectrum | blur magnitude, spectral falloff |
| Edge-based | ESF → LSF → MTF | physically meaningful resolution curve |


## **7.3 FFT-Based MTF**

| Step | Operation |
|------|-----------|
| 1 | compute 2D FFT |
| 2 | magnitude spectrum |
| 3 | radial averaging |
| 4 | normalize by DC |

> **Important**  
> FFT-MTF is not ISO-compliant — it is for qualitative comparison only.


## **7.4 Edge-Based MTF (ISO-Style)**

### **7.4.1 ESF**

A tilted edge provides subpixel sampling of a binary step transition.  
Pixel values along the edge normal are aggregated and binned by fractional-pixel position to produce a smooth ESF:

$$
e(x) = \text{oversampled edge profile}
$$

The Nyquist frequency is:

$$
f_{\mathrm{Nyquist}} = 0.5\ \text{cpp}
$$

### **7.4.2 LSF**

$$l(x) = \frac{d}{dx} e(x)$$

### **7.4.3 MTF**

$$\mathrm{MTF}(f)=|\mathcal{F}\{l(x)\}|$$


## **7.5 System MTF Composition**

$$
\mathrm{MTF}_{\mathrm{system}} =
\mathrm{MTF}_{\mathrm{optics}}\cdot
\mathrm{MTF}_{\mathrm{pixel}}\cdot
\mathrm{MTF}_{\mathrm{sampling}}
$$

| Component | Meaning |
|-----------|---------|
| optics | PSF degradation |
| pixel | aperture filter |
| sampling | Nyquist truncation |


## **7.6 Frequency Units**

| Unit | Interpretation |
|------|----------------|
| cpp | cycles per pixel |
| Nyquist | 0.5 cpp |
| high freq roll-off | dominated by PSF + pixel MTF |


## **7.7 Validation Procedures**

The implemented metrics support several validation procedures:

| Validation | Method |
|------------|--------|
| Gaussian blur | compare MTF to analytic curve |
| Sensor noise | compare SNR trends against $\sigma_{\mathrm{shot}}^2 = N_e$ and $\sigma_r$ |
| Pixel MTF | sinc-shape behavior |
| Sampling | aliasing near Nyquist |



<hr style="border:0.5px solid #ccc; margin:30px 0;">

<h1 id="reference-use" align="center">⚙️ 8. Reference Use </h1>

## **8.1 Reference Examples**

```bash
python src/main.py --scene siemens_star --size 512 --sigma 1.2 --bit_depth 12 --outdir outputs
```


### **Resolution baseline**

```bash
python src/main.py --scene siemens_star --sigma 0.6
```

### **Defocus surrogate**

```bash
python src/main.py --scene barcode --sigma 1.0
```

### **Quantization / DR**

```bash
python src/main.py --scene checker --bit_depth 10 --sigma 0.5
```

### **Slanted-edge MTF**

```bash
python src/main.py --scene slanted_edge --sigma 0.7
```

## **8.2 Output Structure**

| File | Contents |
|-------|----------|
| `pipeline_overview.png` | montage of scene/optics/sensor |
| `scene.npy` | ideal irradiance |
| `after_optics.npy` | blurred irradiance |
| `sensor_electrons.npy` | electrons |
| `sensor_dn.npy` | quantized DN |

<hr style="border:0.5px solid #ccc; margin:30px 0;">

<h1 id="extensibility-advanced-development" align="center">🚀 9. Extensibility and Advanced Development</h1>

## **9.1 Extensible Components**

| Subsystem | Extensible Behavior |
|-----------|----------------------|
| Scene generator | analytic, custom images, multi-spectral |
| Optics model | PSF substitution, multi-wavelength |
| Sensor model | noise sources, PRNU/DSNU, HDR |
| Metrics | ISO-complete MTF, PSD |


## **9.2 Extending Optics**

| Extension | Description |
|-----------|-------------|
| new single-λ PSFs | custom PSF generator |
| Zernike models | aberration simulation |
| spatially varying PSFs | field-split convolution |



## **9.3 Extending Sensor Modeling**

| Feature | Description |
|---------|-------------|
| wavelength-dependent QE | spectral sensitivity |
| PRNU/DSNU | nonuniformities |
| row/column noise | structured noise |
| rolling shutter | temporal integration |


## **9.4 Extending Scene Models**

| Feature | Description |
|---------|-------------|
| PBR | physically based scenes |
| natural images | dataset integration |
| multi-plane irradiance | depth-based scenes |
| time-varying scenes | rolling shutter tests |


## **9.5 Extending Metrics**

| Metric | Description |
|--------|-------------|
| full ISO12233 | full workflow |
| aperture-weighted MTF | f-number behavior |
| temporal metrics | rolling shutter |
| noise PSD | frequency-domain noise |


## **9.6 Implementation Constraints**

| Constraint | Requirement |
|------------|-------------|
| determinism | identical output from identical inputs |
| energy | PSFs normalized |
| dimensions | aligned tensors |
| sampling | consistent pixel pitch |
| API compatibility | same function interfaces |


<hr style="border:0.5px solid #ccc; margin:30px 0;">

<h2 id="intended-uses" align="left">📈 Intended Uses</h2>

- Imaging-chain parameter sweeps  
- Virtual testing  
- Visualization of imaging system behavior  


<h2 id="license" align="left">📜 License</h2>

Distributed under the MIT License.


<h2 id="author" align="left">👤 Ali Pouya</h2>

Optical Engineer — Optics &amp; Metrology System Design.\
GitHub: https://github.com/ali-pouya

---

<h2 id="references" align="left">📚 References</h2>


#### Optics & Fourier Theory
- **Goodman, J. W. (2017). _Introduction to Fourier Optics_ (4th ed.).**  
  Foundational reference for PSF/OTF/MTF relationships, convolution, Gaussian PSF, Fourier-domain behavior.

- **Born, M., & Wolf, E. (1999). _Principles of Optics_ (7th ed.).**  
  Diffraction theory, Airy pattern, pupil function, Fourier propagation.

- **Smith, W. J. (2007). _Modern Optical Engineering_ (4th ed.).**  
  System blur approximations, f/# relations, Gaussian surrogates.


#### ISO Standards & MTF Measurement
- **ISO 12233:2017. _Electronic still picture imaging — Resolution measurements / SFR (Slanted-edge method)._**  
  Reference for ESF → LSF → MTF pipeline, pixel-aperture MTF, frequency conventions.

- **Burns, P. D. (2000). “Slanted-edge MTF for digital camera and scanner analysis.”**  
  Primary description of practical slanted-edge MTF processing.

- **General Siemens star usage (Imatest, vendor app notes).**  
  Radial spatial-frequency targets for qualitative MTF/aliasing visualization.


#### Sensor Modeling, Noise, and Electronics
- **Janesick, J. R. (2007). _Photon Transfer: DN → λ_. SPIE Press.**  
  Photon shot noise, read noise, full-well capacity, conversion gain, and SNR behaviors used in the sensor section.

- **Holst, G. C. (2011). _CMOS/CCD Sensors and Camera Systems_ (2nd ed.). SPIE Press.**  
  Sensor architecture, PRNU/DSNU, dark current, pixel-aperture and sampling considerations.



<hr style="border:0.5px solid #ccc; margin:40px 0;">

<p align="center"><sub>Imaging Pipeline Simulator — MIT License</sub></p>

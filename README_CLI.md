
<h1 id="cli-usage-reference-experiments-workflows" align="center">⚙️ 8. CLI Usage, Reference Experiments, and Workflows</h1>

## **8.1 CLI Overview**

```bash
python src/main.py
```


## **8.2 CLI Parameters**

### **Scene Selection**

```bash
--scene slanted_edge
--scene barcode
--scene gradient
--scene siemens_star
--scene checker
```

### **Core Parameters Table**

| Flag | Description |
|------|-------------|
| `--size` | scene dimension |
| `--sigma` | Gaussian PSF std (px) |
| `--bit_depth` | ADC bit depth |
| `--outdir` | output directory |


## **8.3 Regression Testing**

| Step | Action |
|-------|--------|
| 1 | generate baseline outputs |
| 2 | rerun pipeline |
| 3 | compare SNR, histogram, MTF curves |



## **8.4 Batch Experimentation**

| Sweep | Command |
|--------|---------|
| Sigma sweep | `for sigma in [...] python src/main.py --scene siemens_star ...` |
| Bit-depth sweep | `for b in [...] python src/main.py --scene checker ...` |
| Multi-scene | `for s in [...] python src/main.py --scene $s` |

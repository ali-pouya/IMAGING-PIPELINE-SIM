# Contributing to Camera MTF Bench

Thanks for your interest in contributing!  
This project is designed to be clear, modular, and easy to extend.  
Before submitting changes, please review the guidelines below.

---

## 📦 Project Structure

Key modules:

- `bench.targets` — Siemens star geometry, radial sampling  
- `bench.metrics` — focus metrics, Siemens MTF  
- `bench.instruments` — camera & stage backends  
- `bench.workflows` — autofocus + MTF pipelines  
- `bench.gui` — Streamlit manual focus UI  
- `bench.cli` — command-line interface  

Please keep new modules consistent with the existing architecture.

---

## 🧩 Adding Features

When adding functionality:

- Use clear, self-contained modules or classes  
- Add docstrings (NumPy-style preferred)  
- Keep imports local where possible  
- Avoid breaking existing CLI commands  
- Provide a small example or test when relevant  

If adding a new metric, place it under `bench.metrics`.  
If adding hardware, place it under `bench.instruments`.

---

## 🧪 Tests

Manual test scripts live in `tests/`.  
If you add hardware-dependent code, please:

- implement a mock version, or  
- guard imports with try/except, or  
- add a `--dry-run` mode  

This keeps the project runnable without hardware.

---

## 🔧 Style

- Black formatting recommended  
- Use type hints (Python 3.10+)  
- Avoid overly long functions  
- Keep CLI args descriptive and explicit  

---

## 📬 Pull Requests

Please include:

1. Description of the update  
2. Rationale (what it improves or fixes)  
3. Any relevant usage example  
4. Notes on backward-compatibility  


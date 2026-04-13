# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PuLID for FLUX with Nunchaku SVDQuant — a Gradio web app that generates identity-preserving images using 4-bit quantized FLUX models via the [Nunchaku](https://github.com/nunchaku-ai/nunchaku) inference engine. It combines PuLID (identity transfer from reference photos) with FLUX text-to-image generation at reduced VRAM usage.

## Running the App

```bash
# Launch with default model (flux-krea-dev) on GPU 1
./run_gradio.sh

# Or directly:
python app_flux.py --name flux-krea-dev --device cuda --port 7860

# Model choices: flux-dev, flux-krea-dev
```

## Setup

Python 3.10 with a `.venv` virtual environment. Nunchaku must be installed from the GitHub wheel release (not PyPI — the PyPI package is different):

```bash
pip install https://github.com/nunchaku-ai/nunchaku/releases/download/v1.2.1/nunchaku-1.2.1%2Bcu12.8torch2.8-cp310-cp310-linux_x86_64.whl
pip install -r requirements.txt
```

## Architecture

Single-file application (`app_flux.py`):

- **`FluxGenerator`** — Loads the quantized FLUX transformer via `NunchakuFluxTransformer2dModel`, builds a `PuLIDFluxPipeline`, and monkey-patches the transformer's forward method with `pulid_forward` to enable identity injection. Exposes `generate_image()` for single-image inference.
- **`generate_batch()`** — Gradio callback that runs every combination of uploaded ID images × prompts. Supports per-prompt overrides for `id_weight` and `start_step` via a text field. Saves outputs to `outputs/<timestamp>/` and bundles them as a zip.
- **`create_demo()`** — Builds the Gradio UI.

All model/pipeline code comes from the `nunchaku` package (`nunchaku.models`, `nunchaku.pipeline`, `nunchaku.utils`). Generated images are saved to the `outputs/` directory.

## Key Parameters

- `start_step`: Controls identity fidelity vs editability (0 = max fidelity, 4 = more editability for photorealistic)
- `true_cfg`: 1.0 = fake CFG (default/recommended), >1.0 = true CFG with negative prompt support
- `id_weight`: Controls strength of identity transfer from reference image

# PuLID for FLUX (Nunchaku SVDQuant)

A Gradio web UI for identity-preserving image generation using [PuLID](https://arxiv.org/abs/2404.16022) on top of [FLUX](https://github.com/black-forest-labs/flux), accelerated with [Nunchaku](https://github.com/nunchaku-ai/nunchaku) 4-bit SVDQuant for low VRAM usage.

Upload one or more reference (ID) images, write one or more prompts, and generate every prompt against every face — with optional LoRA and per-image control.

## Features

- **Low VRAM** — 4-bit quantized FLUX transformer via Nunchaku SVDQuant.
- **T5 quantization** — choose `bf16` (~9.5 GB), `int8` (~5 GB), or `nf4` (~3 GB) for the text encoder.
- **Batch generation** — N reference images × M prompts in one run.
- **Per-image control** — override `id_weight` and `start_step` per prompt.
- **LoRA support** — load any `.safetensors` LoRA with adjustable strength.
- **True / fake CFG** — toggle real classifier-free guidance with a negative prompt.
- **Auto-save** — all outputs saved to a timestamped folder + downloadable zip.

## Requirements

- NVIDIA GPU with CUDA
- Python 3.10+
- `torch`, `gradio`, `transformers`, `bitsandbytes`, `nunchaku`, `huggingface_hub`, `Pillow`, `numpy`

```bash
pip install torch gradio transformers bitsandbytes huggingface_hub pillow numpy
# install nunchaku per: https://github.com/nunchaku-ai/nunchaku
```

## Usage

```bash
python app.py
```

Then open the printed local URL in your browser.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--name` | `flux-krea-dev` | Base model: `flux-dev` or `flux-krea-dev` |
| `--device` | `cuda` | Torch device |
| `--port` | `None` | Server port |
| `--lora_path` | `None` | Path to a LoRA `.safetensors` to load at startup |
| `--lora_scale` | `1.0` | LoRA strength |
| `--t5_quant` | `bf16` | T5-XXL quantization: `bf16`, `int8`, or `nf4` |

Example:

```bash
python app.py --name flux-dev --t5_quant nf4 --port 7860
```

## Tips

- **`timestep to start inserting ID`** — lower = higher face fidelity but less editability. Recommended 0–4 (photorealistic: 4, stylized: 0–1).
- **`true CFG scale`** — keep at `1` (fake CFG) for most cases; `>1` enables true CFG with the negative prompt.

## Notes

Model weights are pulled from the Hugging Face cache. The script expects a shared cache at `/home/shared/.cache/huggingface` (set via `HF_HOME`) — adjust the paths at the top of the file for your environment.

## Credits

- [PuLID](https://github.com/ToTheBeginning/PuLID)
- [FLUX](https://github.com/black-forest-labs/flux) by Black Forest Labs
- [Nunchaku](https://github.com/nunchaku-ai/nunchaku)

import os

# Shared cache for Flux models (read-only)
SHARED_CACHE = "/home/shared/.cache/huggingface"

# Local cache for EVA-CLIP
LOCAL_EVA_CACHE = "/home/tanzim/.cache/huggingface"

# Force Hugging Face to look in shared cache by default
os.environ["HF_HOME"] = SHARED_CACHE

import time
import zipfile
from datetime import datetime
from pathlib import Path
from types import MethodType

import gradio as gr
import numpy as np
import torch
from PIL import Image

from nunchaku.models.pulid.pulid_forward import pulid_forward
from nunchaku.models.transformers.transformer_flux import NunchakuFluxTransformer2dModel
from nunchaku.pipeline.pipeline_flux_pulid import PuLIDFluxPipeline
from nunchaku.utils import get_precision

from transformers import BitsAndBytesConfig, T5EncoderModel

from huggingface_hub import snapshot_download


def print_cuda_memory(tag: str):
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    print(f"[CUDA MEM] {tag}: allocated={allocated:.2f} GiB reserved={reserved:.2f} GiB")


class FluxGenerator:
    def __init__(self, args):
        self.device = torch.device(args.device)

        precision = get_precision()
        print(precision)
        print(f"Auto-detected precision: {precision}")
        print_cuda_memory("before_load")

        model_id = "black-forest-labs/FLUX.1-krea-dev" if args.name == "flux-krea-dev" else "black-forest-labs/FLUX.1-dev"
        weights_id = "nunchaku-flux.1-krea-dev" if args.name == "flux-krea-dev" else "nunchaku-flux.1-dev"
        
        transformer = NunchakuFluxTransformer2dModel.from_pretrained(
            f"nunchaku-tech/{weights_id}/svdq-{precision}_r32-{weights_id.replace('nunchaku-', '')}.safetensors"
        )
        print_cuda_memory("after_transformer")

        # T5-XXL (text_encoder_2) quantization mode: bf16 | nf4 | int8
        t5_quant = getattr(args, "t5_quant", "bf16")
        text_encoder_2 = None
        if t5_quant in ("nf4", "int8"):
            if t5_quant == "nf4":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                )
            else:
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            text_encoder_2 = T5EncoderModel.from_pretrained(
                model_id,
                subfolder="text_encoder_2",
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
            )
            print_cuda_memory(f"after_text_encoder_2_{t5_quant}")

        pipe_kwargs = dict(transformer=transformer, torch_dtype=torch.bfloat16)
        if text_encoder_2 is not None:
            pipe_kwargs["text_encoder_2"] = text_encoder_2

        self.pipeline = PuLIDFluxPipeline.from_pretrained(model_id, **pipe_kwargs)
        if text_encoder_2 is None:
            # No quantized modules — safe to move whole pipeline
            self.pipeline.to("cuda")
        else:
            # text_encoder_2 already placed by bnb; move the rest individually
            self.pipeline.transformer.to("cuda")
            self.pipeline.text_encoder.to("cuda")
            self.pipeline.vae.to("cuda")

        # Monkey-patch the forward to support PuLID injection
        self.pipeline.transformer.forward = MethodType(pulid_forward, self.pipeline.transformer)
        print_cuda_memory("after_pipeline")

        # LoRA state
        self.current_lora_path = None
        if getattr(args, "lora_path", None) and os.path.exists(args.lora_path):
            self.load_lora(args.lora_path, getattr(args, "lora_scale", 1.0))

    def load_lora(self, lora_path, lora_scale=1.0):
        if lora_path and os.path.exists(lora_path):
            print(f"Loading LoRA from {lora_path} (scale={lora_scale})...")
            self.pipeline.transformer.update_lora_params(lora_path)
            self.pipeline.transformer.set_lora_strength(lora_scale)
            self.current_lora_path = lora_path
            print("LoRA loaded successfully.")
            print_cuda_memory("after_lora")
        elif self.current_lora_path is not None:
            self.pipeline.transformer.set_lora_strength(0.0)
            self.current_lora_path = None

    def set_lora_strength(self, lora_scale):
        if self.current_lora_path is not None:
            self.pipeline.transformer.set_lora_strength(lora_scale)

    @torch.no_grad()
    def generate_image(
        self,
        prompt,
        id_image=None,
        width=768,
        height=1152,
        num_steps=20,
        start_step=0,
        guidance=4.0,
        seed=-1,
        id_weight=1.0,
        neg_prompt="",
        true_cfg=1.0,
        max_sequence_length=128,
    ):
        seed = int(seed)
        generator = None
        if seed != -1:
            generator = torch.Generator(device="cpu").manual_seed(seed)
        else:
            seed = torch.Generator(device="cpu").seed()
            generator = torch.Generator(device="cpu").manual_seed(seed)

        print(f"Generating '{prompt}' with seed {seed}, start_step {start_step}, id_weight {id_weight}")
        t0 = time.perf_counter()

        use_true_cfg = abs(true_cfg - 1.0) > 1e-2

        result = self.pipeline(
            prompt=prompt,
            negative_prompt=neg_prompt if use_true_cfg else None,
            true_cfg_scale=true_cfg if use_true_cfg else 1.0,
            id_image=id_image,
            id_weight=id_weight,
            start_step=start_step,
            width=width,
            height=height,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            generator=generator,
            max_sequence_length=max_sequence_length,
        )

        t1 = time.perf_counter()
        print(f"Done in {t1 - t0:.1f}s.")

        return result.images[0], str(seed)


_HEADER_ = '''
<div style="text-align: center; max-width: 650px; margin: 0 auto;">
    <h1 style="font-size: 2.5rem; font-weight: 700; margin-bottom: 1rem; display: contents;">PuLID for FLUX (Nunchaku SVDQuant)</h1>
    <p style="font-size: 1rem; margin-bottom: 1.5rem;">
        Uses <a href='https://github.com/nunchaku-ai/nunchaku' target='_blank'>Nunchaku</a> 4-bit quantized FLUX for lower VRAM usage.
        Paper: <a href='https://arxiv.org/abs/2404.16022' target='_blank'>PuLID</a>
    </p>
</div>

**Tips:**
- `timestep to start inserting ID:` Lower = higher fidelity but lower editability. Recommended 0-4. Photorealistic: 4, stylized: 0-1.
- `true CFG scale:` 1 = fake CFG (recommended for most cases). >1 = true CFG (better in some cases).
'''


def create_demo(args):
    generator = FluxGenerator(args)

    def generate_batch(
        width, height, num_steps, start_step, guidance, seed,
        prompts_text, id_images, id_weight, neg_prompt,
        true_cfg, max_sequence_length, per_image_params,
        lora_file, lora_scale, lora_enabled,
    ):
        # Handle LoRA
        if lora_enabled and lora_file is not None:
            lora_path = lora_file.name if hasattr(lora_file, "name") else lora_file
            if lora_path != generator.current_lora_path:
                generator.load_lora(lora_path, lora_scale)
            else:
                generator.set_lora_strength(lora_scale)
        elif lora_enabled and generator.current_lora_path is not None:
            generator.set_lora_strength(lora_scale)
        elif not lora_enabled and generator.current_lora_path is not None:
            generator.set_lora_strength(0.0)

        prompts = [p.strip() for p in prompts_text.split('\n') if p.strip()]
        if not prompts:
            return [], "No prompts provided", None

        if not id_images:
            images_pil = [None]
        else:
            images_pil = []
            for item in id_images:
                img_data = item[0] if isinstance(item, (tuple, list)) else item
                if isinstance(img_data, str):
                    images_pil.append(Image.open(img_data).convert("RGB"))
                elif isinstance(img_data, np.ndarray):
                    images_pil.append(Image.fromarray(img_data).convert("RGB"))
                else:
                    images_pil.append(img_data)

        def get_img_params(idx):
            rows = [line.strip() for line in (per_image_params or "").splitlines() if line.strip()]
            if idx < len(rows):
                parts = rows[idx].replace(',', ' ').split()
                try:
                    iw = float(parts[0]) if len(parts) > 0 else id_weight
                except Exception:
                    iw = id_weight
                try:
                    ss = int(float(parts[1])) if len(parts) > 1 else start_step
                except Exception:
                    ss = start_step
                return iw, ss
            return id_weight, start_step

        all_images = []
        all_seeds = []

        for img_idx, id_img in enumerate(images_pil):
            for prompt_idx, prompt in enumerate(prompts):
                prompt_id_weight, prompt_start_step = get_img_params(prompt_idx)

                result_img, used_seed = generator.generate_image(
                    prompt=prompt,
                    id_image=id_img,
                    width=width,
                    height=height,
                    num_steps=num_steps,
                    start_step=prompt_start_step,
                    guidance=guidance,
                    seed=seed,
                    id_weight=prompt_id_weight,
                    neg_prompt=neg_prompt,
                    true_cfg=true_cfg,
                    max_sequence_length=max_sequence_length,
                )
                all_images.append(result_img)
                all_seeds.append(used_seed)

        # Save images and create zip
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("outputs") / timestamp
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(all_images):
            img_idx = i // len(prompts)
            prompt_idx = i % len(prompts)
            img.save(out_dir / f"img{img_idx + 1:02d}_p{prompt_idx + 1:02d}_seed{all_seeds[i]}.png")
        zip_path = str(out_dir) + ".zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(out_dir.glob("*.png")):
                zf.write(f, f.name)

        return all_images, ", ".join(all_seeds), zip_path

    with gr.Blocks() as demo:
        gr.Markdown(_HEADER_)

        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(
                    label="Prompts (one per line)",
                    lines=5,
                    value="portrait, color, cinematic",
                )
                id_images = gr.Gallery(
                    label="ID Images (upload one or more; every image runs with every prompt)",
                    type="filepath",
                    columns=3,
                    height="auto",
                    interactive=True,
                )
                per_image_params = gr.Textbox(
                    label="Per-image parameters (optional) — one row per prompt: id_weight start_step",
                    placeholder="1.0 0\n0.8 4\n1.2 2",
                    lines=4,
                )
                id_weight = gr.Number(value=1.0, label="id weight (global default)", precision=2)

                width = gr.Slider(256, 1536, 768, step=16, label="Width")
                height = gr.Slider(256, 1536, 1152, step=16, label="Height")
                num_steps = gr.Slider(1, 30, 20, step=1, label="Number of steps")
                start_step = gr.Number(value=0, label="timestep to start inserting ID (global default)", precision=0)
                guidance = gr.Slider(1.0, 10.0, 4, step=0.1, label="Guidance")
                seed = gr.Textbox(-1, label="Seed (-1 for random)")
                max_sequence_length = gr.Slider(128, 512, 128, step=128,
                                                label="max_sequence_length for prompt (T5)")

                with gr.Accordion("LoRA Settings", open=False):
                    lora_enabled = gr.Checkbox(value=False, label="Enable LoRA")
                    lora_file = gr.File(
                        label="LoRA file (.safetensors)",
                        file_types=[".safetensors"],
                    )
                    lora_scale = gr.Slider(0.0, 2.0, 1.0, step=0.05, label="LoRA strength")

                with gr.Accordion("Advanced Options (True CFG)", open=False):
                    neg_prompt = gr.Textbox(
                        label="Negative Prompt",
                        value="bad quality, worst quality, text, signature, watermark, extra limbs",
                    )
                    true_cfg = gr.Slider(1.0, 10.0, 1, step=0.1, label="true CFG scale (1 = fake CFG)")

                generate_btn = gr.Button("Generate", variant="primary")

            with gr.Column():
                output_gallery = gr.Gallery(label="Generated Images", columns=2, height="auto")
                seed_output = gr.Textbox(label="Used Seeds")
                download_btn = gr.File(label="Download all images (zip)", interactive=False)

        generate_btn.click(
            fn=generate_batch,
            inputs=[
                width, height, num_steps, start_step, guidance, seed,
                prompt, id_images, id_weight, neg_prompt,
                true_cfg, max_sequence_length, per_image_params,
                lora_file, lora_scale, lora_enabled,
            ],
            outputs=[output_gallery, seed_output, download_btn],
        )

    return demo


if __name__ == "__main__":
    import argparse
    import os

    os.environ['GRADIO_TEMP_DIR'] = os.path.join(os.getcwd(), 'gradio_tmp')
    os.makedirs(os.environ['GRADIO_TEMP_DIR'], exist_ok=True)

    parser = argparse.ArgumentParser(description="PuLID for FLUX (Nunchaku SVDQuant)")
    parser.add_argument("--name", type=str, default="flux-krea-dev",
                        choices=["flux-dev", "flux-krea-dev"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA .safetensors file")
    parser.add_argument("--lora_scale", type=float, default=1.0, help="LoRA strength (0.0-1.0)")
    parser.add_argument("--t5_quant", type=str, default="bf16",
                        choices=["bf16", "nf4", "int8"],
                        help="T5-XXL (text_encoder_2) quantization: bf16 (~9.5GB), int8 (~5GB), nf4 (~3GB)")
    args = parser.parse_args()

    demo = create_demo(args)
    demo.launch(server_name="0.0.0.0", server_port=args.port)

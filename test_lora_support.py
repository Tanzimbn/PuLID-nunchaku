import os
import argparse
import time
from datetime import datetime
from pathlib import Path
from types import MethodType

import torch
from PIL import Image

from nunchaku.models.pulid.pulid_forward import pulid_forward
from nunchaku.models.transformers.transformer_flux import NunchakuFluxTransformer2dModel
from nunchaku.pipeline.pipeline_flux_pulid import PuLIDFluxPipeline
from nunchaku.utils import get_precision

SHARED_CACHE = "/home/shared/.cache/huggingface"
LOCAL_EVA_CACHE = "/home/tanzim/.cache/huggingface"
os.environ["HF_HOME"] = SHARED_CACHE


def generate_with_lora(
    lora_path,
    lora_scale=1.0,
    model_name="flux-krea-dev",
    prompt="portrait, color, cinematic",
    id_image_path=None,
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    precision = get_precision()
    print(f"Auto-detected precision: {precision}")

    model_id = "black-forest-labs/FLUX.1-krea-dev" if model_name == "flux-krea-dev" else "black-forest-labs/FLUX.1-dev"
    weights_id = "nunchaku-flux.1-krea-dev" if model_name == "flux-krea-dev" else "nunchaku-flux.1-dev"

    # Load transformer
    print(f"Loading transformer: {weights_id}...")
    transformer = NunchakuFluxTransformer2dModel.from_pretrained(
        f"nunchaku-tech/{weights_id}/svdq-{precision}_r32-{weights_id.replace('nunchaku-', '')}.safetensors"
    )

    # Load pipeline
    print(f"Loading pipeline: {model_id}...")
    pipeline = PuLIDFluxPipeline.from_pretrained(
        model_id,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    ).to(device)

    # Monkey-patch forward for PuLID identity injection
    pipeline.transformer.forward = MethodType(pulid_forward, pipeline.transformer)

    # Load LoRA
    print(f"Loading LoRA from {lora_path} (scale={lora_scale})...")
    transformer.update_lora_params(lora_path)
    transformer.set_lora_strength(lora_scale)
    print("LoRA loaded successfully.")

    # Load ID image if provided
    id_image = None
    if id_image_path and os.path.exists(id_image_path):
        id_image = Image.open(id_image_path).convert("RGB")
        print(f"Loaded ID image: {id_image_path}")

    # Setup seed
    seed = int(seed)
    if seed == -1:
        seed = torch.Generator(device="cpu").seed()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    print(f"Using seed: {seed}")

    # Generate image
    use_true_cfg = abs(true_cfg - 1.0) > 1e-2
    print(f"Generating: '{prompt}'")
    t0 = time.perf_counter()

    with torch.no_grad():
        result = pipeline(
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
    print(f"Generation done in {t1 - t0:.1f}s")

    # Save output
    out_dir = Path("outputs") / f"lora_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"seed{seed}.png"
    result.images[0].save(out_path)
    print(f"Saved to {out_path}")

    return result.images[0], seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test LoRA image generation for PuLID-FLUX")
    parser.add_argument("--lora_path", type=str, default="/home/tanzim/filter/reaging/PuLID-nunchaku/strategy-neon-krea.safetensors")
    parser.add_argument("--lora_scale", type=float, default=1.0)
    parser.add_argument("--name", type=str, default="flux-krea-dev", choices=["flux-dev", "flux-krea-dev"])
    parser.add_argument("--prompt", type=str, default="portrait, color, cinematic")
    parser.add_argument("--id_image", type=str, default="/home/tanzim/filter/reaging/PuLID-nunchaku/input/office_8.png", help="Path to ID reference image")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=1152)
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--id_weight", type=float, default=1.0)
    parser.add_argument("--true_cfg", type=float, default=1.0)

    args = parser.parse_args()

    if not os.path.exists(args.lora_path):
        print(f"Error: LoRA file not found at {args.lora_path}")
    else:
        generate_with_lora(
            lora_path=args.lora_path,
            lora_scale=args.lora_scale,
            model_name=args.name,
            prompt=args.prompt,
            id_image_path=args.id_image,
            width=args.width,
            height=args.height,
            num_steps=args.num_steps,
            start_step=args.start_step,
            guidance=args.guidance,
            seed=args.seed,
            id_weight=args.id_weight,
            true_cfg=args.true_cfg,
        )

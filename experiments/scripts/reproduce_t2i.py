#!/usr/bin/env python3
"""
Script to reproduce MMaDA text-to-image generation results.
"""

import sys
import os
import torch
from pathlib import Path
from PIL import Image
import numpy as np

# Add MMaDA to path
mmada_path = Path(__file__).parent.parent.parent / "MMaDA"
sys.path.insert(0, str(mmada_path))

from transformers import AutoTokenizer
from models import MMadaModelLM, MAGVITv2, get_mask_schedule
from training.prompting_utils import UniversalPrompting
from training.utils import image_transform, get_config


def generate_image(model, vq_model, uni_prompting, device, prompt, config, guidance_scale=3.5, timesteps=15):
    """Generate an image from a text prompt."""
    print(f"\nGenerating image for prompt: {prompt}")
    
    # Prepare image tokens (all masked initially)
    mask_token_id = model.config.mask_token_id
    image_tokens = torch.ones((1, config.model.mmada.num_vq_tokens),
                              dtype=torch.long, device=device) * mask_token_id
    
    # Use uni_prompting to construct the input
    input_ids, attention_mask = uni_prompting(([prompt], image_tokens), 't2i_gen')
    
    # Prepare unconditional input for classifier-free guidance
    if guidance_scale > 0:
        uncond_input_ids, uncond_attention_mask = uni_prompting(([''], image_tokens), 't2i_gen')
    else:
        uncond_input_ids = None
        uncond_attention_mask = None
    
    # Get mask schedule
    from models import get_mask_schedule
    mask_schedule = get_mask_schedule("cosine")
    
    # Generate image tokens
    with torch.no_grad():
        if device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                gen_token_ids = model.t2i_generate(
                    input_ids=input_ids,
                    uncond_input_ids=uncond_input_ids,
                    attention_mask=attention_mask,
                    uncond_attention_mask=uncond_attention_mask,
                    guidance_scale=guidance_scale,
                    temperature=1.0,
                    timesteps=timesteps,
                    noise_schedule=mask_schedule,
                    noise_type="mask",
                    seq_len=config.model.mmada.num_vq_tokens,
                    uni_prompting=uni_prompting,
                    config=config,
                )
        else:
            gen_token_ids = model.t2i_generate(
                input_ids=input_ids,
                uncond_input_ids=uncond_input_ids,
                attention_mask=attention_mask,
                uncond_attention_mask=uncond_attention_mask,
                guidance_scale=guidance_scale,
                temperature=1.0,
                timesteps=timesteps,
                noise_schedule=mask_schedule,
                noise_type="mask",
                seq_len=config.model.mmada.num_vq_tokens,
                uni_prompting=uni_prompting,
                config=config,
            )
    
    # Clamp token IDs to valid range
    gen_token_ids = torch.clamp(gen_token_ids, max=config.model.mmada.codebook_size - 1, min=0)
    
    # Decode image tokens to pixels
    images = vq_model.decode_code(gen_token_ids)
    
    # Post-process image
    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
    images *= 255.0
    images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    image = Image.fromarray(images[0])
    
    return image


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "Gen-Verse/MMaDA-8B-MixCoT"
    
    print(f"Loading model: {model_name}")
    print(f"Device: {device}")
    
    # Load config (use demo config as base)
    try:
        config = get_config()
        # Override with demo config if available
        config_path = mmada_path / "configs" / "mmada_demo.yaml"
        if config_path.exists():
            import omegaconf
            demo_config = omegaconf.OmegaConf.load(config_path)
            config = omegaconf.OmegaConf.merge(config, demo_config)
    except:
        # Create minimal config if get_config fails
        from omegaconf import OmegaConf
        config = OmegaConf.create({
            "model": {
                "mmada": {
                    "pretrained_model_path": model_name,
                    "num_vq_tokens": 1024,
                    "codebook_size": 8192,
                },
                "vq_model": {
                    "type": "magvitv2",
                    "vq_model_name": "showlab/magvitv2"
                }
            },
            "dataset": {
                "preprocessing": {
                    "max_seq_length": 512
                }
            },
            "training": {
                "cond_dropout_prob": 0.1
            }
        })
    
    # Load tokenizer and setup prompting
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    uni_prompting = UniversalPrompting(
        tokenizer,
        max_text_len=config.dataset.preprocessing.max_seq_length,
        special_tokens=(
            "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>",
            "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"
        ),
        ignore_id=-100,
        cond_dropout_prob=config.training.cond_dropout_prob,
        use_reserved_token=True
    )
    
    # Load VQ model
    print("Loading MAGVIT-v2 tokenizer...")
    vq_model = MAGVITv2.from_pretrained(config.model.vq_model.vq_model_name).to(device)
    vq_model.eval()
    vq_model.requires_grad_(False)
    
    # Load MMaDA model
    print("Loading MMaDA model...")
    model = MMadaModelLM.from_pretrained(
        config.model.mmada.pretrained_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    
    # Test prompts
    test_prompts = [
        "A serene mountain landscape at sunset with snow-capped peaks",
        "A futuristic city with flying cars and neon lights",
        "A cat wearing sunglasses sitting on a beach",
        "An abstract painting with vibrant colors",
    ]
    
    # Create output directory
    output_dir = Path(__file__).parent / "generated_images"
    output_dir.mkdir(exist_ok=True)
    
    # Generate images
    for i, prompt in enumerate(test_prompts):
        try:
            print(f"\n{'='*60}")
            print(f"Generating image {i+1}/{len(test_prompts)}")
            image = generate_image(
                model, vq_model, uni_prompting, device, prompt, config,
                guidance_scale=3.5, timesteps=15
            )
            
            # Save image
            output_path = output_dir / f"generated_{i+1}.png"
            image.save(output_path)
            print(f"Saved to {output_path}")
        except Exception as e:
            print(f"Error generating image: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("Text-to-Image Generation Tests Complete!")
    print(f"Images saved to {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()


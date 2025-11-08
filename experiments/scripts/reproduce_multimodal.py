#!/usr/bin/env python3
"""
Script to reproduce MMaDA multimodal understanding results.
"""

import sys
import os
import json
import torch
from pathlib import Path
from PIL import Image

# Add MMaDA to path
mmada_path = Path(__file__).parent.parent.parent / "MMaDA"
sys.path.insert(0, str(mmada_path))

from transformers import AutoTokenizer
from models import MMadaModelLM, MAGVITv2
from training.prompting_utils import UniversalPrompting
from training.utils import image_transform, image_transform_squash


def test_multimodal_understanding(model, vq_model, uni_prompting, device, image_path, question):
    """Test multimodal understanding on a single image-question pair."""
    print(f"\nImage: {image_path}")
    print(f"Question: {question}")
    
    # Load and preprocess image
    try:
        image_ori = Image.open(image_path).convert("RGB")
        # Use appropriate transform based on image type
        if any(tag in image_path for tag in ['ai2d', 'clevr', 'docvqa', 'geo', 'llava']):
            image = image_transform_squash(image_ori, resolution=512).to(device)
        else:
            image = image_transform(image_ori, resolution=512).to(device)
        image = image.unsqueeze(0)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
    
    # Encode image to tokens
    image_tokens = vq_model.get_code(image) + len(uni_prompting.text_tokenizer)
    
    # Prepare text prompt
    messages = [{"role": "user", "content": question}]
    text_token_ids = uni_prompting.text_tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)
    
    # Construct full input
    batch_size = image_tokens.shape[0]
    input_ids = torch.cat([
        (torch.ones(batch_size, 1) * uni_prompting.sptids_dict['<|mmu|>']).to(device),
        (torch.ones(batch_size, 1) * uni_prompting.sptids_dict['<|soi|>']).to(device),
        image_tokens,
        (torch.ones(batch_size, 1) * uni_prompting.sptids_dict['<|eoi|>']).to(device),
        (torch.ones(batch_size, 1) * uni_prompting.sptids_dict['<|sot|>']).to(device),
        text_token_ids
    ], dim=1).long()
    
    # Generate response
    with torch.no_grad():
        if device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_ids = model.mmu_generate(
                    input_ids,
                    max_new_tokens=512,
                    steps=256,
                    block_length=128,
                )
        else:
            output_ids = model.mmu_generate(
                input_ids,
                max_new_tokens=512,
                steps=256,
                block_length=128,
            )
    
    generated_ids = output_ids[:, input_ids.shape[1]:]
    response_text = uni_prompting.text_tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    print(f"Response: {response_text}")
    return response_text


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "Gen-Verse/MMaDA-8B-MixCoT"
    
    print(f"Loading model: {model_name}")
    print(f"Device: {device}")
    
    # Load tokenizer and setup prompting
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    uni_prompting = UniversalPrompting(
        tokenizer,
        max_text_len=512,
        special_tokens=(
            "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>",
            "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"
        ),
        ignore_id=-100,
        cond_dropout_prob=0.1,
        use_reserved_token=True
    )
    
    # Load VQ model
    print("Loading MAGVIT-v2 tokenizer...")
    vq_model = MAGVITv2.from_pretrained("showlab/magvitv2").to(device)
    vq_model.eval()
    vq_model.requires_grad_(False)
    
    # Load MMaDA model
    print("Loading MMaDA model...")
    model = MMadaModelLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    
    # Test on validation images
    mmu_validation_dir = mmada_path / "mmu_validation"
    prompts_file = mmu_validation_dir / "prompts_with_vqa.json"
    
    if prompts_file.exists():
        print(f"\nLoading prompts from {prompts_file}")
        with open(prompts_file, "r", encoding="utf-8") as f:
            validation_data = json.load(f)
        
        results = []
        for item in validation_data[:5]:  # Test first 5 for quick demo
            file_name = item["file_name"]
            messages = item["messages"]
            image_path = mmu_validation_dir / file_name
            
            if not image_path.exists():
                print(f"Warning: Image not found: {image_path}")
                continue
            
            question = messages[0]["content"] if messages else "What is in this image?"
            response = test_multimodal_understanding(
                model, vq_model, uni_prompting, device, str(image_path), question
            )
            
            if response:
                results.append({
                    "file_name": file_name,
                    "question": question,
                    "response": response
                })
        
        # Save results
        output_file = Path(__file__).parent / "mmu_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_file}")
    else:
        print(f"Prompts file not found: {prompts_file}")
        print("Testing with a single example...")
        
        # Try to find any image in the directory
        image_files = list(mmu_validation_dir.glob("*.jpg")) + list(mmu_validation_dir.glob("*.png"))
        if image_files:
            test_image = image_files[0]
            test_multimodal_understanding(
                model, vq_model, uni_prompting, device,
                str(test_image), "What is in this image? Describe it in detail."
            )
        else:
            print("No test images found. Please ensure mmu_validation directory contains images.")
    
    print("\n" + "="*60)
    print("Multimodal Understanding Tests Complete!")
    print("="*60)


if __name__ == '__main__':
    main()


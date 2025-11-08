#!/usr/bin/env python3
"""
Script to reproduce MMaDA text generation results.
This script tests text generation capabilities on various prompts.
"""

import sys
import os
import torch
from pathlib import Path

# Add MMaDA to path
mmada_path = Path(__file__).parent.parent.parent / "MMaDA"
sys.path.insert(0, str(mmada_path))

from transformers import AutoTokenizer
from models import MMadaModelLM
from generate import generate


def test_math_reasoning(model, tokenizer, device):
    """Test mathematical reasoning capabilities."""
    print("\n" + "="*60)
    print("Testing Mathematical Reasoning")
    print("="*60)
    
    prompts = [
        "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?",
        "If a train travels 300 km in 2 hours, what is its average speed?",
        "A store sells apples for $2 per pound. If you buy 5 pounds and get a 10% discount, how much do you pay?",
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        messages = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        input_ids = tokenizer(
            prompt_text, return_tensors="pt", padding=True
        )['input_ids'].to(device)
        
        output = generate(
            model, input_ids,
            steps=128, gen_length=128, block_length=128,
            temperature=1.0, cfg_scale=0.0, remasking='low_confidence'
        )
        
        generated_text = tokenizer.batch_decode(
            output[:, input_ids.shape[1]:], skip_special_tokens=True
        )[0]
        print(f"Generated: {generated_text}")


def test_general_knowledge(model, tokenizer, device):
    """Test general knowledge and reasoning."""
    print("\n" + "="*60)
    print("Testing General Knowledge")
    print("="*60)
    
    prompts = [
        "Explain the process of photosynthesis in simple terms.",
        "What are the main differences between renewable and non-renewable energy sources?",
        "Describe the water cycle.",
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        messages = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        input_ids = tokenizer(
            prompt_text, return_tensors="pt", padding=True
        )['input_ids'].to(device)
        
        output = generate(
            model, input_ids,
            steps=128, gen_length=256, block_length=128,
            temperature=1.0, cfg_scale=0.0, remasking='low_confidence'
        )
        
        generated_text = tokenizer.batch_decode(
            output[:, input_ids.shape[1]:], skip_special_tokens=True
        )[0]
        print(f"Generated: {generated_text[:500]}...")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = "Gen-Verse/MMaDA-8B-MixCoT"
    
    print(f"Loading model: {model_name}")
    print(f"Device: {device}")
    
    # Load model
    model = MMadaModelLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(device).eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.chat_template = (
        "{% set loop_messages = messages %}"
        "{% for message in loop_messages %}"
        "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n'"
        "+ message['content'] | trim + '<|eot_id|>' %}"
        "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
        "{{ content }}"
        "{% endfor %}"
        "{{ '<|start_header_id|>assistant<|end_header_id|>\n' }}"
    )
    
    # Run tests
    test_math_reasoning(model, tokenizer, device)
    test_general_knowledge(model, tokenizer, device)
    
    print("\n" + "="*60)
    print("Text Generation Tests Complete!")
    print("="*60)


if __name__ == '__main__':
    main()


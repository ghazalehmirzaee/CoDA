# MMaDA Results Reproduction Guide

This guide will help you reproduce MMaDA results as a foundation for CoDA development.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Model Checkpoints](#model-checkpoints)
4. [Reproducing Text Generation Results](#reproducing-text-generation-results)
5. [Reproducing Multimodal Understanding Results](#reproducing-multimodal-understanding-results)
6. [Reproducing Text-to-Image Generation Results](#reproducing-text-to-image-generation-results)
7. [Evaluation Benchmarks](#evaluation-benchmarks)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with at least 16GB VRAM (24GB+ recommended for inference)
- **RAM**: 32GB+ system RAM recommended
- **Storage**: ~50GB free space for models and datasets

### Software Requirements
- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- PyTorch 2.0+
- Git

## Environment Setup

### 1. Activate Your Virtual Environment

```bash
# Activate your existing virtual environment (e.g., 'coda')
source coda/bin/activate
# Or wherever your virtual environment is located
```

### 2. Navigate to MMaDA Directory and Install Dependencies

```bash
cd /home/ghazal/Documents/CS_Projects/CoDA/MMaDA

# Install requirements
pip install -r requirements.txt

# Install MMaDA package in development mode
pip install -e .
```

**Note**: Make sure your virtual environment is activated before installing dependencies.

### 3. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

## Model Checkpoints

MMaDA provides three checkpoints:

1. **MMaDA-8B-Base**: Basic capabilities after pretraining and instruction tuning
   - HuggingFace: `Gen-Verse/MMaDA-8B-Base`
   - Use for: Basic text generation, image generation, image captioning

2. **MMaDA-8B-MixCoT**: Enhanced reasoning after Chain-of-Thought fine-tuning
   - HuggingFace: `Gen-Verse/MMaDA-8B-MixCoT`
   - Use for: Complex reasoning tasks, multimodal understanding
   - **Recommended for most evaluations**

3. **MMaDA-8B-Max**: (Coming soon) After UniGRPO reinforcement learning

Models will be automatically downloaded from HuggingFace when first used.

### Additional Required Models

- **MAGVIT-v2 Image Tokenizer**: `showlab/magvitv2`
  - Automatically downloaded when running inference scripts

## Reproducing Text Generation Results

### Quick Test

```bash
cd /home/ghazal/Documents/CS_Projects/CoDA/MMaDA
python generate.py
```

This runs a simple math problem example using MMaDA-8B-MixCoT.

### Custom Text Generation

Create a script `experiments/scripts/test_text_generation.py`:

```python
import torch
from transformers import AutoTokenizer
from MMaDA.models import MMadaModelLM
from MMaDA.generate import generate

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = "Gen-Verse/MMaDA-8B-MixCoT"

# Load model
model = MMadaModelLM.from_pretrained(
    model_name, 
    trust_remote_code=True, 
    torch_dtype=torch.bfloat16
).to(device).eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\n' }}"

# Prepare prompt
prompt = "Solve: If a train travels 300 km in 2 hours, what is its average speed?"
messages = [{"role": "user", "content": prompt}]
prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
input_ids = tokenizer(prompt_text, return_tensors="pt", padding=True)['input_ids'].to(device)

# Generate
output = generate(
    model, 
    input_ids, 
    steps=128, 
    gen_length=128, 
    block_length=128, 
    temperature=1.0, 
    cfg_scale=0.0, 
    remasking='low_confidence'
)

# Decode
generated_text = tokenizer.batch_decode(
    output[:, input_ids.shape[1]:], 
    skip_special_tokens=True
)[0]
print(f"Generated: {generated_text}")
```

### Evaluation on Language Benchmarks

For comprehensive language model evaluation:

```bash
cd MMaDA/evaluation/lm
pip install lm-eval
bash eval.sh
```

This evaluates on standard benchmarks like GSM8K, MMLU, etc.

## Reproducing Multimodal Understanding Results

### Setup WandB (Required)

```bash
wandb login
# Follow prompts to enter your API key
```

### Run Multimodal Understanding Inference

```bash
cd /home/ghazal/Documents/CS_Projects/CoDA/MMaDA

python3 inference_mmu.py \
  config=configs/mmada_demo.yaml \
  mmu_image_root=./mmu_validation \
  mmu_prompts_file=./mmu_validation/prompts_with_vqa.json
```

This will:
- Load images from `mmu_validation/`
- Process prompts from `prompts_with_vqa.json`
- Generate responses and log to WandB
- Save results to `inference_results.json`

### Custom Multimodal Understanding

You can modify `mmu_validation/prompts_with_vqa.json` to test on your own images:

```json
[
  {
    "file_name": "your_image.jpg",
    "messages": [
      {
        "role": "user",
        "content": "What is in this image?"
      }
    ]
  }
]
```

### Evaluation on VLM Benchmarks

For comprehensive vision-language evaluation:

```bash
cd MMaDA/evaluation/VLMEvalKit
pip install -e .

# Configure model in vlmeval/config.py (see eval.md)
# Then run evaluation
CUDA_VISIBLE_DEVICES=0 python run.py --data {dataset_name} --model MMaDA-MixCoT
```

Supported datasets include: MathVista, MathVerse, AI2D, DocVQA, etc.

## Reproducing Text-to-Image Generation Results

### Run Text-to-Image Inference

```bash
cd /home/ghazal/Documents/CS_Projects/CoDA/MMaDA

python3 inference_t2i.py \
  config=configs/mmada_demo.yaml \
  batch_size=1 \
  validation_prompts_file=validation_prompts/text2image_prompts.txt \
  guidance_scale=3.5 \
  generation_timesteps=15 \
  mode='t2i'
```

### Custom Text-to-Image Prompts

Edit `validation_prompts/text2image_prompts.txt` or create your own:

```
A serene mountain landscape at sunset
A futuristic city with flying cars
A cat wearing sunglasses
```

### Evaluation on Image Generation Benchmarks

MMaDA uses [GenEval](https://github.com/djghosh13/geneval) for image generation evaluation. Refer to their documentation for setup and usage.

## Evaluation Benchmarks

### Language Model Benchmarks
- **GSM8K**: Math word problems
- **MMLU**: Multi-task language understanding
- **HellaSwag**: Commonsense reasoning
- **ARC**: Science questions

### Vision-Language Benchmarks
- **MathVista**: Visual math reasoning
- **MathVerse**: Mathematical visual understanding
- **AI2D**: Diagram understanding
- **DocVQA**: Document visual question answering
- **CLEVR**: Visual reasoning

### Image Generation Metrics
- **CLIP Score**: Text-image alignment
- **ImageReward**: Human preference score
- **FID**: Fréchet Inception Distance

## Expected Results

Based on the MMaDA paper, you should expect:

### Language Tasks (MMaDA-8B-MixCoT)
- GSM8K: ~75% accuracy
- MMLU: Competitive with LLaMA-3-7B

### Multimodal Understanding
- MathVista: Competitive with specialized VLM models
- DocVQA: Strong performance on document understanding

### Image Generation
- CLIP Score: Comparable to Stable Diffusion XL
- ImageReward: High human preference scores

**Note**: Exact numbers may vary based on hardware, random seeds, and model checkpoint versions.

## Troubleshooting

### Out of Memory Errors

1. **Reduce batch size**: Edit config files to use smaller batch sizes
2. **Use gradient checkpointing**: Already enabled in configs
3. **Use CPU offloading**: For inference, consider moving some operations to CPU
4. **Reduce sequence length**: Lower `max_seq_length` in configs

### Model Download Issues

If HuggingFace model download fails:
```bash
# Set HuggingFace cache directory
export HF_HOME=/path/to/cache
# Or use huggingface-cli login for authenticated access
huggingface-cli login
```

### WandB Issues

If WandB login fails:
```bash
# Use offline mode
export WANDB_MODE=offline
# Or disable WandB in configs
```

### CUDA/GPU Issues

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# Set specific GPU
export CUDA_VISIBLE_DEVICES=0
```

### Import Errors

If you get import errors:

**MMaDA Import Issues:**
MMaDA doesn't have a `setup.py` file. The reproduction scripts automatically add MMaDA to `sys.path`. If you need to import MMaDA modules manually:

```bash
# Add MMaDA to PYTHONPATH
export PYTHONPATH=/home/ghazal/Documents/CS_Projects/CoDA/MMaDA:$PYTHONPATH

# Or in Python:
import sys
sys.path.insert(0, '/home/ghazal/Documents/CS_Projects/CoDA/MMaDA')
```

**PyTorch Import Errors (iJIT_NotifyEvent):**
If you see `undefined symbol: iJIT_NotifyEvent` when importing torch:

```bash
bash experiments/scripts/quick_fix_pytorch.sh
```

## Next Steps for CoDA

Once you've successfully reproduced MMaDA results:

1. **Baseline Establishment**: Document your reproduction results as baseline
2. **Code Analysis**: Study MMaDA's architecture for CoDA integration
3. **Continual Learning Setup**: Begin implementing CoDA modules:
   - Generative Replay Module
   - Parameter Preservation Module
   - Modular Adaptation Module
   - Evaluation & Monitoring

## References

- MMaDA Paper: https://arxiv.org/abs/2505.15809
- MMaDA HuggingFace: https://huggingface.co/Gen-Verse/MMaDA-8B-MixCoT
- MMaDA GitHub: https://github.com/Gen-Verse/MMaDA


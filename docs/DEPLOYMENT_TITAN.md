# Deploying CoDA to Titan GPU Server

This guide explains how to deploy and run CoDA/MMaDA on the Titan GPU server with 8 NVIDIA RTX GPUs (24GB each).

## GPU Benefits for This Project

**Yes, GPUs significantly accelerate this work:**

1. **Model Loading**: MMaDA-8B has 8 billion parameters (~16GB in bfloat16). GPUs load this much faster than CPU.

2. **Inference Speed**: 
   - **Text Generation**: 10-100x faster on GPU vs CPU
   - **Image Generation**: Essential - CPU would take hours per image, GPU takes seconds
   - **Multimodal Understanding**: 20-50x faster on GPU

3. **Training** (for CoDA development):
   - Required for any model training/fine-tuning
   - Enables batch processing (multiple examples simultaneously)
   - Allows gradient accumulation for large effective batch sizes

4. **Memory**: 24GB per GPU allows:
   - Loading full 8B model
   - Batch processing for faster throughput
   - Running multiple experiments in parallel

## Prerequisites

1. **SSH Access**: You already have this (`ssh gmirzaee@titan.statler.wvu.edu`)
2. **Conda/Python Environment**: Set up on titan
3. **Git**: For syncing code
4. **Storage**: Ensure sufficient space (~50GB for models + data)

## Step 1: Transfer Project to Titan

### Option A: Using Git (Recommended)

```bash
# On your local machine, ensure code is committed
cd /home/ghazal/Documents/CS_Projects/CoDA
git add .
git commit -m "Initial CoDA setup"
git push origin main  # Push to GitHub/GitLab

# On titan server
ssh gmirzaee@titan.statler.wvu.edu
cd ~/projects  # or wherever you keep projects
git clone <your-repo-url> CoDA
cd CoDA
```

### Option B: Using rsync (Direct Transfer)

```bash
# From your local machine
cd /home/ghazal/Documents/CS_Projects
rsync -avz --exclude 'MMaDA/venv' --exclude '__pycache__' --exclude '*.pyc' \
  CoDA/ gmirzaee@titan.statler.wvu.edu:~/projects/CoDA/
```

### Option C: Using scp (Small Files)

```bash
# For individual files or small directories
scp -r CoDA gmirzaee@titan.statler.wvu.edu:~/projects/
```

## Step 2: Set Up Environment on Titan

```bash
# SSH into titan
ssh gmirzaee@titan.statler.wvu.edu

# Navigate to project
cd ~/projects/CoDA

# Create conda environment (if not exists)
conda create -n coda python=3.10 -y
conda activate coda

# Install CUDA toolkit (if not already installed)
conda install -c conda-forge cudatoolkit=11.8 -y

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install MMaDA requirements
cd MMaDA
pip install -r requirements.txt

# Verify GPU access
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
```

## Step 3: Configure GPU Usage

### Check Available GPUs

```bash
# Check GPU status
nvidia-smi

# Check which GPUs are available
python -c "import torch; print(f'Available GPUs: {torch.cuda.device_count()}')"
```

### Set GPU for Single Process

```bash
# Use specific GPU (e.g., GPU 0)
export CUDA_VISIBLE_DEVICES=0

# Or use multiple GPUs (for training)
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

### Multi-GPU Training (Future CoDA Development)

For training, you'll use `accelerate` or `torchrun`:

```bash
# Using accelerate (recommended)
accelerate config  # Configure once
accelerate launch training/train_mmada.py config=configs/mmada_demo.yaml

# Using torchrun (PyTorch native)
torchrun --nproc-per-node=4 training/train_mmada.py config=configs/mmada_demo.yaml
```

## Step 4: Run Reproduction Scripts on GPU

```bash
# Activate environment
conda activate coda
cd ~/projects/CoDA

# Set GPU (use GPU 0 for single process)
export CUDA_VISIBLE_DEVICES=0

# Run text generation
python experiments/scripts/reproduce_text_generation.py

# Run multimodal understanding (requires wandb login)
wandb login
python experiments/scripts/reproduce_multimodal.py

# Run text-to-image generation
python experiments/scripts/reproduce_t2i.py
```

## Step 5: Running in Background (Long Jobs)

For long-running experiments:

```bash
# Using nohup
nohup python experiments/scripts/reproduce_text_generation.py > output.log 2>&1 &

# Using screen
screen -S mmada_exp
python experiments/scripts/reproduce_text_generation.py
# Press Ctrl+A then D to detach
# Reattach: screen -r mmada_exp

# Using tmux
tmux new -s mmada_exp
python experiments/scripts/reproduce_text_generation.py
# Press Ctrl+B then D to detach
# Reattach: tmux attach -t mmada_exp
```

## Step 6: Monitor GPU Usage

```bash
# Watch GPU usage in real-time
watch -n 1 nvidia-smi

# Or check periodically
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
```

## Step 7: Sync Results Back to Local Machine

```bash
# From local machine
rsync -avz gmirzaee@titan.statler.wvu.edu:~/projects/CoDA/experiments/results/ \
  ~/Documents/CS_Projects/CoDA/experiments/results/
```

## Troubleshooting

### GPU Not Detected

```bash
# Check CUDA installation
nvcc --version

# Reinstall PyTorch with CUDA
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Out of Memory

```bash
# Use smaller batch sizes in configs
# Or use gradient checkpointing (already enabled)
# Or use multiple GPUs with model parallelism
```

### Permission Issues

```bash
# Check file permissions
chmod +x experiments/scripts/*.sh
chmod +x experiments/scripts/*.py
```

## Performance Expectations

With RTX GPUs (24GB):

- **Text Generation**: ~50-100 tokens/second
- **Image Generation**: ~5-10 seconds per 512x512 image
- **Multimodal Understanding**: ~2-5 seconds per image+question

## Next Steps for CoDA Development

When implementing CoDA continual learning:

1. Use multiple GPUs for parallel task training
2. Implement gradient checkpointing for memory efficiency
3. Use mixed precision (bfloat16) for faster training
4. Monitor GPU utilization to optimize batch sizes


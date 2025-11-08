# CoDA Project: Comprehensive Overview

**For Advisor Review and Project Documentation**

## Executive Summary

**CoDA (Continual Diffusion Adaptation)** extends MMaDA (Multimodal Large Diffusion Language Models) with continual learning capabilities, enabling the model to learn new tasks and modalities over time without catastrophic forgetting.

## 1. Background: What is MMaDA?

### 1.1 Core Innovation

MMaDA (Yang et al., NeurIPS 2025) is a unified multimodal diffusion model that treats both text and image generation as discrete token diffusion problems. Key innovations:

1. **Unified Architecture**: Single transformer handles both text and images
2. **Discrete Diffusion**: Both modalities use masked token prediction
3. **Chain-of-Thought Reasoning**: Generates reasoning traces before outputs
4. **Unified RL Training**: UniGRPO optimizes across all task types

### 1.2 Technical Details

**Architecture:**
- Base: LLaDA (diffusion-based language model)
- Image Tokenization: MAGVIT-v2 VQ-VAE (512×512 → 1024 tokens, 32×32 grid)
- Text Tokenization: Standard language model tokenizer
- Unified Vocabulary: Text tokens + Image tokens (126,464 + 8,192 = 134,656 total)

**Training Process:**
1. **Stage 1**: Pretraining on ImageNet + Image-Text pairs + Text instructions
2. **Stage 2**: Mixed Chain-of-Thought fine-tuning (text + multimodal + image reasoning)
3. **Stage 3**: UniGRPO reinforcement learning (diversified rewards)

**Capabilities:**
- Text generation (math, reasoning, general knowledge)
- Multimodal understanding (VQA, visual reasoning)
- Text-to-image generation (with reasoning)

### 1.3 Limitations

**Critical Limitation**: Once training completes, MMaDA is frozen. Adding new capabilities requires:
- Full model retraining (computationally expensive)
- Risk of catastrophic forgetting
- No mechanism for incremental learning

## 2. CoDA: Our Contribution

### 2.1 Research Question

**Can we enable MMaDA to learn new tasks/modalities incrementally without forgetting previous capabilities?**

### 2.2 Proposed Solution

CoDA adds four continual learning modules to MMaDA:

1. **Generative Replay Module**
   - Generates synthetic examples from previous tasks
   - Uses hierarchical prompt selection strategies
   - Prevents forgetting through rehearsal

2. **Parameter Preservation Module**
   - Estimates importance of attention heads and layers
   - Protects critical parameters during new task learning
   - Maintains cross-modal alignment

3. **Modular Adaptation Module**
   - Low-rank adapters (LoRA) for task-specific learning
   - Learned routing mechanisms
   - Enables task-specific transformations without modifying core model

4. **Evaluation & Monitoring**
   - Cross-modal coherence metrics
   - Catastrophic forgetting detection
   - Task-specific performance tracking

### 2.3 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CoDA Framework                        │
├─────────────────────────────────────────────────────────┤
│  Base: MMaDA Unified Diffusion Architecture             │
│  • Single transformer backbone                          │
│  • Discrete token diffusion (text + image)              │
│  • Mixed CoT reasoning                                  │
├─────────────────────────────────────────────────────────┤
│  CoDA Continual Learning Extensions:                    │
│                                                          │
│  1. Generative Replay Module                            │
│     └─ Hierarchical prompt selection & sampling         │
│                                                          │
│  2. Parameter Preservation Module                       │
│     ├─ Attention head importance estimation             │
│     ├─ Layer-wise criticality analysis                  │
│     └─ Cross-modal alignment protection                 │
│                                                          │
│  3. Modular Adaptation Module                           │
│     ├─ Low-rank adapter networks (LoRA)                 │
│     ├─ Task-specific transformations                    │
│     └─ Learned routing mechanisms                       │
│                                                          │
│  4. Evaluation & Monitoring                             │
│     ├─ Cross-modal coherence metrics                    │
│     ├─ Catastrophic forgetting detection                │
│     └─ Task-specific performance tracking               │
└─────────────────────────────────────────────────────────┘
```

## 3. Current Project Status

### 3.1 Completed

1. **Repository Setup**
   - Cloned MMaDA repository
   - Created CoDA project structure
   - Set up documentation framework

2. **Reproduction Infrastructure**
   - Created scripts to reproduce MMaDA baseline results
   - Set up environment configuration
   - Documented reproduction process

3. **Documentation**
   - Comprehensive reproduction guide
   - Deployment guide for GPU servers
   - Project structure documentation

### 3.2 In Progress

1. **Baseline Establishment**
   - Reproducing MMaDA results on text generation
   - Reproducing multimodal understanding results
   - Reproducing text-to-image generation results

2. **Code Analysis**
   - Understanding MMaDA architecture
   - Identifying integration points for CoDA modules
   - Planning implementation strategy

### 3.3 Next Steps

1. **Implement Base Wrapper** (`coda/models/base_mmada.py`)
   - Wrap MMaDA model for CoDA integration
   - Provide interface for continual learning

2. **Implement Generative Replay**
   - Prompt selection strategies
   - Synthetic example generation
   - Replay buffer management

3. **Implement Parameter Preservation**
   - Importance estimation algorithms
   - Parameter masking during training
   - Cross-modal alignment preservation

4. **Implement Modular Adaptation**
   - LoRA adapter integration
   - Task-specific routing
   - Adapter training loop

5. **Evaluation Framework**
   - Continual learning metrics
   - Forgetting measurement
   - Cross-modal coherence evaluation

## 4. Technical Challenges

### 4.1 Unified Architecture Challenge

**Problem**: MMaDA's unified architecture means parameters are shared across modalities. Modifying parameters for one task affects all modalities.

**Solution**: 
- Use LoRA adapters for task-specific modifications
- Preserve critical shared parameters
- Maintain cross-modal coherence through careful regularization

### 4.2 Discrete Diffusion Continual Learning

**Problem**: Continual learning for diffusion models is less explored than autoregressive models.

**Solution**:
- Adapt replay strategies for discrete token space
- Use importance sampling for diffusion timesteps
- Maintain masking schedule consistency across tasks

### 4.3 Cross-Modal Coherence

**Problem**: Learning new tasks must not break existing text-image alignment.

**Solution**:
- Monitor cross-modal coherence metrics
- Use alignment-preserving regularization
- Replay cross-modal examples during new task learning

## 5. Experimental Plan

### 5.1 Baseline Evaluation

**Tasks**:
1. Text generation (GSM8K, MMLU)
2. Multimodal understanding (MathVista, DocVQA)
3. Text-to-image generation (CLIP Score, ImageReward)

**Metrics**: Accuracy, quality scores, generation time

### 5.2 Continual Learning Experiments

**Task Sequences**:
1. **Sequence A**: Text → Multimodal → Image
2. **Sequence B**: Image → Text → Multimodal
3. **Sequence C**: Mixed interleaved tasks

**Metrics**:
- Forward transfer (new task performance)
- Backward transfer (old task performance after new learning)
- Forgetting rate
- Cross-modal coherence

### 5.3 Ablation Studies

- Effect of replay ratio
- Importance of parameter preservation
- LoRA rank sensitivity
- Prompt selection strategy comparison

## 6. Computational Resources

### 6.1 Current Setup

- **Local Machine**: Development and testing
- **Titan GPU Server**: 8× NVIDIA RTX GPUs (24GB each)

### 6.2 Resource Requirements

**Inference**:
- Single GPU sufficient
- ~16GB VRAM for 8B model
- Fast inference (seconds per example)

**Training**:
- Multiple GPUs recommended (4-8 GPUs)
- Gradient accumulation for large batches
- Mixed precision training (bfloat16)

**Storage**:
- Model checkpoints: ~16GB per checkpoint
- Datasets: ~50GB+ depending on tasks
- Results/logs: ~10GB per experiment run

## 7. Timeline and Milestones

### Phase 1: Baseline Reproduction (Current)
- **Week 1-2**: Reproduce MMaDA results
- **Deliverable**: Baseline performance metrics

### Phase 2: CoDA Implementation
- **Week 3-4**: Base wrapper and replay module
- **Week 5-6**: Parameter preservation module
- **Week 7-8**: Modular adaptation module

### Phase 3: Evaluation
- **Week 9-10**: Continual learning experiments
- **Week 11-12**: Ablation studies and analysis

### Phase 4: Writing
- **Week 13-14**: Paper writing and results documentation

## 8. Expected Contributions

1. **Methodological**: First continual learning framework for unified multimodal diffusion models

2. **Empirical**: Comprehensive evaluation of continual learning strategies for diffusion models

3. **Practical**: Enables incremental capability addition without full retraining

## 9. Related Work

- **MMaDA**: Base unified diffusion model
- **LLaDA**: Diffusion language model foundation
- **Continual Learning**: EWC, Replay, LoRA adaptations
- **Multimodal Learning**: Unified architectures, cross-modal alignment

## 10. Risk Mitigation

**Risk**: CoDA modules may degrade base MMaDA performance
**Mitigation**: Careful ablation studies, preserve base model capabilities

**Risk**: Continual learning may not work well for diffusion models
**Mitigation**: Extensive literature review, pilot experiments, alternative strategies

**Risk**: Computational resources insufficient
**Mitigation**: Optimize for efficiency, use gradient checkpointing, leverage Titan server

## 11. Success Criteria

1. **Reproduce MMaDA baseline** within 5% of reported performance
2. **Demonstrate continual learning** with <10% forgetting on previous tasks
3. **Maintain cross-modal coherence** during sequential learning
4. **Show efficiency gains** compared to full retraining

## 12. Code Organization

```
CoDA/
├── MMaDA/              # Original MMaDA codebase (baseline)
├── coda/               # CoDA continual learning modules
│   ├── models/         # Model extensions
│   ├── training/       # Continual learning training
│   ├── evaluation/    # Metrics and evaluation
│   └── data/          # Task sequences and datasets
├── experiments/        # Reproduction and experiments
└── docs/              # Documentation
```

## 13. Key Files and Their Purpose

- **`experiments/scripts/reproduce_*.py`**: Baseline reproduction
- **`coda/models/base_mmada.py`**: MMaDA wrapper for CoDA
- **`coda/models/replay_module.py`**: Generative replay implementation
- **`coda/models/preservation.py`**: Parameter importance and preservation
- **`coda/models/adapters.py`**: LoRA adapters for task-specific learning
- **`coda/training/trainer.py`**: Continual learning training loop
- **`coda/evaluation/metrics.py`**: Continual learning evaluation metrics

## 14. Questions for Discussion

1. **Task Selection**: Which task sequences are most interesting/important?
2. **Evaluation Metrics**: What metrics best capture continual learning success?
3. **Baseline Comparison**: Should we compare against full retraining or other CL methods?
4. **Scope**: Focus on specific modalities or general framework?

## 15. References

- Yang et al. (2025). MMaDA: Multimodal Large Diffusion Language Models. NeurIPS 2025.
- LLaDA: Diffusion-based language models
- Continual Learning literature (EWC, Replay, LoRA)
- Multimodal learning frameworks

---

**Last Updated**: [Current Date]
**Status**: Baseline Reproduction Phase
**Next Review**: [Date]


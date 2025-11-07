# CoDA: Continual Diffusion Adaptation

**Enabling Lifelong Learning in Unified Multimodal Diffusion Models**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

Current unified multimodal diffusion models like [MMaDA](https://github.com/Gen-Verse/MMaDA) achieve impressive joint text-image generation by treating both modalities as discrete diffusion problems. However, once training completes through their three-stage process (mixed Chain-of-Thought fine-tuning → UniGRPO optimization), these models become completely frozen, unable to adapt to new tasks or modalities without full retraining.

**CoDA addresses this fundamental limitation** by introducing continual learning mechanisms specifically designed for unified diffusion architectures, enabling these systems to acquire new capabilities over time without catastrophic forgetting.

## Key Innovation

While MMaDA unifies text and image generation through:
- Discrete token diffusion for both modalities
- Mixed Long Chain-of-Thought (CoT) fine-tuning
- Unified Group Relative Policy Optimization (UniGRPO)

**CoDA extends this with continual learning capabilities:**
- Diffusion-based generative replay with hierarchical prompt strategies
- Selective parameter preservation adapted to shared transformer architectures
- Modular low-rank adaptation with learned routing mechanisms
- Cross-modal coherence preservation during sequential learning

## Architecture
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

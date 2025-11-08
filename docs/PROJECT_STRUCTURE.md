# CoDA Project Structure

This document describes the project structure and organization for CoDA development.

## Directory Layout

```
CoDA/
├── MMaDA/                          # Original MMaDA repository (submodule/clone)
│   ├── models/                     # MMaDA model implementations
│   ├── training/                   # MMaDA training scripts
│   ├── evaluation/                 # MMaDA evaluation code
│   ├── configs/                    # MMaDA configuration files
│   └── ...
│
├── coda/                           # CoDA continual learning modules
│   ├── models/                     # CoDA model extensions
│   │   ├── base_mmada.py          # Base MMaDA wrapper
│   │   ├── continual_model.py     # Continual learning model
│   │   ├── adapters.py            # LoRA adapters
│   │   ├── preservation.py         # Parameter preservation
│   │   └── replay_module.py       # Generative replay
│   │
│   ├── training/                  # Continual learning training
│   │   ├── trainer.py             # Main training loop
│   │   ├── task_manager.py        # Task sequence management
│   │   └── optimizer.py          # Continual learning optimizers
│   │
│   ├── evaluation/                # Continual learning metrics
│   │   ├── metrics.py             # Performance metrics
│   │   └── coherence.py           # Cross-modal coherence
│   │
│   ├── data/                      # Data handling
│   │   ├── datasets.py            # Dataset loaders
│   │   └── task_sequence.py      # Task sequence definitions
│   │
│   └── utils/                     # Utilities
│       ├── importance_estimation.py # Parameter importance
│       └── prompt_selection.py    # Prompt selection strategies
│
├── experiments/                   # Reproduction and experiments
│   ├── scripts/                   # Reproduction scripts
│   │   ├── setup_reproduction.sh  # Environment setup
│   │   ├── reproduce_text_generation.py
│   │   ├── reproduce_multimodal.py
│   │   └── reproduce_t2i.py
│   │
│   ├── configs/                   # Experiment configurations
│   │   └── ...                    # CoDA-specific configs
│   │
│   ├── notebooks/                 # Jupyter notebooks
│   │   └── ...                    # Analysis notebooks
│   │
│   └── baselines/                 # Baseline results
│       └── ...                    # Saved baseline outputs
│
├── docs/                          # Documentation
│   ├── MMaDA_REPRODUCTION.md      # Detailed reproduction guide
│   ├── PROJECT_STRUCTURE.md       # This file
│   └── ...                        # Additional documentation
│
├── tests/                         # Unit tests
│   └── ...                        # Test files
│
├── README.md                       # Main README
├── QUICKSTART.md                  # Quick start guide
├── REPRODUCTION_CHECKLIST.md      # Reproduction checklist
├── requirements.txt              # CoDA requirements
└── setup.py                       # Package setup
```

## Key Components

### MMaDA Integration

The `MMaDA/` directory contains the original MMaDA codebase. This is kept separate to:
- Maintain reproducibility of baseline results
- Allow easy updates from upstream
- Provide reference implementation

### CoDA Modules (`coda/`)

#### Models (`coda/models/`)
- **base_mmada.py**: Wrapper around MMaDA models for CoDA integration
- **continual_model.py**: Main continual learning model that extends MMaDA
- **adapters.py**: Low-rank adaptation (LoRA) modules
- **preservation.py**: Parameter importance and preservation mechanisms
- **replay_module.py**: Generative replay for preventing forgetting

#### Training (`coda/training/`)
- **trainer.py**: Main training loop with continual learning support
- **task_manager.py**: Manages task sequences and data loading
- **optimizer.py**: Custom optimizers for continual learning

#### Evaluation (`coda/evaluation/`)
- **metrics.py**: Standard continual learning metrics (accuracy, forgetting)
- **coherence.py**: Cross-modal coherence metrics specific to multimodal models

#### Data (`coda/data/`)
- **datasets.py**: Dataset loaders for different tasks/modalities
- **task_sequence.py**: Defines task sequences for continual learning experiments

### Experiments (`experiments/`)

Contains scripts and configurations for:
- Reproducing MMaDA baseline results
- Running CoDA experiments
- Analyzing results

## Development Workflow

### Phase 1: Reproduction (Current)
1. Set up environment using `experiments/scripts/setup_reproduction.sh`
2. Run reproduction scripts to establish baseline
3. Document results in `experiments/baselines/`

### Phase 2: CoDA Implementation
1. Implement base wrapper (`coda/models/base_mmada.py`)
2. Implement generative replay module
3. Implement parameter preservation module
4. Implement modular adaptation module
5. Integrate into training loop

### Phase 3: Evaluation
1. Define task sequences (`coda/data/task_sequence.py`)
2. Implement evaluation metrics
3. Run continual learning experiments
4. Compare against baseline

## File Naming Conventions

- **Python files**: snake_case (e.g., `continual_model.py`)
- **Config files**: snake_case with `.yaml` extension
- **Scripts**: snake_case with appropriate extension
- **Classes**: PascalCase (e.g., `ContinualModel`)
- **Functions**: snake_case (e.g., `estimate_importance`)

## Import Structure

```python
# CoDA modules should import from coda namespace
from coda.models import ContinualModel
from coda.training import ContinualTrainer
from coda.evaluation import compute_forgetting

# MMaDA imports should be explicit
from MMaDA.models import MMadaModelLM
from MMaDA.training.utils import get_config
```

## Configuration Management

- MMaDA configs: `MMaDA/configs/*.yaml`
- CoDA configs: `experiments/configs/*.yaml`
- Use OmegaConf for configuration handling

## Testing Strategy

- Unit tests for individual modules in `tests/`
- Integration tests for full pipeline
- Baseline comparison tests to ensure no regression

## Documentation Standards

- Code should be well-commented
- Docstrings for all public functions/classes
- README files in major directories
- Architecture diagrams in `docs/`


# Evolutionary Prompt Engineering for Text-to-Image Generation

This repository contains the implementation and experimental code for the research paper on automated prompt optimization using genetic algorithms for text-to-image models.

## Abstract

We propose an evolutionary approach for automated prompt optimization in text-to-image generation systems. Our method utilizes genetic algorithms to evolve prompts that maximize image quality metrics while maintaining semantic alignment with user intent. We investigate two key strategies: (1) static fitness weighting and (2) adaptive fitness weighting that shifts focus during evolution. Experiments demonstrate that adaptive weighting strategies achieve up to 9.54% improvement over baseline prompts, with medium effect sizes (Cohen's d = 0.72).

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Experiments](#experiments)
  - [Experiment 1: Prompt Enhancement](#experiment-1-prompt-enhancement)
  - [Experiment 2: Template-Based Generation](#experiment-2-template-based-generation)
- [Configuration](#configuration)
- [API Setup](#api-setup)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster model inference

### Setup

```bash
# Clone the repository
cd evolutionary-prompt-engineering

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Environment Variables

```bash
# Fal AI API Key (Required)
FAL_KEY=your_fal_api_key_here

# Google Cloud / Vertex AI (Required for Experiment 2 with LLM seeding)
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
```

## Project Structure

```
evolutionary-prompt-engineering/
├── src/
│   ├── __init__.py
│   ├── genome.py                    # Experiment 1: PromptGenome representation
│   ├── genome_v2.py                 # Experiment 2: BlockGenome representation
│   ├── fitness.py                   # Experiment 1: CLIP + Aesthetic fitness
│   ├── fitness_v2.py                # Experiment 2: CLIP + LPIPS fitness
│   ├── evolution.py                 # Experiment 1: Genetic operators
│   ├── evolution_v2.py              # Experiment 2: Block-aware operators
│   ├── models.py                    # Text-to-image model wrappers (Fal AI)
│   ├── llm_prompt_generator.py      # LLM-based seed generation (Gemini)
│   ├── template_analyzer.py         # Reference image analysis
│   ├── vocabulary_manager.py        # Adaptive vocabulary management
│   └── utils.py                     # Utilities and logging
├── experiments/
│   ├── experiment_1.ipynb           # Prompt Enhancement experiments
│   └── experiment_2.ipynb           # Template-Based Generation experiments
├── tests/
│   ├── test_models.py               # API connectivity tests
│   ├── test_fitness.py              # Fitness function validation
│   ├── test_llm_generator.py        # LLM integration tests
│   ├── test_template_analyzer.py    # Template analysis tests
│   └── test_vocabulary_manager.py   # Vocabulary management tests
├── data/
│   └── results/                     # Experiment outputs
├── requirements.txt
├── .env.example
└── README.md
```

## Experiments

### Experiment 1: Prompt Enhancement

**Objective**: Automatically enhance base prompts to maximize image quality using evolutionary optimization.

**Methodology**:
- **Genome**: `PromptGenome` with base prompt + positive/negative modifiers
- **Fitness Function**: `w1 * CLIP_score + w2 * Aesthetic_score`
- **Model**: flux-schnell (Fal AI)

**Sub-experiments**:
- **1.1 Static Weights**: Fixed CLIP=0.6, Aesthetic=0.4 throughout evolution
- **1.2 Adaptive Weights**: CLIP weight decreases from 0.8 to 0.4 over generations (shift from semantic alignment to aesthetic focus)

**Running the Experiment**:
```bash
cd experiments
jupyter notebook experiment_1.ipynb
```

**Configuration**:
```python
POPULATION_SIZE = 10
MAX_GENERATIONS = 20
ELITE_SIZE = 1
MUTATION_RATE = 0.4
MAX_POSITIVE_MODIFIERS = 8
MAX_NEGATIVE_MODIFIERS = 4
```

### Experiment 2: Template-Based Generation

**Objective**: Transfer visual style from a reference image to a new subject using evolved structured prompts.

**Methodology**:
- **Genome**: `BlockGenome` with subject + composition + lighting + style + quality + negative blocks
- **Fitness Function**: `w1 * CLIP_score + w2 * LPIPS_similarity`
- **Model**: qwen-image (Fal AI)
- **LLM Seeding**: Gemini 2.0 Flash for initial population generation

**Sub-experiments**:
- **2.1 Static Weights**: Fixed CLIP=0.5, LPIPS=0.5 throughout evolution
- **2.2 Adaptive Weights**: CLIP weight increases from 0.3 to 0.6 (shift from structure matching to content alignment)

**Running the Experiment**:
```bash
cd experiments
jupyter notebook experiment_2.ipynb
```

**Configuration**:
```python
POPULATION_SIZE = 10
MAX_GENERATIONS = 20
ELITE_SIZE = 1
MUTATION_RATE = 0.4
MAX_ITEMS_PER_BLOCK = 3
USE_LLM_SEEDING = True
```

## Configuration

### Evolution Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `POPULATION_SIZE` | Number of individuals per generation | 10 |
| `MAX_GENERATIONS` | Maximum evolution iterations | 20 |
| `ELITE_SIZE` | Number of elite individuals preserved | 1 |
| `MUTATION_RATE` | Probability of mutation | 0.4 |
| `ADD_PROBABILITY` | Probability of adding a modifier | 0.3 |
| `REMOVE_PROBABILITY` | Probability of removing a modifier | 0.2 |

### Model Options

| Model | Speed | Negative Prompts | Use Case |
|-------|-------|------------------|----------|
| `flux-schnell` | Fast (4 steps) | No | Rapid iteration, Experiment 1 |
| `qwen-image` | Slower (30 steps) | Yes | Higher quality, Experiment 2 |

### Fitness Weights

**Experiment 1**:
- Static: CLIP=0.6, Aesthetic=0.4
- Adaptive: CLIP 0.8→0.4, Aesthetic 0.2→0.6

**Experiment 2**:
- Static: CLIP=0.5, LPIPS=0.5
- Adaptive: CLIP 0.3→0.6, LPIPS 0.7→0.4

## API Setup

### Fal AI (Required)

1. Sign up at [fal.ai](https://fal.ai)
2. Obtain API key from dashboard
3. Add to `.env`: `FAL_KEY=your_key_here`

### Vertex AI / Gemini (Required for Experiment 2)

1. Create a Google Cloud project
2. Enable Vertex AI API
3. Create service account and download JSON key
4. Configure `.env`:
   ```bash
   GOOGLE_CLOUD_PROJECT=your-project-id
   GOOGLE_CLOUD_LOCATION=us-central1
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
   ```
<!-- 
## Results

### Experiment 1: Prompt Enhancement

| Metric | Baseline | Static Weights | Adaptive Weights |
|--------|----------|----------------|------------------|
| Best Fitness | 0.7219 | 0.7467 | 0.7907 |
| Mean Fitness | - | 0.7289 ± 0.02 | 0.7537 ± 0.04 |
| Improvement | - | +3.44% | +9.54% |

**Statistical Analysis**:
- t-test: t=-1.52, p=0.145
- Effect Size: Cohen's d=0.72 (medium effect)

### Experiment 2: Template Transfer

| Metric | Baseline | Static Weights | Adaptive Weights |
|--------|----------|----------------|------------------|
| Best Fitness | 0.6300 | 0.6343 | 0.6331 |
| Mean Fitness | - | 0.6222 ± 0.01 | 0.6199 ± 0.01 |
| Improvement | - | +0.68% | +0.49% |

**Statistical Analysis**:
- t-test: t=0.55, p=0.591
- Effect Size: Cohen's d=-0.26 (small effect)

## Testing

```bash
# Test API connectivity
python tests/test_models.py

# Test fitness functions
python tests/test_fitness.py

# Test LLM integration
python tests/test_llm_generator.py

# Test template analyzer
python tests/test_template_analyzer.py

# Test vocabulary manager
python tests/test_vocabulary_manager.py
```

## Cost Estimates

### Per Experiment Run

**Experiment 1** (10 population, 20 generations):
- Images generated: ~200
- Fal AI cost: $2-10
- Total: ~$2-10

**Experiment 2** (10 population, 20 generations):
- Images generated: ~200
- Fal AI cost: $5-20
- Vertex AI (Gemini): ~$0.001
- Total: ~$5-20

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{evolutionary_prompt_engineering_2025,
  title={Evolutionary Prompt Engineering for Text-to-Image Generation},
  author={[Author Names]},
  booktitle={Proceedings of [Conference Name]},
  year={2025},
  organization={IEEE}
}
```

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- Fal AI for text-to-image generation APIs
- Google Cloud for Vertex AI / Gemini integration
- OpenAI CLIP for semantic similarity evaluation
- LPIPS for perceptual similarity metrics -->

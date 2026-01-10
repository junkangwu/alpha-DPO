# α-DPO: Adaptive Reward Margin Direct Preference Optimization

This repository contains the implementation of α-DPO (alpha-DPO) based on the latest [SimPO](https://github.com/princeton-nlp/SimPO) codebase.

> **Note**: This implementation is updated from the original [α-DPO repository](https://github.com/junkangwu/alpha-DPO/tree/fab32091cd1e36a275ad0b08adb09920c01cd65f) to be compatible with newer versions of `transformers`, `trl`, and other dependencies.

## Key Changes

The original α-DPO code was built on an older version of SimPO. Due to version incompatibilities with newer packages (especially `transformers>=4.44` and `trl>=0.9`), we re-implemented α-DPO on the latest SimPO codebase.

**Main modifications:**
- `scripts/alpha_dpo_trainer.py` - α-DPO trainer with adaptive reward margin
- `scripts/run_alpha_dpo.py` - Unified training script supporting both SimPO and α-DPO
- `scripts/simpo_config.py` - Extended config with α-DPO parameters (`alpha`, `ln`, `trainer_type`)
- `training_configs/*-alpha-dpo*.yaml` - Training configurations for different models

## Installation

```bash
conda create -n alpha-dpo python=3.10 && conda activate alpha-dpo
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## Training

```bash
# Llama-3-8B-Instruct
CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file accelerate_configs/deepspeed_zero3.yaml \
    scripts/run_alpha_dpo.py training_configs/llama-3-8b-instruct-alpha-dpo.yaml

# Llama-3-8B-Instruct v0.2 (ArmoRM annotated)
CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file accelerate_configs/deepspeed_zero3.yaml \
    scripts/run_alpha_dpo.py training_configs/llama-3-8b-instruct-alpha-dpo-v2.yaml

# Mistral-7B-Instruct
CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file accelerate_configs/deepspeed_zero3.yaml \
    scripts/run_alpha_dpo.py training_configs/mistral-7b-instruct-alpha-dpo.yaml

# Gemma-2-9B-IT
CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file accelerate_configs/deepspeed_zero3.yaml \
    scripts/run_alpha_dpo.py training_configs/gemma-2-9b-it-alpha-dpo.yaml
```

## Hyperparameters

| Model | β | γ/β | α | lr |
|-------|---|-----|---|-----|
| Llama-3-8B-Instruct | 2.5 | 0.6 | 0.2 | 1e-6 |
| Llama-3-8B-Instruct v0.2 | 10 | 0.4 | 0.2 | 1e-6 |
| Mistral-7B-Instruct | 2.5 | 0.15 | 0.05 | 6e-7 |
| Gemma-2-9B-IT | 10 | 0.4 | 0.05 | 8e-7 |

## Acknowledgement

- [SimPO](https://github.com/princeton-nlp/SimPO)

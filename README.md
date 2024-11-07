# LaNE

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Static Badge](https://img.shields.io/badge/Project-Page-a)](https://philipzrh.com/lane/)

## Overview
Implementation of CoRL 2024 paper: [LaNE: Accelerating Visual Sparse-Reward Learning with Latent Nearest-Demonstration-Guided Explorations](https://philipzrh.com/lane/)

LaNE is a an efficient reinforcement learning (RL) framework for learning image-based robot manipulation tasks. It densifies sparse task rewards with exploration bonuses around demonstrations.

## Prerequisites
- Python 3.10
- CUDA-compatible GPU (recommended)

## Installation

1. Create and activate a conda environment:
```bash
conda create --name lane python=3.10.12
conda activate lane
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Collecting Demonstrations
To collect and save task demonstrations, execute the following scripts:
```bash
python robosuite_utils/save_demo_lift.py
python robosuite_utils/save_demo_door.py
python robosuite_utils/save_demo_stack.py
python robosuite_utils/save_demo_pick_place_can.py
```

### Training
To train the LaNE model on different tasks, use the following scripts:
```bash
bash scripts/lane_dino/robosuite_lift.sh
bash scripts/lane_dino/robosuite_door.sh
bash scripts/lane_dino/robosuite_stack.sh
bash scripts/lane_dino/robosuite_pick_place_can.sh
```

Scripts correponding to some baselines and ablation studies are also provided in the `scripts` folder.

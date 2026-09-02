# C2: Cenerative learning module enhanced decision transformer with Constraint-aware loss for auto-bidding (ICLR 2026 Workshop AIMS Submission)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository is the official implementation of the paper **C2: Cenerative learning module enhanced decision transformer with Constraint-aware loss for auto-bidding** (submitted to ICLR 2026 Workshop AIMS).

## 1. Method Overview
Decision Transformer (DT) is promising for generative auto-bidding but faces two key limitations:  
- Insufficient cross-correlation modeling among state, action, and Return-to-Go (RTG) sequences;  
- Indiscriminate learning of both optimal and suboptimal historical bidding behaviors.  

To address these, we propose the C2 framework with two core innovations:  
### 1.1 Cross Learning Block (CLB)
A cross-attention based module to explicitly model inter-sequence correlations between state, action, and RTG, enhancing the capture of auction trajectory dependencies.  
### 1.2 Constraint-aware Loss (CL)
A novel loss function incorporating budget and Cost-Per-Acquisition (CPA) constraints, enabling selective learning of optimal bidding trajectories while filtering suboptimal behaviors.

## 2. Key Experimental Results
Evaluations on the AuctionNet dataset demonstrate:
| Model          | Score  | Gain over Vanilla DT |
|----------------|--------|----------------------|
| Vanilla DT     | 33.3   | -                    |
| DT + CL        | 34.2   | 2.7%                 |
| CLB-DT         | 35.4   | 6.3%                 |
| C2 (CLB-DT+CL) | 38.3   | 15.0%                |
- C2 outperforms SOTA baseline GAVE by up to 4.24% across 50%-150% budget settings.
- Ablation studies confirm complementary synergy of CLB (6.3% gain) and CL (2.7% gain).

## 3. Installation

The code is tested with Python 3.9 and PyTorch 2.6.0. Install the required packages with:

```bash
pip install -r requirements.txt
```

## 4. Dataset Preparation

Download the AuctionNet dataset from the [official repository](https://github.com/alimama-tech/AuctionNet) and follow its instructions to prepare the data. Place the processed training file and evaluation period files as follows:

```text
data/traffic/
├── training_data_all-rlData.csv
└── period-*.csv
```

The evaluation loader automatically creates `raw_data.pickle` beside the evaluation CSV on its first run and reuses it in subsequent runs.

## 5. Training

Run the default C2 configuration (`hidden_size=256`, `n_layer=6`, `n_head=4`, `n_inner=256`, and `K=10`) for 10,000 training steps:

```bash
python run/train_C2.py \
  --data_path data/traffic/training_data_all-rlData.csv \
  --output_dir saved_model/C2 \
  --device cuda:0
```

Checkpoints are saved from step 3,500 to 10,000 at intervals of 100 steps.

## 6. Evaluation

Evaluate all saved checkpoints on the default evaluation split:

```bash
EVAL_DEVICE=cuda:0 RTG_SCALE=1.3 python run/run_evaluate.py \
  --ckpt_dir saved_model/C2
```

The script reports the score, constraint exceed rate, and reward of every checkpoint, followed by the checkpoint with the highest score.

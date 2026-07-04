# C2: Cenerative learning module enhanced decision transformer with Constraint-aware loss for auto-bidding (ICLR 2026 Workshop AIMS Submission)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository is the official implementation of the paper **C2: Cenerative learning module enhanced decision transformer with Constraint-aware loss for auto-bidding** (submitted to ICLR 2026 Workshop AIMS).

> **Code: Coming Soon**

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
| Vanilla DT     | 32.4   | -                    |
| DT + CL        | 35.8   | 10.5%                |
| CLB-DT         | 36.5   | 12.7%                |
| C2 (CLB-DT+CL) | 37.8   | 16.7%                |
- C2 outperforms SOTA baseline GAVE by up to 1.61% across 50%-150% budget settings.
- Ablation studies confirm complementary synergy of CLB (12.7% gain) and CL (10.5% gain).

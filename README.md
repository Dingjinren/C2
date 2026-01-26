# C2: Constraint-aware Cross Learning Framework for Generative Auto-bidding (ICML 2026 Submission)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository is the official implementation of the paper **C2: Enhancing Decision Transformer with Cross Learning Block and Constraint-aware Loss for Auto-bidding** (submitted to ICML 2026).

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
All experiments are conducted on the AuctionNet dataset (consistent with [su2024a]) under ICML reproducibility standards. Key results (average over 5 independent runs):

| Model          | Core Metric Score | Relative Gain over Vanilla DT |
|----------------|-------------------|-------------------------------|
| Vanilla DT     | 33.3              | -                             |
| DT + CL        | 35.7              | 7.2%                          |
| CLB-DT         | 36.7              | 10.2%                         |
| C2 (CLB-DT+CL) | 38.4              | 15.3%                         |

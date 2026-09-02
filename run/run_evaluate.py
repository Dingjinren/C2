import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import math

import numpy as np

from bidding_train_env.dataloader.test_dataloader import TestDataLoader
from bidding_train_env.environment.offline_env import OfflineEnv
from bidding_train_env.strategy import C2BiddingStrategy

os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def getScore_nips(reward, cpa, cpa_constraint):
    beta = 2
    penalty = 1
    if cpa > cpa_constraint:
        penalty = pow(cpa_constraint / (cpa + 1e-10), beta)
    return penalty * reward


def run_test(policy_load_dir, data_loader, model_kwargs=None):
    env = OfflineEnv()
    keys = data_loader.keys
    total_score = total_reward = exceed = 0.0

    for key in keys:
        num_ts, pValues, pValueSigmas, leastWinningCosts, budget, cpa, _ = data_loader.mock_data(key)
        budget = 1. * budget
        agent = C2BiddingStrategy(
            budget=budget, cpa=cpa, load_dir=policy_load_dir,
            **(model_kwargs or {}),
        )
        rewards = np.zeros(num_ts)
        hist = {k: [] for k in ('historyBids', 'historyAuctionResult', 'historyImpressionResult',
                                'historyLeastWinningCost', 'historyPValueInfo')}

        for t in range(num_ts):
            pv, sig, lwc = pValues[t], pValueSigmas[t], leastWinningCosts[t]
            if agent.remaining_budget < env.min_remaining_budget:
                bid = np.zeros(pv.shape[0])
            else:
                bid, _ = agent.bidding(t, pv, sig, hist["historyPValueInfo"], hist["historyBids"],
                                       hist["historyAuctionResult"], hist["historyImpressionResult"],
                                       hist["historyLeastWinningCost"])

            _, tick_cost, tick_status, tick_conv = env.simulate_ad_bidding(pv, sig, bid, lwc)
            over = max((np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)
            while over > 0:
                pv_idx = np.where(tick_status == 1)[0]
                drop = np.random.choice(pv_idx, int(math.ceil(pv_idx.shape[0] * over)), replace=False)
                bid[drop] = 0
                _, tick_cost, tick_status, tick_conv = env.simulate_ad_bidding(pv, sig, bid, lwc)
                over = max((np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)

            agent.remaining_budget -= np.sum(tick_cost)
            rewards[t] = np.sum(tick_conv)
            hist["historyPValueInfo"].append(np.stack([pv, sig], axis=1))
            hist["historyBids"].append(bid)
            hist["historyLeastWinningCost"].append(lwc)
            hist["historyAuctionResult"].append(np.stack([tick_status, tick_status, tick_cost], axis=1))
            hist["historyImpressionResult"].append(np.stack([tick_conv, tick_conv], axis=1))

        all_reward = np.sum(rewards)
        cpa_real = np.clip((agent.budget - agent.remaining_budget) / (all_reward + 1e-10), 0, 100.)
        total_score += getScore_nips(all_reward, cpa_real, agent.cpa)
        total_reward += all_reward
        if cpa_real > agent.cpa:
            exceed += 1

    n = len(keys)
    return total_score / n, exceed / n, total_reward / n


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='C2')
    parser.add_argument('--ckpt_dir', default=None,
                        help='directory containing epoch subdirs (default: saved_model/<method>/)')
    parser.add_argument('--ckpt_start', type=int, default=3500)
    parser.add_argument('--ckpt_end', type=int, default=10000)
    parser.add_argument('--ckpt_step', type=int, default=100)
    parser.add_argument('--eval_file', default='data/traffic/period-19.csv')
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--n_inner', type=int, default=256)
    parser.add_argument('--K', type=int, default=10)
    args = parser.parse_args()
    if args.ckpt_step <= 0:
        parser.error('--ckpt_step must be positive')

    model_name = args.model_name
    ckpt_dir = args.ckpt_dir or f"saved_model/{model_name}"
    policy_load_dir = [os.path.join(ckpt_dir, str(i))
                       for i in range(args.ckpt_start, args.ckpt_end + 1, args.ckpt_step)]
    policy_load_dir = [p for p in policy_load_dir if os.path.exists(os.path.join(p, 'dt.pt'))]
    if not policy_load_dir:
        if os.path.exists(os.path.join(ckpt_dir, 'dt.pt')):
            policy_load_dir = [ckpt_dir]
        else:
            logger.error(f'No checkpoints found in {ckpt_dir}')
            sys.exit(1)

    logger.info(f"Loading test data from {args.eval_file}")
    data_loader = TestDataLoader(file_path=args.eval_file)

    logger.info(f"Evaluating {len(policy_load_dir)} ckpts on {args.eval_file}")
    best_score, best_dir = float('-inf'), ""
    model_kwargs = {
        "hidden_size": args.hidden_size,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "n_inner": args.n_inner,
        "K": args.K,
    }
    for policy in policy_load_dir:
        score, exc, reward = run_test(policy, data_loader, model_kwargs)
        logger.info(f"ckpt {policy} score={score:.4f} exceed={exc:.4f} reward={reward:.4f}")
        if score > best_score:
            best_score, best_dir = score, policy

    logger.info(f"best score: {best_score:.4f}, dir: {best_dir}")

import os
import pickle

import numpy as np
import torch

from bidding_train_env.baseline.C2 import C2
from bidding_train_env.strategy.base_bidding_strategy import BaseBiddingStrategy


class C2BiddingStrategy(BaseBiddingStrategy):
    """Decision-Transformer (C2) player strategy."""

    def __init__(self, budget=100, name="DT-Player", cpa=2, category=1, load_dir=None):
        super().__init__(budget, name, cpa, category)

        model_path = os.path.join(load_dir, "dt.pt")
        picklePath = os.path.join(load_dir, "normalize_dict.pkl")

        device = os.environ.get("EVAL_DEVICE")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        with open(picklePath, 'rb') as f:
            normalize_dict = pickle.load(f)

        self.model = C2(state_dim=16, act_dim=1,
                        state_mean=normalize_dict["state_mean"],
                        state_std=normalize_dict["state_std"],
                        target_return=1., device=device)
        self.model.load_net(model_path, device=device)
        self.model.to(device)
        self.model.eval()

        self.remaining_budget_last = self.budget

    def reset(self):
        self.remaining_budget = self.budget

    def bidding(self, timeStepIndex, pValues, pValueSigmas, historyPValueInfo, historyBid,
                historyAuctionResult, historyImpressionResult, historyLeastWinningCost,
                actual_excuted_action=None):
        cost_cur = self.remaining_budget_last - self.remaining_budget

        time_left = (48 - timeStepIndex) / 48
        budget_left = self.remaining_budget / self.budget if self.budget > 0 else 0
        history_xi = [r[:, 0] for r in historyAuctionResult]
        history_pValue = [r[:, 0] for r in historyPValueInfo]
        history_conversion = [r[:, 1] for r in historyImpressionResult]

        def _mean(seq):
            return np.mean([np.mean(x) for x in seq]) if seq else 0

        def _mean_last(seq, n):
            return _mean(seq[-n:]) if seq else 0

        test_state = np.array([
            time_left, budget_left,
            _mean(historyBid), _mean_last(historyBid, 3),
            _mean(historyLeastWinningCost), _mean(history_pValue),
            _mean(history_conversion), _mean(history_xi),
            _mean_last(historyLeastWinningCost, 3), _mean_last(history_pValue, 3),
            _mean_last(history_conversion, 3), _mean_last(history_xi, 3),
            np.mean(pValues), len(pValues),
            sum(len(b) for b in historyBid[-3:]) if historyBid else 0,
            sum(len(b) for b in historyBid) if historyBid else 0,
        ])

        if timeStepIndex == 0:
            self.model.init_eval()

        pre_reward = sum(history_conversion[-1]) if history_conversion else None
        pre_cost = cost_cur if history_conversion else None

        alpha = self.model.take_actions(
            test_state, actual_excuted_action,
            target_return=self.budget / self.cpa * float(os.environ.get('RTG_SCALE', '1.3')),
            pre_reward=pre_reward, pre_cost=pre_cost, cpa_constrain=self.cpa,
        )
        self.remaining_budget_last = self.remaining_budget

        return alpha * pValues, alpha

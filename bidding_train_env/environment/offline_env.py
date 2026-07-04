import numpy as np


class OfflineEnv:
    """Simulate an advertising bidding environment."""

    def __init__(self, min_remaining_budget: float = 0.01):
        self.min_remaining_budget = min_remaining_budget

    def simulate_ad_bidding(self, pValues, pValueSigmas, bids, leastWinningCosts):
        tick_status = bids >= leastWinningCosts
        tick_cost = leastWinningCosts * tick_status
        values = np.random.normal(loc=pValues, scale=pValueSigmas) * tick_status
        tick_value = np.clip(values, 0, 1)
        tick_conversion = np.random.binomial(n=1, p=tick_value)
        return tick_value, tick_cost, tick_status, tick_conversion

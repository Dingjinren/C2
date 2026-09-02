import ast
import random
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


EPS = 1e-10


def _literal(value):
    if not isinstance(value, str):
        return value
    value = re.sub(r"np\.\w+\(([^)]+)\)", r"\1", value.strip())
    return ast.literal_eval(value)


class EpisodeReplayBuffer(Dataset):
    def __init__(self, state_dim, act_dim, data_path=None, df=None,
                 max_ep_len=48, scale=80, K=10, device="cpu",
                 threshold=1.15, beta=4.0, sampler_lambda=1.0,
                 reward_tolerance=0.2, sampler_tolerance=0.2):
        super().__init__()
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_ep_len = max_ep_len
        self.scale = scale
        self.K = K

        if df is None:
            if data_path is None:
                raise ValueError("data_path is required when df is not provided")
            training_data = pd.read_csv(data_path)
        else:
            training_data = df.reset_index(drop=True).copy()

        required = {
            "state", "action", "reward", "done",
            "realAllCost", "budget", "CPAConstraint",
        }
        missing = sorted(required.difference(training_data.columns))
        if missing:
            raise ValueError(f"training data missing columns: {missing}")
        training_data["state"] = training_data["state"].map(_literal)

        trajectories = []
        states, actions, raw_rewards = [], [], []
        for _, row in training_data.iterrows():
            states.append(np.asarray(row["state"], dtype=np.float64))
            actions.append(float(row["action"]))
            raw_rewards.append(float(row["reward"]))

            if not row["done"]:
                continue
            if len(states) == 1:
                states, actions, raw_rewards = [], [], []
                continue

            budget = float(row["budget"])
            constraint = float(row["CPAConstraint"])
            terminal_cost = float(row["realAllCost"])
            shaped_rewards = []
            for index, reward in enumerate(raw_rewards):
                recursive_reward = sum(shaped_rewards) + reward
                cumulative_cost = (
                    terminal_cost if index == len(raw_rewards) - 1
                    else (1.0 - states[index][1]) * budget
                )
                current_cpa = cumulative_cost / (recursive_reward + EPS)
                coefficient = constraint / (current_cpa + EPS)
                distance = abs(coefficient - 1.0)
                factor = 1.0 / (
                    1.0 + (distance > reward_tolerance) * distance
                )
                shaped_rewards.append(reward * factor)

            shaped_rewards = np.asarray(shaped_rewards, dtype=np.float64)
            cpa_ratio = (
                terminal_cost / (shaped_rewards.sum() + EPS)
                / (constraint + EPS)
            )
            cpa_factor = 1.0 / cpa_ratio if cpa_ratio > threshold else 1.0
            utilization = terminal_cost / (budget + EPS)
            trajectory_weight = (cpa_factor * utilization) ** beta

            trajectories.append({
                "observations": np.asarray(states, dtype=np.float64),
                "actions": np.asarray(actions, dtype=np.float64).reshape(-1, 1),
                "rewards": shaped_rewards.reshape(-1, 1),
                "cpa_ratio": cpa_ratio,
                "penalty": np.full(
                    (len(states), 1), trajectory_weight, dtype=np.float64
                ),
            })
            states, actions, raw_rewards = [], [], []

        if not trajectories:
            raise ValueError("training data contains no multi-step trajectories")

        self.trajectories = trajectories
        self.traj_lens = np.asarray([len(t["observations"]) for t in trajectories])
        self.returns = np.asarray([t["rewards"].sum() for t in trajectories])
        all_states = np.concatenate([t["observations"] for t in trajectories])
        self.state_mean = all_states.mean(axis=0)
        self.state_std = np.maximum(all_states.std(axis=0), 1e-6)

        self.sorted_inds = np.argsort(self.returns)
        length_weight = self.traj_lens[self.sorted_inds].astype(np.float64)
        cpa_ratio = np.asarray([t["cpa_ratio"] for t in trajectories])[
            self.sorted_inds
        ]
        distance = np.abs(cpa_ratio - 1.0)
        proximity = 1.0 / (
            1.0 + (distance > sampler_tolerance) * distance
        )
        sample_weight = length_weight * np.power(proximity, sampler_lambda)
        self.p_sample = sample_weight / sample_weight.sum()

    def __len__(self):
        return len(self.sorted_inds)

    def __getitem__(self, index):
        trajectory = self.trajectories[int(self.sorted_inds[index])]
        end_t = random.randint(0, len(trajectory["rewards"]) - 1)
        start_t = max(0, end_t - self.K + 1)

        states = trajectory["observations"][start_t:end_t + 1]
        actions = trajectory["actions"][start_t:end_t + 1]
        penalty = trajectory["penalty"][start_t:end_t + 1]
        timesteps = np.arange(start_t, start_t + len(states))
        timesteps = np.minimum(timesteps, self.max_ep_len - 1)
        rtg = self.discount_cumsum(
            trajectory["rewards"][start_t:], gamma=1.0
        )[:len(states) + 1] / self.scale
        if len(rtg) == len(states):
            rtg = np.concatenate([rtg, np.zeros((1, 1))])

        length = len(states)
        padding = self.K - length
        states = np.concatenate([np.zeros((padding, self.state_dim)), states])
        states = (states - self.state_mean) / self.state_std
        actions = np.concatenate([np.zeros((padding, self.act_dim)), actions])
        penalty = np.concatenate([np.zeros((padding, 1)), penalty])
        rtg = np.concatenate([np.zeros((padding, 1)), rtg])
        timesteps = np.concatenate([np.zeros(padding), timesteps + 1])
        attention_mask = np.concatenate([np.zeros(padding), np.ones(length)])

        tensor = lambda value, dtype: torch.as_tensor(
            value, dtype=dtype, device=self.device
        )
        return (
            tensor(states, torch.float32),
            tensor(actions, torch.float32),
            tensor(rtg, torch.float32),
            tensor(timesteps, torch.long),
            tensor(attention_mask, torch.float32),
            tensor(penalty, torch.float32),
        )

    @staticmethod
    def discount_cumsum(values, gamma=1.0):
        result = np.zeros_like(values)
        result[-1] = values[-1]
        for index in reversed(range(values.shape[0] - 1)):
            result[index] = values[index] + gamma * result[index + 1]
        return result

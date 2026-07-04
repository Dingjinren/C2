import ast
import random
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class EpisodeReplayBuffer(Dataset):
    def __init__(self, state_dim, act_dim, data_path=None, df=None,
                 max_ep_len=48, scale=80, K=6, device='cpu'):
        super().__init__()
        self.device = device
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_ep_len = max_ep_len
        self.scale = scale
        self.K = K

        if df is None:
            assert data_path is not None
            print(f'loading data from {data_path}...')
            training_data = pd.read_csv(data_path)
        else:
            training_data = df.reset_index(drop=True).copy()

        def clean_numpy_literals(s):
            return re.sub(r"np\.\w+\(([^)]+)\)", r"\1", s)

        def safe_literal_eval(val):
            if pd.isna(val):
                return val
            try:
                if isinstance(val, str):
                    val = clean_numpy_literals(val.strip())
                return ast.literal_eval(val)
            except Exception as e:
                print(f"parse failed: {e}, value: {repr(val)}")
                return val

        if not isinstance(training_data["state"].iloc[0], (list, tuple)):
            training_data["state"] = training_data["state"].apply(safe_literal_eval)
        training_data["next_state"] = training_data["state"].shift(-1)
        training_data.at[training_data.index[-1], 'next_state'] = training_data.at[0, 'state']

        self.states, self.rewards, self.actions = [], [], []
        self.returns, self.traj_lens, self.penalty = [], [], []
        total_cpa_ratio, cpa_constraints = [], []

        state, reward, action = [], [], []
        beta = 3.5
        bias = 0.2
        over_thresh = 1.25

        for _, row in training_data.iterrows():
            state.append(row["state"])
            reward.append(row['reward'])
            action.append(row["action"])

            curr_coef = row['CPAConstraint'] / (((1 - row["state"][1]) * row['budget'] / (sum(reward) + 1e-10)) + 1e-10)
            t_penalty = 1.0 / (1.0 + (abs(curr_coef - 1.0) > bias) * abs(curr_coef - 1.0))
            reward[-1] = reward[-1] * t_penalty

            if row["done"]:
                if len(state) != 1:
                    r_arr = np.expand_dims(np.array(reward), axis=1)
                    ep_cpa_ratio = ((1 - row["state"][1]) * row['budget'] + 1e-5) / (np.sum(reward) + 1e-5)
                    cost_ratio = ep_cpa_ratio / (row['CPAConstraint'] + 1e-10)
                    ep_penalty = np.where(cost_ratio > over_thresh, 1.0 / cost_ratio, 1.0) * (1 - row["state"][1])
                    ep_penalty = pow(ep_penalty, beta)

                    self.states.append(np.array(state))
                    self.actions.append(np.expand_dims(np.array(action), axis=1))
                    self.rewards.append(r_arr)
                    self.penalty.append(np.ones(r_arr.shape) * ep_penalty)
                    self.returns.append(sum(reward))
                    self.traj_lens.append(len(state))
                    total_cpa_ratio.append(ep_cpa_ratio)
                    cpa_constraints.append(row['CPAConstraint'])
                state, reward, action = [], [], []

        self.traj_lens = np.array(self.traj_lens)
        self.returns = np.array(self.returns)

        tmp_states = np.concatenate(self.states, axis=0)
        self.state_mean = np.mean(tmp_states, axis=0)
        self.state_std = np.std(tmp_states, axis=0)

        self.trajectories = [
            {"observations": self.states[i], "actions": self.actions[i],
             "rewards": self.rewards[i], "penalty": self.penalty[i]}
            for i in range(len(self.states))
        ]

        sorted_inds = np.argsort(self.returns)
        self.sorted_inds = sorted_inds

        p_sample = self.traj_lens[self.sorted_inds] / sum(self.traj_lens[self.sorted_inds])
        cost_ratio = np.array(total_cpa_ratio) / np.array(cpa_constraints)
        sample_penalty = 1.0 / (1.0 + (abs(cost_ratio - 1.0) > bias) * abs(cost_ratio - 1.0))
        p_sample = p_sample * sample_penalty
        self.p_sample = p_sample / sum(p_sample)

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, index):
        traj = self.trajectories[int(self.sorted_inds[index])]
        end_t = random.randint(0, traj['rewards'].shape[0] - 1)
        start_t = max(0, end_t - self.K + 1)

        s = traj['observations'][start_t: end_t + 1]
        a = traj['actions'][start_t: end_t + 1]
        p = traj['penalty'][start_t: end_t + 1].reshape(-1, 1)

        timesteps = np.arange(start_t, start_t + s.shape[0])
        timesteps[timesteps >= self.max_ep_len] = self.max_ep_len - 1

        rtg = self.discount_cumsum(traj['rewards'][start_t:], gamma=1.0)[:s.shape[0] + 1].reshape(-1, 1)
        rtg = rtg / self.scale
        if rtg.shape[0] <= s.shape[0]:
            rtg = np.concatenate([rtg, np.zeros((1, 1))], axis=0)

        tlen = s.shape[0]
        pad = self.K - tlen
        s = np.concatenate([np.zeros((pad, self.state_dim)), s], axis=0)
        s = (s - self.state_mean) / self.state_std
        a = np.concatenate([np.zeros((pad, self.act_dim)), a], axis=0)
        p = np.concatenate([np.zeros((pad, 1)), p], axis=0)
        rtg = np.concatenate([np.zeros((pad, 1)), rtg], axis=0)
        timesteps = np.concatenate([np.zeros(pad), timesteps + 1], axis=0)
        mask = np.concatenate([np.zeros(pad), np.ones(tlen)], axis=0)

        to_f = lambda x: torch.from_numpy(x).to(dtype=torch.float32, device=self.device)
        return (
            to_f(s), to_f(a), to_f(rtg),
            torch.from_numpy(timesteps).to(dtype=torch.long, device=self.device),
            torch.from_numpy(mask).to(device=self.device),
            to_f(p),
        )

    @staticmethod
    def discount_cumsum(x, gamma=1.):
        out = np.zeros_like(x)
        out[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            out[t] = x[t] + gamma * out[t + 1]
        return out

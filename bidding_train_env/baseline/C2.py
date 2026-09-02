import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def getScore_nips(reward, cpa, cpa_constraint):
    bias = 0.2
    coefficient = cpa / cpa_constraint
    distance = abs(coefficient - 1.0)
    penalty = 1.0 / (1.0 + (distance > bias) * distance)
    return penalty * reward


class CrossSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config['n_embd'] % config['n_head'] == 0
        self.key = nn.Linear(config['n_embd'], config['n_embd'])
        self.query = nn.Linear(config['n_embd'], config['n_embd'])
        self.value = nn.Linear(config['n_embd'], config['n_embd'])
        self.attn_drop = nn.Dropout(config['attn_pdrop'])
        self.resid_drop = nn.Dropout(config['resid_pdrop'])
        self.register_buffer(
            "bias", torch.tril(torch.ones(config['n_ctx'], config['n_ctx'])).view(
                1, 1, config['n_ctx'], config['n_ctx']))
        self.register_buffer("masked_bias", torch.tensor(-1e4))
        self.proj = nn.Linear(config['n_embd'], config['n_embd'])
        self.n_head = config['n_head']

    def forward(self, source, target, mask):
        B, T, C = source.size()
        k = self.key(source).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(target).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(source).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        mask = (1.0 - mask.view(B, -1)[:, None, None, :]) * -10000.0
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = torch.where(self.bias[:, :, :T, :T].bool(), att, self.masked_bias.to(att.dtype))
        att = F.softmax(att + mask, dim=-1)
        att = self.attn_drop(att)
        output = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(output))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_x = nn.LayerNorm(config['n_embd'])
        self.ln_y = nn.LayerNorm(config['n_embd'])
        self.ln_z = nn.LayerNorm(config['n_embd'])
        self.ln_x2 = nn.LayerNorm(config['n_embd'])
        self.ln_y2 = nn.LayerNorm(config['n_embd'])
        self.ln_z2 = nn.LayerNorm(config['n_embd'])
        self.attn_xy = CrossSelfAttention(config)
        self.attn_yx = CrossSelfAttention(config)
        self.attn_xz = CrossSelfAttention(config)
        self.attn_zx = CrossSelfAttention(config)
        self.attn_yz = CrossSelfAttention(config)
        self.attn_zy = CrossSelfAttention(config)

        def mlp():
            return nn.Sequential(
                nn.Linear(config['n_embd'], config['n_inner']),
                nn.GELU(),
                nn.Linear(config['n_inner'], config['n_embd']),
                nn.Dropout(config['resid_pdrop']),
            )
        self.mlp_x = mlp()
        self.mlp_y = mlp()
        self.mlp_z = mlp()

    def forward(self, x, y, z, attn_mask):
        x_attn = (
            x
            + self.attn_xy(self.ln_y(y), self.ln_x(x), attn_mask)
            + self.attn_xz(self.ln_z(z), self.ln_x(x), attn_mask)
        )
        y_attn = (
            y
            + self.attn_yx(self.ln_x(x), self.ln_y(y), attn_mask)
            + self.attn_yz(self.ln_z(z), self.ln_y(y), attn_mask)
        )
        z_attn = (
            z
            + self.attn_zx(self.ln_x(x), self.ln_z(z), attn_mask)
            + self.attn_zy(self.ln_y(y), self.ln_z(z), attn_mask)
        )
        x = x_attn + self.mlp_x(self.ln_x2(x_attn))
        y = y_attn + self.mlp_y(self.ln_y2(y_attn))
        z = z_attn + self.mlp_z(self.ln_z2(z_attn))
        return x, y, z


class C2(nn.Module):
    def __init__(self, state_dim, act_dim, state_mean, state_std,
                 hidden_size=256, n_layer=6, n_head=4, n_inner=256, K=10,
                 max_ep_len=48, scale=40, target_return=40, device="cpu",
                 learning_rate=1e-5, rtg_loss_coef=0.1):
        super().__init__()
        self.device = device
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.state_mean = state_mean
        self.state_std = state_std
        self.max_length = K
        self.max_ep_len = max_ep_len
        self.scale = scale
        self.target_return = target_return
        self.hidden_size = hidden_size
        self.time_dim = 8
        self.warmup_steps = 10000
        self.weight_decay = 0.0001
        self.learning_rate = learning_rate
        self.rtg_loss_coef = rtg_loss_coef

        self.block_config = {
            "n_ctx": 1024,
            "n_embd": self.hidden_size,
            "n_layer": n_layer,
            "n_head": n_head,
            "n_inner": n_inner,
            "resid_pdrop": 0.1,
            "attn_pdrop": 0.1,
        }
        self.transformer = nn.ModuleList(
            [Block(self.block_config) for _ in range(self.block_config['n_layer'])]
        )

        self.embed_timestep = nn.Embedding(max_ep_len + 1, self.time_dim)
        self.embed_return = nn.Linear(1, self.hidden_size)
        self.embed_state = nn.Linear(state_dim, self.hidden_size)
        self.embed_action = nn.Linear(act_dim, self.hidden_size)
        self.trans_return = nn.Linear(self.time_dim + self.hidden_size, self.hidden_size)
        self.trans_state = nn.Linear(self.time_dim + self.hidden_size, self.hidden_size)
        self.trans_action = nn.Linear(self.time_dim + self.hidden_size, self.hidden_size)

        self.embed_ln = nn.LayerNorm(self.hidden_size)
        self.predict_action = nn.Sequential(nn.Linear(self.hidden_size, act_dim))
        self.predict_return = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 4), nn.GELU(),
            nn.Linear(self.hidden_size // 4, 16), nn.GELU(),
            nn.Linear(16, 1),
        )

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda steps: min((steps + 1) / self.warmup_steps, 1))

        self.init_eval()

    def forward(self, states, actions, returns_to_go, timesteps, attention_mask=None):
        B, T = states.shape[0], states.shape[1]
        if attention_mask is None:
            attention_mask = torch.ones((B, T), dtype=torch.long, device=states.device)

        se = self.embed_state(states)
        ae = self.embed_action(actions)
        re = self.embed_return(returns_to_go)
        te = self.embed_timestep(timesteps)

        x = self.trans_state(torch.cat((se, te), dim=-1))
        y = self.trans_action(torch.cat((ae, te), dim=-1))
        z = self.trans_return(torch.cat((re, te), dim=-1))

        for block in self.transformer:
            x, y, z = block(x, y, z, attention_mask)

        return self.predict_action(x), self.predict_return(y)

    @staticmethod
    def _shift_actions(actions):
        action_inputs = torch.zeros_like(actions)
        action_inputs[:, 1:] = actions[:, :-1]
        return action_inputs

    def get_action(self, states, actions, returns_to_go, timesteps):
        self.eval()
        K = self.max_length
        states = states.reshape(1, -1, self.state_dim)[:, -K:]
        actions = actions.reshape(1, -1, self.act_dim)[:, -K:]
        returns_to_go = returns_to_go.reshape(1, -1, 1)[:, -K:]
        timesteps = timesteps.reshape(1, -1)[:, -K:]

        T = states.shape[1]
        attention_mask = torch.cat([torch.zeros(K - T), torch.ones(T)]).to(
            dtype=torch.long, device=states.device).reshape(1, -1)

        def _pad(t, dims):
            pad = K - t.shape[1]
            if pad <= 0:
                return t
            zeros = torch.zeros((1, pad, *dims), device=t.device, dtype=t.dtype)
            return torch.cat([zeros, t], dim=1)

        states = _pad(states, (self.state_dim,)).float()
        actions = _pad(actions, (self.act_dim,)).float()
        returns_to_go = _pad(returns_to_go, (1,)).float()
        pad_t = K - timesteps.shape[1]
        if pad_t > 0:
            timesteps = torch.cat([torch.zeros((1, pad_t), dtype=torch.long, device=timesteps.device),
                                   timesteps], dim=1)

        mean = torch.tensor(self.state_mean, device=self.device, dtype=states.dtype)
        std = torch.tensor(self.state_std, device=self.device, dtype=states.dtype)
        states = (states - mean) / std
        action_inputs = self._shift_actions(actions)
        action_preds, _ = self.forward(
            states, action_inputs, returns_to_go, timesteps, attention_mask
        )
        return action_preds[0, -1]

    def step(self, states, actions, rtg, timesteps, attention_mask, penalty):
        states = states.to(self.device)
        actions = actions.to(self.device)
        rtg = rtg.to(self.device)
        timesteps = timesteps.to(self.device)
        penalty = penalty.to(self.device)
        attention_mask = attention_mask.to(self.device).to(actions.dtype)

        action_target = actions.clone()
        rtg_target = rtg[:, 1:].clone()
        action_inputs = self._shift_actions(actions)

        action_preds, rtg_preds = self.forward(
            states, action_inputs, rtg[:, :-1], timesteps, attention_mask)

        mask = attention_mask.reshape(-1) > 0
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[mask]
        action_target = action_target.reshape(-1, act_dim)[mask]
        rtg_preds = rtg_preds.reshape(-1, 1)[mask]
        rtg_target = rtg_target.reshape(-1, 1)[mask]
        penalty = penalty.reshape(-1, 1)[mask]

        action_loss = torch.mean(penalty * (action_preds - action_target) ** 2)
        rtg_loss = (
            torch.mean(penalty * (rtg_preds - rtg_target) ** 2)
            * self.rtg_loss_coef
        )
        loss = action_loss + rtg_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), .25)
        self.optimizer.step()

        return loss.item(), action_loss.item(), rtg_loss.item()

    def take_actions(self, state, actual_excuted_action, target_return=None,
                     pre_reward=None, pre_cost=None, cpa_constrain=None):
        if self.eval_states is None:
            self.eval_states = torch.from_numpy(state).reshape(1, self.state_dim).to(self.device)
            ep_return = torch.tensor(target_return, device=self.device) if target_return is not None \
                else torch.tensor(self.target_return, device=self.device)
            ep_return = ep_return * float(os.environ.get('EVAL_TARGET_MULT', '1.0')) / self.scale
            self.eval_target_return = ep_return.clone().detach().reshape(1, 1).float()
        else:
            cur_state = torch.from_numpy(state).reshape(1, self.state_dim).to(self.device)
            self.eval_states = torch.cat([self.eval_states, cur_state], dim=0)

            adj_reward = pre_reward
            if pre_reward != 0:
                adj_reward = getScore_nips(pre_reward, pre_cost / pre_reward, cpa_constrain)
            pred_return = self.eval_target_return[0, -1] - (adj_reward / self.scale)
            self.eval_target_return = torch.cat(
                [self.eval_target_return, pred_return.reshape(1, 1)], dim=1
            )

            self.eval_timesteps = torch.cat([
                self.eval_timesteps,
                torch.ones((1, 1), dtype=torch.long, device=self.device) * self.eval_timesteps[:, -1] + 1
            ], dim=1)

        if actual_excuted_action is None:
            self.eval_actions = torch.cat(
                [self.eval_actions, torch.zeros(1, self.act_dim, device=self.device)], dim=0)
        else:
            self.eval_actions[-1] = torch.from_numpy(actual_excuted_action).reshape(1, self.act_dim).to(self.device)
            self.eval_actions = torch.cat(
                [self.eval_actions, torch.zeros(1, self.act_dim, device=self.device)], dim=0)

        action = self.get_action(
            self.eval_states.float(),
            self.eval_actions.float(),
            self.eval_target_return.float(),
            self.eval_timesteps.long(),
        )
        self.eval_actions[-1] = action
        return action.detach().cpu().numpy()

    def init_eval(self):
        self.eval_states = None
        self.eval_actions = torch.zeros((0, self.act_dim), dtype=torch.float32, device=self.device)
        self.eval_target_return = None
        self.eval_timesteps = torch.ones((1, 1), dtype=torch.long, device=self.device)

    def save_net(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_path, 'dt.pt'))

    def load_net(self, load_path, device='cpu'):
        state_dict = torch.load(load_path, map_location=device, weights_only=True)
        self.load_state_dict(state_dict, strict=True)

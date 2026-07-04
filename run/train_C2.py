import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging

import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from bidding_train_env.baseline.C2 import C2
from bidding_train_env.baseline.utils import EpisodeReplayBuffer
from bidding_train_env.common.utils import save_normalize_dict

os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

DEFAULT_DATA_PATH = 'data/traffic/training_data_all-rlData.csv'


def select_hq_trajectories(df, cpa_low=0.85, cpa_high=1.05):
    """Filter high-quality trajectories: episode-end CPA ratio within [cpa_low, cpa_high]
    and at least one conversion. Returns full-row DataFrame for matching (period, advertiser)."""
    traj_end = df[df['done'] == 1].copy()
    traj_end = traj_end[traj_end['realAllConversion'] > 0]
    traj_end['cpa_ratio'] = (traj_end['realAllCost']
                             / (traj_end['realAllConversion'] + 1e-8)
                             / traj_end['CPAConstraint'])
    hq = traj_end[(traj_end['cpa_ratio'] >= cpa_low) & (traj_end['cpa_ratio'] <= cpa_high)]
    hq_keys = set(zip(hq['deliveryPeriodIndex'], hq['advertiserNumber']))
    keys_series = list(zip(df['deliveryPeriodIndex'], df['advertiserNumber']))
    mask = [k in hq_keys for k in keys_series]
    return df[mask], len(hq_keys)


def run_epochs(model, replay_buffer, num_steps, batch_size, logger, phase,
               save_range=None, ckpt_root=None, state_mean=None, state_std=None):
    sampler = WeightedRandomSampler(replay_buffer.p_sample, num_samples=num_steps * batch_size, replacement=True)
    dataloader = DataLoader(replay_buffer, sampler=sampler, batch_size=batch_size)
    model.train()
    for i, (states, actions, rtg, timesteps, attention_mask, penalty) in enumerate(dataloader):
        loss, action_loss, rtg_loss = model.step(
            states=states, actions=actions, rtg=rtg,
            timesteps=timesteps, attention_mask=attention_mask, penalty=penalty)
        if i % 100 == 0:
            logger.info(f"[{phase}] step {i} loss={loss:.4f} action={action_loss:.4f} rtg={rtg_loss:.4f}")
        if save_range and save_range[0] <= i <= save_range[1] and i % 10 == 0:
            ckpt = f"{ckpt_root}/{i}"
            model.save_net(ckpt)
            save_normalize_dict({"state_mean": state_mean, "state_std": state_std}, ckpt)
        model.scheduler.step()


def train(model_name='C2', data_path=None,
          pretrain_steps=2000, finetune_steps=10001, batch_size=128,
          cpa_low=0.85, cpa_high=1.05,
          ckpt_start=2000, ckpt_end=10000, device=None):
    if data_path is None:
        data_path = DEFAULT_DATA_PATH

    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    ckpt_root = f"saved_model/{model_name}"
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info(f'loading full data: {data_path}')
    full_df = pd.read_csv(data_path)
    hq_df, n_hq = select_hq_trajectories(full_df, cpa_low, cpa_high)
    logger.info(f'HQ trajectories: {n_hq} (cpa_ratio in [{cpa_low}, {cpa_high}])')

    # Phase 1: pretrain on HQ trajectories
    logger.info(f"=== Phase 1: pretrain {pretrain_steps} steps on HQ subset ===")
    rb_hq = EpisodeReplayBuffer(state_dim=16, act_dim=1, df=hq_df, device=device)
    model = C2(state_dim=16, act_dim=1,
               state_mean=rb_hq.state_mean, state_std=rb_hq.state_std, device=device)
    model.to(device)
    logger.info(f"Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    run_epochs(model, rb_hq, pretrain_steps, batch_size, logger, phase='pre')

    # Phase 2: finetune on full data (reset optimizer/scheduler + use full-data stats)
    logger.info(f"=== Phase 2: finetune {finetune_steps} steps on full data ===")
    rb_full = EpisodeReplayBuffer(state_dim=16, act_dim=1, df=full_df, device=device)
    save_normalize_dict(
        {"state_mean": rb_full.state_mean, "state_std": rb_full.state_std},
        f"saved_model/{model_name}")
    model.optimizer = torch.optim.AdamW(model.parameters(), lr=model.learning_rate,
                                        weight_decay=model.weight_decay)
    model.scheduler = torch.optim.lr_scheduler.LambdaLR(
        model.optimizer, lambda steps: min((steps + 1) / model.warmup_steps, 1))
    run_epochs(model, rb_full, finetune_steps, batch_size, logger, phase='ft',
               save_range=(ckpt_start, ckpt_end), ckpt_root=ckpt_root,
               state_mean=rb_full.state_mean, state_std=rb_full.state_std)

    model.save_net(f"saved_model/{model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='C2')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--pretrain_steps', type=int, default=2000)
    parser.add_argument('--finetune_steps', type=int, default=10001)
    parser.add_argument('--cpa_low', type=float, default=0.85)
    parser.add_argument('--cpa_high', type=float, default=1.05)
    parser.add_argument('--ckpt_start', type=int, default=2000)
    parser.add_argument('--ckpt_end', type=int, default=10001)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()
    train(model_name=args.model_name, data_path=args.data_path,
          pretrain_steps=args.pretrain_steps, finetune_steps=args.finetune_steps,
          cpa_low=args.cpa_low, cpa_high=args.cpa_high,
          ckpt_start=args.ckpt_start, ckpt_end=args.ckpt_end, device=args.device)

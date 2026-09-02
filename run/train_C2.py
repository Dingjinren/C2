import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from bidding_train_env.baseline.C2 import C2
from bidding_train_env.baseline.utils import EpisodeReplayBuffer
from bidding_train_env.common.utils import save_normalize_dict

os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

DEFAULT_DATA_PATH = (
    "data/traffic/training_data_rlData_folder/training_data_all-rlData.csv"
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(model_name="C2", data_path=DEFAULT_DATA_PATH, output_dir=None,
          num_steps=10000, batch_size=128, seed=20260822,
          hidden_size=256, n_layer=6, n_head=4, n_inner=256, K=10,
          learning_rate=1e-5, rtg_loss_coef=0.1,
          ckpt_start=3500, ckpt_end=10000,
          ckpt_step=100, device=None):
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    if output_dir is None:
        output_dir = os.path.join("saved_model", model_name)
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if ckpt_step <= 0:
        raise ValueError("ckpt_step must be positive")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(seed)
    logger.info(f"loading full data: {data_path}")
    replay_buffer = EpisodeReplayBuffer(
        state_dim=16, act_dim=1, data_path=data_path, K=K, device=device,
        threshold=1.15, beta=4.0, sampler_lambda=1.0,
        reward_tolerance=0.2, sampler_tolerance=0.2,
    )
    model = C2(
        state_dim=16, act_dim=1,
        state_mean=replay_buffer.state_mean,
        state_std=replay_buffer.state_std,
        hidden_size=hidden_size, n_layer=n_layer, n_head=n_head,
        n_inner=n_inner, K=K, learning_rate=learning_rate,
        rtg_loss_coef=rtg_loss_coef,
        device=device,
    ).to(device)
    logger.info(f"Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = WeightedRandomSampler(
        replay_buffer.p_sample, num_samples=num_steps * batch_size,
        replacement=True, generator=generator,
    )
    dataloader = DataLoader(
        replay_buffer, sampler=sampler, batch_size=batch_size, num_workers=0
    )
    normalize_dict = {
        "state_mean": replay_buffer.state_mean,
        "state_std": replay_buffer.state_std,
    }

    model.train()
    for step, (states, actions, rtg, timesteps, attention_mask, penalty) in enumerate(
            dataloader, start=1):
        loss, action_loss, rtg_loss = model.step(
            states=states, actions=actions, rtg=rtg, timesteps=timesteps,
            attention_mask=attention_mask, penalty=penalty,
        )
        model.scheduler.step()

        if step == 1 or step % 100 == 0:
            logger.info(
                f"step {step} loss={loss:.4f} "
                f"action={action_loss:.4f} rtg={rtg_loss:.4f}"
            )
        if (ckpt_start <= step <= ckpt_end
                and (step - ckpt_start) % ckpt_step == 0):
            checkpoint_dir = os.path.join(output_dir, str(step))
            model.save_net(checkpoint_dir)
            save_normalize_dict(normalize_dict, checkpoint_dir)

        if step >= num_steps:
            break

    model.save_net(output_dir)
    save_normalize_dict(normalize_dict, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="C2")
    parser.add_argument("--data_path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--num_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_inner", type=int, default=256)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--rtg_loss_coef", type=float, default=0.1)
    parser.add_argument("--ckpt_start", type=int, default=3500)
    parser.add_argument("--ckpt_end", type=int, default=10000)
    parser.add_argument("--ckpt_step", type=int, default=100)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    train(**vars(args))

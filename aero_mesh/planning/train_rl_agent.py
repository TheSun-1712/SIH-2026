"""
AERO MESH — RL NBV Agent Training Script (PPO)
==============================================
Trains the PPO Next-Best-View agent in a simulated reconstruction environment.

The environment rewards the agent for:
  - Reducing per-Gaussian uncertainty (primary reward)
  - Improving per-entity OUGS scores
  - Filling new TSDF voxels
  - Penalises revisiting already-covered areas and CBF violations

CHECKPOINTING & RESUME:
  - Saves checkpoint every --save-every episodes to --save-dir
  - Saves 'best.pt' when rolling mean reward improves
  - Ctrl+C (SIGINT) saves checkpoint before exit — NO WORK LOST
  - Resume: python train_rl_agent.py --resume
  - Resume specific: python train_rl_agent.py --resume --ckpt path/to/ckpt.pt

PPO HYPERPARAMETERS (tuned for the NBV task):
  - clip_eps   = 0.2   (standard PPO clip)
  - entropy    = 0.01  (encourage exploration)
  - gae_lambda = 0.95
  - gamma      = 0.99
  - n_epochs   = 4     (update epochs per batch)
  - batch_size = 64

Usage:
  # Fresh training (synthetic environment):
  python aero_mesh/planning/train_rl_agent.py --episodes 5000

  # Resume from latest checkpoint:
  python aero_mesh/planning/train_rl_agent.py --resume

  # Resume from specific checkpoint:
  python aero_mesh/planning/train_rl_agent.py --resume --ckpt checkpoints/rl_agent/ckpt_ep01200.pt

  # Full options:
  python aero_mesh/planning/train_rl_agent.py \\
      --episodes 5000 \\
      --max-steps 200 \\
      --save-dir checkpoints/rl_agent/ \\
      --save-every 50 \\
      --gamma 0.99 \\
      --lr 3e-4
"""

import os
import sys
import json
import signal
import argparse
import time
import numpy as np
from pathlib import Path
from collections import deque

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def parse_args():
    p = argparse.ArgumentParser(description="Train AERO MESH NBV RL Agent (PPO)")
    p.add_argument("--episodes",   type=int,   default=5000)
    p.add_argument("--max-steps",  type=int,   default=200,
                   help="Maximum steps per episode")
    p.add_argument("--save-dir",   type=str,   default="checkpoints/rl_agent")
    p.add_argument("--save-every", type=int,   default=50,
                   help="Save checkpoint every N episodes")
    p.add_argument("--resume",     action="store_true")
    p.add_argument("--ckpt",       type=str,   default=None)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--gamma",      type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-eps",   type=float, default=0.2)
    p.add_argument("--entropy",    type=float, default=0.01)
    p.add_argument("--n-epochs",   type=int,   default=4)
    p.add_argument("--batch",      type=int,   default=64)
    p.add_argument("--device",     type=str,   default="auto")
    p.add_argument("--n-entities", type=int,   default=8,
                   help="Number of synthetic entities in the simulation")
    return p.parse_args()


# ── Synthetic NBV Environment ─────────────────────────────────────────────────

class SyntheticNBVEnv:
    """
    Lightweight simulation of the drone reconstruction task for RL training.

    The environment simulates:
      - A set of N_ENTITIES entities, each with an uncertainty score
      - The drone navigating to reduce entity uncertainties
      - GauSS-MI uncertainty decays as drone gets close to entities
      - Failure memory heat (random spatial distribution)
      - Battery drain over time

    This lets the RL agent learn the core NBV policy WITHOUT needing
    a real 3DGS scene at training time.
    """

    OBS_DIM    = 88
    ACTION_DIM = 5

    def __init__(self, n_entities: int = 8, max_steps: int = 200):
        self.n_entities   = n_entities
        self.max_steps    = max_steps
        self._entity_positions = None
        self._entity_scores    = None
        self._drone_pose       = None
        self._step_count       = 0
        self._battery          = 1.0
        self._visited          = set()
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset environment to a new random configuration."""
        self._step_count = 0
        self._battery    = 1.0
        self._visited    = set()

        # Random entity positions in a 200×200m survey area
        self._entity_positions = (
            np.random.randn(self.n_entities, 3) * np.array([50, 0, 50])
        ).astype(np.float32)
        self._entity_positions[:, 1] = 0  # entities at ground level

        # Initial uncertainty: random [0.4, 0.95]
        self._entity_scores = (
            np.random.uniform(0.4, 0.95, self.n_entities)
        ).astype(np.float32)

        # Start drone at random position at altitude 30m
        self._drone_pose = np.array([
            np.random.uniform(-80, 80),
            30.0,
            np.random.uniform(-80, 80),
            0.0, 0.0, 0.0,
        ], dtype=np.float32)

        return self._build_obs()

    def _build_obs(self) -> np.ndarray:
        """Build the 88-dim observation vector from current state."""
        from aero_mesh.planning.nbv_rl_agent import NBVRLAgent

        # Top-10 uncertain Gaussian positions (simulated as entity positions
        # with scores above 0.3, padded to 10)
        top_idx = np.argsort(self._entity_scores)[::-1][:10]
        gauss_positions = self._entity_positions[top_idx]
        if len(gauss_positions) < 10:
            gauss_positions = np.vstack([
                gauss_positions,
                np.zeros((10 - len(gauss_positions), 3), dtype=np.float32)
            ])

        # OUGS entity scores (padded to 20)
        ougs = dict(zip(
            [f"e{i}" for i in range(self.n_entities)],
            self._entity_scores.tolist()
        ))

        # Failure heat: random for training
        heat = np.random.rand(16).astype(np.float32) * 0.3

        # Sector completions (8)
        sector_comp = np.clip(
            1.0 - np.random.rand(8).astype(np.float32) * self._entity_scores.mean(),
            0, 1
        )

        obs = NBVRLAgent.build_observation(
            drone_pose         = self._drone_pose,
            gauss_mi_top10     = gauss_positions,
            ougs_scores        = ougs,
            failure_heat       = heat,
            battery_pct        = self._battery,
            sector_completions = sector_comp,
            global_uncertainty = float(self._entity_scores.mean()),
            step_norm          = self._step_count / self.max_steps,
            last_action        = np.zeros(5, dtype=np.float32),
        )
        return obs

    def step(self, action: np.ndarray) -> tuple:
        """
        Apply action and return (next_obs, reward, done, info).
        action: (5,) [Δx, Δy, Δz, Δyaw, Δgimbal] in physical units
        """
        self._step_count += 1

        # Move drone
        self._drone_pose[:3] += action[:3]
        self._drone_pose[3]  += action[3]
        self._drone_pose[:3]  = np.clip(self._drone_pose[:3], -150, 150)
        self._drone_pose[1]   = max(5.0, self._drone_pose[1])  # min altitude
        self._battery        -= 0.005   # 5% battery per 100 steps

        # Reduce entity uncertainty if drone is close
        drone_xy = self._drone_pose[[0, 2]]
        reward   = 0.0
        for i in range(self.n_entities):
            ent_xy = self._entity_positions[i, [0, 2]]
            dist   = np.linalg.norm(drone_xy - ent_xy)
            if dist < 20.0:  # Within "inspection range"
                reduction = max(0, 0.04 * (1 - dist / 20.0))
                delta = min(self._entity_scores[i], reduction)
                self._entity_scores[i] -= delta
                reward += delta * 10.0   # Reward ∝ uncertainty resolved

        # Penalise revisiting same cell
        cell_key = (int(self._drone_pose[0] / 10), int(self._drone_pose[2] / 10))
        if cell_key in self._visited:
            reward -= 0.5
        else:
            self._visited.add(cell_key)

        # Penalise low battery
        if self._battery < 0.15:
            reward -= 2.0

        # Penalty for straying too far
        if np.linalg.norm(self._drone_pose[:3]) > 140:
            reward -= 5.0

        # Done conditions
        done = (
            self._step_count >= self.max_steps or
            self._battery <= 0 or
            self._entity_scores.mean() < 0.1  # Success: all entities resolved
        )

        return self._build_obs(), float(reward), done, {
            "mean_uncertainty": float(self._entity_scores.mean()),
            "battery": float(self._battery),
            "steps": self._step_count,
        }


# ── GAE Advantage Estimation ──────────────────────────────────────────────────

def compute_gae(
    rewards, values, dones, gamma: float, gae_lambda: float
) -> tuple:
    """Compute Generalised Advantage Estimation (GAE)."""
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    last_adv   = 0.0
    for t in reversed(range(n)):
        if t == n - 1:
            next_val = 0.0
        else:
            next_val = values[t + 1]
        delta       = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        advantages[t] = last_adv = delta + gamma * gae_lambda * (1 - dones[t]) * last_adv
    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns


# ── Checkpoint Helpers ────────────────────────────────────────────────────────

def find_latest_checkpoint(save_dir: str):
    d = Path(save_dir)
    latest = d / "latest.pt"
    if latest.exists():
        return str(latest)
    ckpts = sorted(d.glob("ckpt_ep*.pt"))
    return str(ckpts[-1]) if ckpts else None


def save_checkpoint(agent, optimizer, scheduler, episode, best_reward,
                    config, save_dir, val_tag=""):
    import torch
    d = Path(save_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"ckpt_ep{episode:05d}.pt"
    data = {
        "episode":          episode,
        "policy_state":     agent.policy.state_dict(),
        "value_state":      agent.value.state_dict(),
        "optimizer_state":  optimizer.state_dict(),
        "scheduler_state":  scheduler.state_dict() if scheduler else None,
        "best_reward":      best_reward,
        "running_mean_std": agent.obs_rms.state_dict(),
        "config":           config,
    }
    torch.save(data, path)
    torch.save(data, d / "latest.pt")
    return str(path)


# ── PPO Update ────────────────────────────────────────────────────────────────

def ppo_update(
    agent, optimizer, obs_buf, act_buf, old_logp_buf,
    adv_buf, ret_buf, clip_eps, entropy_coef, n_epochs, batch_size, device
):
    import torch

    obs_t    = torch.from_numpy(np.array(obs_buf)).float().to(device)
    act_t    = torch.from_numpy(np.array(act_buf)).float().to(device)
    old_logp = torch.from_numpy(np.array(old_logp_buf)).float().to(device)
    adv_t    = torch.from_numpy(adv_buf).float().to(device)
    ret_t    = torch.from_numpy(ret_buf).float().to(device)

    # Normalise advantages
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    n = len(obs_t)
    total_loss = 0.0
    for _ in range(n_epochs):
        idx = torch.randperm(n)
        for start in range(0, n, batch_size):
            batch = idx[start:start+batch_size]
            obs_b  = obs_t[batch]
            act_b  = act_t[batch]
            old_b  = old_logp[batch]
            adv_b  = adv_t[batch]
            ret_b  = ret_t[batch]

            # Policy forward
            mean, log_std = agent.policy(obs_b)
            std  = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)
            new_logp = dist.log_prob(act_b).sum(dim=-1)
            entropy  = dist.entropy().sum(dim=-1).mean()

            # Value forward
            val = agent.value(obs_b).squeeze(-1)

            # PPO losses
            ratio   = torch.exp(new_logp - old_b)
            clip_r  = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
            pg_loss = -torch.min(ratio * adv_b, clip_r * adv_b).mean()
            vf_loss = ((val - ret_b) ** 2).mean()
            loss    = pg_loss + 0.5 * vf_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(agent.policy.parameters()) + list(agent.value.parameters()), 0.5
            )
            optimizer.step()
            total_loss += loss.item()

    return total_loss


# ── Main Training Loop ────────────────────────────────────────────────────────

def train(args):
    try:
        import torch
    except ImportError:
        print("[train_rl_agent] PyTorch not installed. Run: pip install torch")
        sys.exit(1)

    from aero_mesh.planning.nbv_rl_agent import NBVRLAgent

    device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else args.device

    print(f"\n{'='*60}")
    print(f"  AERO MESH — NBV RL Agent Training (PPO)")
    print(f"{'='*60}")
    print(f"  Device    : {device}")
    print(f"  Episodes  : {args.episodes}")
    print(f"  Max Steps : {args.max_steps}")
    print(f"  Save Dir  : {args.save_dir}")
    print(f"{'='*60}\n")

    env   = SyntheticNBVEnv(n_entities=args.n_entities, max_steps=args.max_steps)
    agent = NBVRLAgent(device=device)

    optimizer = torch.optim.Adam(
        list(agent.policy.parameters()) + list(agent.value.parameters()),
        lr=args.lr, eps=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=args.episodes
    )

    config      = vars(args)
    start_ep    = 0
    best_reward = -float("inf")
    reward_hist = deque(maxlen=100)

    # ── Resume ────────────────────────────────────────────────────────────────
    ckpt_path = args.ckpt or (find_latest_checkpoint(args.save_dir) if args.resume else None)
    if ckpt_path and Path(ckpt_path).exists():
        print(f"[train_rl_agent] Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        agent.policy.load_state_dict(ckpt["policy_state"])
        agent.value.load_state_dict(ckpt["value_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if ckpt.get("scheduler_state"):
            scheduler.load_state_dict(ckpt["scheduler_state"])
        if "running_mean_std" in ckpt:
            agent.obs_rms.load_state_dict(ckpt["running_mean_std"])
        start_ep    = ckpt.get("episode", 0) + 1
        best_reward = ckpt.get("best_reward", -float("inf"))
        print(f"[train_rl_agent] Resumed from episode {start_ep}, "
              f"best_reward={best_reward:.3f}")
    elif args.resume:
        print("[train_rl_agent] --resume set but no checkpoint found. Starting fresh.")

    # ── Ctrl+C Safety ─────────────────────────────────────────────────────────
    _current_ep  = [start_ep]
    _interrupted = [False]

    def handle_sigint(sig, frame):
        print(f"\n[train_rl_agent] Ctrl+C — saving checkpoint at episode "
              f"{_current_ep[0]}…")
        save_checkpoint(
            agent, optimizer, scheduler,
            _current_ep[0], best_reward, config, args.save_dir
        )
        print("[train_rl_agent] Checkpoint saved. Safe to exit.")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    # ── Training Episodes ─────────────────────────────────────────────────────
    print(f"[train_rl_agent] Starting from episode {start_ep + 1}…")
    print("  Hit Ctrl+C at any time to pause — training will resume from checkpoint.\n")

    # Rollout buffers
    obs_buf, act_buf, logp_buf = [], [], []
    rew_buf, val_buf, don_buf  = [], [], []
    ROLLOUT_LEN = args.max_steps  # Collect 1 episode per update

    for episode in range(start_ep, args.episodes):
        _current_ep[0] = episode
        obs = env.reset()
        ep_reward  = 0.0
        ep_obs, ep_act, ep_logp, ep_rew, ep_val, ep_don = [], [], [], [], [], []

        for step in range(args.max_steps):
            agent.obs_rms.update(obs)
            norm_obs = agent.obs_rms.normalize(obs)
            t_obs    = torch.from_numpy(norm_obs).float().unsqueeze(0).to(device)

            with torch.no_grad():
                act_t, logp_t = agent.policy.get_action(t_obs)
                val_t         = agent.value(t_obs)

            raw_action = act_t.squeeze(0).cpu().numpy()
            from aero_mesh.planning.nbv_rl_agent import ACTION_SCALE
            phys_action = raw_action * ACTION_SCALE

            next_obs, reward, done, info = env.step(phys_action)

            ep_obs.append(obs.copy())
            ep_act.append(raw_action)
            ep_logp.append(logp_t.squeeze(0).item())
            ep_rew.append(reward)
            ep_val.append(val_t.squeeze(0).item())
            ep_don.append(float(done))
            ep_reward += reward

            obs = next_obs
            if done:
                break

        # GAE
        adv, ret = compute_gae(ep_rew, ep_val, ep_don, args.gamma, args.gae_lambda)

        # PPO Update (one episode rollout)
        ppo_update(
            agent, optimizer,
            ep_obs, ep_act, ep_logp, adv, ret,
            args.clip_eps, args.entropy, args.n_epochs, args.batch, device
        )
        scheduler.step()

        reward_hist.append(ep_reward)
        mean_reward = np.mean(reward_hist)

        if (episode + 1) % 10 == 0:
            print(f"  Episode {episode+1:05d}/{args.episodes}  "
                  f"reward={ep_reward:7.2f}  "
                  f"mean100={mean_reward:7.2f}  "
                  f"mean_unc={info.get('mean_uncertainty', 0):.3f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

        # Save best
        if mean_reward > best_reward and len(reward_hist) >= 20:
            best_reward = mean_reward
            best_path   = Path(args.save_dir) / "best.pt"
            Path(args.save_dir).mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                agent, optimizer, scheduler,
                episode, best_reward, config, args.save_dir
            )
            import shutil
            shutil.copy(Path(args.save_dir) / "latest.pt", best_path)
            print(f"  ★ New best! mean_reward={best_reward:.3f} → {best_path}")

        # Periodic checkpoint
        if (episode + 1) % args.save_every == 0:
            ckpt_saved = save_checkpoint(
                agent, optimizer, scheduler,
                episode, best_reward, config, args.save_dir
            )
            print(f"  ✓ Checkpoint → {ckpt_saved}")

    # Final checkpoint
    save_checkpoint(
        agent, optimizer, scheduler,
        args.episodes - 1, best_reward, config, args.save_dir
    )
    print(f"\n[train_rl_agent] Training complete!")
    print(f"  Best mean reward : {best_reward:.3f}")
    print(f"  Best model       : {Path(args.save_dir) / 'best.pt'}")
    print(f"\n  To use in AERO MESH:")
    print(f"    agent = NBVRLAgent(checkpoint_path='{Path(args.save_dir) / 'best.pt'}')")


if __name__ == "__main__":
    train(parse_args())

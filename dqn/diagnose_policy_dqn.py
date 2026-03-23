"""Diagnose a DQN macro-action model.

- Loads newest ./models/*.zip by mtime
- Runs a short rollout
- Prints action distribution (rot,x), invalid action rate, lines cleared

Run:
  python diagnose_policy_dqn.py
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN

# Ensure env registration
import tetris_gymnasium.envs  # noqa: F401

from tetris_macro_env import TetrisPlacementEnv


def _find_latest_model(models_dir: str) -> str:
    models = sorted(
        glob.glob(os.path.join(models_dir, "*.zip")),
        key=os.path.getmtime,
        reverse=True,
    )
    if not models:
        raise SystemExit(f"No {models_dir}/*.zip found")
    return models[0]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional path to a specific .zip checkpoint (overrides --models-dir).",
    )
    parser.add_argument(
        "--obs",
        type=str,
        default="features",
        choices=["features", "board"],
        help="Observation mode used during training for this checkpoint.",
    )
    parser.add_argument(
        "--hold-actions",
        action="store_true",
        help="Enable hold/swap as part of the macro-action space (must match the checkpoint).",
    )
    parser.add_argument(
        "--next-n",
        type=int,
        default=1,
        help="How many upcoming pieces from the queue to include in observation (0 disables). Must match the checkpoint.",
    )
    parser.add_argument(
        "--reward-profile",
        type=str,
        default="tetris",
        choices=["tetris", "survival", "balanced"],
        help="Reward shaping profile (only affects reported reward).",
    )
    parser.add_argument("--n-steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a single-line summary (good for checkpoint sweeps).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many top actions to include in the summary/output.",
    )
    args = parser.parse_args()

    model_path = str(args.model_path) if args.model_path else _find_latest_model(str(args.models_dir))
    print("Using model:", model_path)

    base_env = gym.make("tetris_gymnasium/Tetris", render_mode=None, gravity=False)
    env = TetrisPlacementEnv(
        base_env,
        include_hold=bool(args.hold_actions),
        next_n=int(args.next_n),
        reward_profile=str(args.reward_profile),
        observation_mode=args.obs,
    )

    model = DQN.load(model_path, env=env)

    seed_base = int(args.seed)
    obs, _info = env.reset(seed=seed_base)

    counts = np.zeros(env.action_space.n, dtype=np.int64)
    rot_counts = np.zeros(4, dtype=np.int64)
    x_counts = np.zeros(env._x_count, dtype=np.int64)  # type: ignore[attr-defined]
    hold_counts = np.zeros(2, dtype=np.int64)

    invalid = 0
    total_lines = 0
    total_reward = 0.0

    episodes = 0
    ep_lines = 0
    ep_steps = 0
    best_ep_lines = 0
    best_ep_steps = 0

    n_steps = int(args.n_steps)
    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        if isinstance(action, np.ndarray):
            action = action.item()
        action = int(action)
        counts[action] += 1

        decoded = env.decode_action(action)
        rot_counts[decoded.rot_cw % 4] += 1
        x_counts[decoded.x] += 1
        hold_counts[int(decoded.hold)] += 1

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        lines = int(info.get("lines_cleared", 0) or 0)
        total_lines += lines
        ep_lines += lines
        ep_steps += 1

        if info.get("invalid_macro_action"):
            invalid += 1

        if terminated or truncated:
            episodes += 1
            if ep_lines > best_ep_lines:
                best_ep_lines = ep_lines
            if ep_steps > best_ep_steps:
                best_ep_steps = ep_steps
            ep_lines = 0
            ep_steps = 0
            obs, _info = env.reset(seed=seed_base + episodes)

    avg_reward_per_step = total_reward / max(1, n_steps)
    invalid_rate = invalid / max(1, n_steps)
    hold_rate = float(hold_counts[1]) / float(max(1, n_steps))
    lines_per_episode = float(total_lines) / float(max(1, episodes))
    steps_per_episode = float(n_steps) / float(max(1, episodes))

    top_k = max(int(args.top_k), 1)
    top_actions = np.argsort(-counts)[:top_k]

    if args.summary:
        # Single-line summary for easy parsing.
        # Try to extract the checkpoint step from the filename.
        step_str = "?"
        base = os.path.basename(model_path)
        try:
            step_str = base.split("_")[-2]
        except Exception:
            pass

        tops = []
        for a in top_actions[: min(3, len(top_actions))]:
            if counts[a] == 0:
                continue
            d = env.decode_action(int(a))
            share = float(counts[a]) / float(max(1, n_steps))
            tops.append(f"({d.hold},{d.rot_cw},{d.x},{int(counts[a])},{share:.3f})")

        print(
            " ".join(
                [
                    f"step={step_str}",
                    f"n_steps={n_steps}",
                    f"episodes={episodes}",
                    f"lines_total={total_lines}",
                    f"lines_per_ep={lines_per_episode:.2f}",
                    f"steps_per_ep={steps_per_episode:.1f}",
                    f"hold_rate={hold_rate:.3f}",
                    f"invalid_rate={invalid_rate:.4f}",
                    f"avg_rew_step={avg_reward_per_step:.3f}",
                    f"best_ep_lines={best_ep_lines}",
                    f"best_ep_steps={best_ep_steps}",
                    f"top3={'/'.join(tops)}",
                ]
            )
        )
        env.close()
        return

    print("n_steps =", n_steps)
    print("episodes_completed =", episodes)
    print("avg_reward_per_step =", avg_reward_per_step)
    print("lines_cleared_total =", total_lines)
    print("lines_per_episode_avg =", lines_per_episode)
    print("steps_per_episode_avg =", steps_per_episode)
    print("best_episode_lines =", best_ep_lines)
    print("best_episode_steps =", best_ep_steps)
    print("invalid_macro_action_rate =", invalid_rate)
    print("hold_counts =", hold_counts.tolist())
    print("hold_rate =", hold_rate)

    print("Top actions (action_id: count -> (hold,rot,x)):")
    for a in top_actions:
        if counts[a] == 0:
            continue
        d = env.decode_action(int(a))
        print(f"  {int(a)}: {int(counts[a])} -> hold={d.hold} rot={d.rot_cw} x={d.x}")

    print("rot_counts =", rot_counts.tolist())
    print("x_counts (first 15) =", x_counts[:15].tolist())

    env.close()


if __name__ == "__main__":
    main()

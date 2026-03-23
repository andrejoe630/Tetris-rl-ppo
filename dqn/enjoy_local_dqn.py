"""Play a trained DQN (macro-action) agent locally with human rendering.

Usage:
  python enjoy_local_dqn.py
  python enjoy_local_dqn.py --model-path ./models/dqn_tetris_macro_100000_steps.zip
  python enjoy_local_dqn.py --episodes 5 --sleep 0.05
"""

from __future__ import annotations

import argparse
import sys
import time
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


def find_latest_model(models_dir: Path) -> Path:
    models_dir = Path(models_dir)
    zips = [p for p in models_dir.glob("*.zip") if p.is_file()]
    if not zips:
        raise FileNotFoundError(f"No .zip models found in: {models_dir.resolve()}")
    zips.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return zips[0]


def _pump_render_events() -> None:
    try:
        import pygame

        pygame.event.pump()
    except Exception:
        pass

    try:
        import cv2

        cv2.waitKey(1)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="tetris_gymnasium/Tetris")
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument(
        "--obs",
        type=str,
        default="features",
        choices=["features", "board"],
        help="Observation mode used during training for this checkpoint.",
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
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--episode-pause", type=float, default=0.5)
    parser.add_argument(
        "--hold-actions",
        action="store_true",
        help="Enable hold/swap as part of the macro-action space (must match the checkpoint).",
    )
    parser.add_argument("--hold", action="store_true", help="Wait for Enter before closing.")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    model_path = Path(args.model_path) if args.model_path else find_latest_model(models_dir)

    print(f"Loading model: {model_path}")

    base_env = gym.make(args.env_id, render_mode="human", gravity=False)
    env = TetrisPlacementEnv(
        base_env,
        include_hold=bool(args.hold_actions),
        next_n=int(args.next_n),
        reward_profile=str(args.reward_profile),
        observation_mode=args.obs,
    )

    model = DQN.load(str(model_path), env=env)

    try:
        for ep in range(1, int(args.episodes) + 1):
            obs, info = env.reset(seed=args.seed + ep)
            episode_lines = 0
            episode_return = 0.0
            steps = 0

            while True:
                action, _ = model.predict(obs, deterministic=True)
                if isinstance(action, np.ndarray):
                    action = action.item()

                obs, reward, terminated, truncated, info = env.step(int(action))
                episode_return += float(reward)
                episode_lines += int(info.get("lines_cleared", 0) or 0)
                steps += 1

                try:
                    env.render()
                except Exception:
                    pass

                _pump_render_events()

                if args.sleep and args.sleep > 0:
                    time.sleep(float(args.sleep))

                if terminated or truncated:
                    print(
                        f"Game {ep}: placements={steps}  lines_cleared={episode_lines}  return={episode_return:.2f}"
                    )
                    if args.episode_pause and args.episode_pause > 0:
                        time.sleep(float(args.episode_pause))
                    break

        if args.hold:
            input("Press Enter to close... ")

    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        env.close()


if __name__ == "__main__":
    main()

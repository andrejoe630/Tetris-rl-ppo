"""Simple launcher to play the latest trained bot with sensible defaults."""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
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

def _infer_settings_from_name(model_path: Path) -> dict[str, object]:
    name = model_path.name.lower()
    obs = "board" if "board" in name else "features"

    next_n = 1
    m = re.search(r"next(\d+)", name)
    if m:
        try:
            next_n = int(m.group(1))
        except Exception:
            pass

    hold_actions = "hold" in name

    if "survival" in name:
        reward_profile = "survival"
    elif "balanced" in name:
        reward_profile = "balanced"
    else:
        reward_profile = "tetris"

    return {
        "obs": obs,
        "next_n": next_n,
        "hold_actions": hold_actions,
        "reward_profile": reward_profile,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="tetris_gymnasium/Tetris")
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument(
        "--obs",
        type=str,
        default=None,
        choices=["features", "board"],
        help="Observation mode used during training for this checkpoint.",
    )
    parser.add_argument(
        "--next-n",
        type=int,
        default=None,
        help="How many upcoming pieces to include (0 disables). Must match the checkpoint.",
    )
    parser.add_argument(
        "--reward-profile",
        type=str,
        default=None,
        choices=["tetris", "survival", "balanced"],
        help="Reward shaping profile (only affects reported reward).",
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--episode-pause", type=float, default=0.5)
    parser.add_argument(
        "--hold-actions",
        dest="hold_actions",
        action="store_true",
        help="Enable hold/swap as part of the macro-action space.",
    )
    parser.add_argument(
        "--no-hold-actions",
        dest="hold_actions",
        action="store_false",
        help="Disable hold/swap (for checkpoints trained without hold).",
    )
    parser.set_defaults(hold_actions=None)
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    model_path = Path(args.model_path) if args.model_path else find_latest_model(models_dir)
    inferred = _infer_settings_from_name(model_path)
    obs = args.obs or str(inferred["obs"])
    next_n = int(args.next_n) if args.next_n is not None else int(inferred["next_n"])
    reward_profile = args.reward_profile or str(inferred["reward_profile"])
    if args.hold_actions is None:
        hold_actions = bool(inferred["hold_actions"])
    else:
        hold_actions = bool(args.hold_actions)

    print(f"Loading model: {model_path}")
    print(f"Settings: obs={obs} next_n={next_n} hold_actions={hold_actions} reward_profile={reward_profile}")

    base_env = gym.make(args.env_id, render_mode="human", gravity=False)
    env = TetrisPlacementEnv(
        base_env,
        include_hold=bool(hold_actions),
        next_n=int(next_n),
        reward_profile=str(reward_profile),
        observation_mode=str(obs),
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

    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        env.close()


if __name__ == "__main__":
    main()

"""Quick eval to count lines cleared."""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import glob
import os

import gymnasium as gym
from stable_baselines3 import DQN
import tetris_gymnasium.envs  # noqa
from tetris_macro_env import TetrisPlacementEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--obs", type=str, default="board")
    parser.add_argument("--prefix", type=str, default="dqn_tetris_board")
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
        help="Reward shaping profile (only affects reward, not lines).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Env seed (for reproducible eval).")
    args = parser.parse_args()

    base = gym.make("tetris_gymnasium/Tetris", render_mode=None, gravity=False)
    env = TetrisPlacementEnv(
        base,
        include_hold=bool(args.hold_actions),
        next_n=int(args.next_n),
        reward_profile=str(args.reward_profile),
        observation_mode=args.obs,
    )

    pattern = str((_REPO_ROOT / "models") / f"{args.prefix}_*_steps.zip")
    zips = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not zips:
        raise SystemExit(f"No checkpoints found for prefix {args.prefix}")
    model_path = zips[0]
    print(f"Loading: {model_path}")

    model = DQN.load(model_path, env=env)

    total_lines = 0
    episodes = 0
    ep_lines = 0
    ep_placements = 0
    max_lines = 0
    max_placements = 0
    obs, _ = env.reset(seed=int(args.seed))

    for step in range(args.steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        lines = info.get("lines_cleared", 0)
        total_lines += lines
        ep_lines += lines
        ep_placements += 1
        if terminated or truncated:
            if ep_lines > max_lines:
                max_lines = ep_lines
            if ep_placements > max_placements:
                max_placements = ep_placements
            episodes += 1
            ep_lines = 0
            ep_placements = 0
            obs, _ = env.reset()

    print(f"Placements: {args.steps}")
    print(f"Episodes: {episodes}")
    print(f"Total lines cleared: {total_lines}")
    print(f"Lines per episode (avg): {total_lines / max(1, episodes):.1f}")
    print(f"Placements per episode (avg): {args.steps / max(1, episodes):.1f}")
    print(f"Best episode lines: {max_lines}")
    print(f"Best episode placements: {max_placements}")
    env.close()


if __name__ == "__main__":
    main()

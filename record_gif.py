"""Record a short GIF of the trained bot playing."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np

from stable_baselines3 import DQN

# Ensure env registration
import tetris_gymnasium.envs  # noqa: F401

from tetris_macro_env import TetrisPlacementEnv


ENV_ID_CANDIDATES = [
    "tetris_gymnasium/Tetris",
    "Tetris-v0",
    "Tetris-v1",
    "tetris_gymnasium/Tetris-v0",
    "tetris_gymnasium/Tetris-v1",
    "tetris_gymnasium:Tetris-v0",
    "tetris_gymnasium:Tetris-v1",
]


def find_latest_model(models_dir: Path) -> Path:
    models_dir = Path(models_dir)
    zips = [p for p in models_dir.glob("*.zip") if p.is_file()]
    if not zips:
        raise FileNotFoundError(f"No .zip models found in: {models_dir.resolve()}")
    zips.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return zips[0]


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


def _try_make_env(env_id: str, render_mode: Optional[str]) -> gym.Env:
    if render_mode is None:
        return gym.make(env_id, gravity=False)
    try:
        return gym.make(env_id, render_mode=render_mode, gravity=False)
    except TypeError:
        return gym.make(env_id, gravity=False)


def make_tetris_env(env_id: Optional[str], render_mode: Optional[str]) -> gym.Env:
    ids = [env_id] if env_id else ENV_ID_CANDIDATES
    last_exc: Optional[BaseException] = None

    for candidate in ids:
        try:
            return _try_make_env(candidate, render_mode=render_mode)
        except Exception as e:
            last_exc = e

    registry_ids = sorted(list(gym.envs.registry.keys()))
    tetris_like = [rid for rid in registry_ids if "tetris" in rid.lower()]
    msg = (
        "Failed to create a Tetris environment.\n"
        f"Tried env IDs: {ids}\n\n"
        "Available registered env IDs containing 'tetris':\n"
        + ("\n".join(f"  - {rid}" for rid in tetris_like) if tetris_like else "  (none found)\n")
        + "\nTip: pass --env-id <one_of_the_registered_ids>.\n"
    )
    raise RuntimeError(msg) from last_exc


def _extract_frame(frame: object) -> Optional[np.ndarray]:
    if frame is None:
        return None

    if isinstance(frame, tuple) and frame and isinstance(frame[0], np.ndarray):
        frame = frame[0]

    try:
        import pygame

        if isinstance(frame, pygame.Surface):
            arr = pygame.surfarray.array3d(frame)
            frame = np.transpose(arr, (1, 0, 2))
    except Exception:
        pass

    if isinstance(frame, np.ndarray):
        if frame.ndim == 3 and frame.shape[-1] == 4:
            frame = frame[:, :, :3]
        return frame

    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default=None)
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
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--episode-pause", type=float, default=0.5)
    parser.add_argument("--out", type=str, default="assets/tetris_bot.gif")
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

    base_env = make_tetris_env(args.env_id, render_mode="rgb_array")
    env = TetrisPlacementEnv(
        base_env,
        include_hold=bool(hold_actions),
        next_n=int(next_n),
        reward_profile=str(reward_profile),
        observation_mode=str(obs),
    )

    model = DQN.load(str(model_path), env=env)

    frames: list[np.ndarray] = []
    frame_skip = max(int(args.frame_skip), 1)
    max_steps = int(args.max_steps)
    pause_frames = max(int(round(float(args.episode_pause) * float(args.fps))), 0)

    try:
        for ep in range(1, int(args.episodes) + 1):
            obs, info = env.reset(seed=args.seed + ep)

            frame = _extract_frame(env.render())
            if frame is not None:
                frames.append(frame)

            steps = 0
            while True:
                action, _ = model.predict(obs, deterministic=True)
                if isinstance(action, np.ndarray):
                    action = action.item()

                obs, reward, terminated, truncated, info = env.step(int(action))
                steps += 1

                if steps % frame_skip == 0:
                    frame = _extract_frame(env.render())
                    if frame is not None:
                        frames.append(frame)

                if terminated or truncated or (max_steps > 0 and steps >= max_steps):
                    if pause_frames > 0 and frames:
                        frames.extend([frames[-1]] * pause_frames)
                    break
    finally:
        env.close()

    if not frames:
        raise RuntimeError("No frames captured. Check render support for this environment.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=int(args.fps))
    print(f"Wrote GIF: {out_path} ({len(frames)} frames)")


if __name__ == "__main__":
    main()

"""Train DQN on Tetris using macro (placement) actions.

This script is the recommended training setup for this repo:
- Macro-actions: each env step chooses (rotation, x[, hold]) then hard-drops.
- DQN: value-based RL with epsilon-greedy exploration.

Run (server/headless):
  python train_dqn_server.py --timesteps 500000

Logs:
  tensorboard --logdir ./logs

Models:
  Saved to ./models/ as .zip checkpoints.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent
if not (_REPO_ROOT / "tetris_macro_env.py").exists():
    _REPO_ROOT = _REPO_ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Ensure env is registered
try:
    import tetris_gymnasium  # noqa: F401
    import tetris_gymnasium.envs  # noqa: F401
except ImportError as e:
    raise SystemExit(
        "Missing dependency 'tetris-gymnasium'. Install with:\n  pip install -r requirements.txt"
    ) from e

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


def _try_make_env(env_id: str, render_mode: Optional[str]) -> gym.Env:
    if render_mode is None:
        # Disable gravity so internal swap/rotate doesn't cause unintended drops.
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


def build_env(
    env_id: Optional[str],
    seed: int,
    obs_mode: str,
    include_hold: bool,
    next_n: int,
    reward_profile: str,
) -> gym.Env:
    base = make_tetris_env(env_id=env_id, render_mode=None)

    # Macro-action + shaped reward
    env: gym.Env = TetrisPlacementEnv(
        base,
        include_hold=bool(include_hold),
        next_n=int(next_n),
        reward_profile=str(reward_profile),
        observation_mode=obs_mode,
    )

    env = Monitor(env)
    env.reset(seed=seed)
    return env


class ResumeLinearSchedule:
    """Picklable linear schedule anchored to absolute timesteps.

    SB3 schedules receive only `progress_remaining` (1 -> 0 over training).
    When resuming with reset_num_timesteps=False, progress_remaining is computed
    from absolute timesteps, so we can reconstruct the current timestep as:

        t = (1 - progress_remaining) * total_timesteps

    This allows specifying decay windows in absolute timesteps (e.g. decay for the
    next N steps after resume) without resetting the model's timestep counter.
    """

    def __init__(
        self,
        start_value: float,
        end_value: float,
        *,
        start_timestep: int,
        end_timestep: int,
        total_timesteps: int,
    ) -> None:
        self.start_value = float(start_value)
        self.end_value = float(end_value)
        self.start_timestep = int(start_timestep)
        self.end_timestep = int(end_timestep)
        self.total_timesteps = int(total_timesteps) if int(total_timesteps) > 0 else 1

        if self.end_timestep < self.start_timestep:
            self.end_timestep = self.start_timestep

    def __call__(self, progress_remaining: float) -> float:
        pr = float(progress_remaining)
        # Clamp to avoid weird values from user-defined callbacks, etc.
        if pr < 0.0:
            pr = 0.0
        elif pr > 1.0:
            pr = 1.0

        cur_timestep = int(round((1.0 - pr) * float(self.total_timesteps)))

        if cur_timestep <= self.start_timestep:
            return float(self.start_value)
        if cur_timestep >= self.end_timestep:
            return float(self.end_value)

        denom = float(self.end_timestep - self.start_timestep)
        if denom <= 0:
            return float(self.end_value)

        alpha = float(cur_timestep - self.start_timestep) / denom
        return float(self.start_value + alpha * (self.end_value - self.start_value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default=None)
    parser.add_argument(
        "--obs",
        type=str,
        default="features",
        choices=["features", "board"],
        help="Observation mode: 'features' (legacy compact) or 'board' (20x10 occupancy + features).",
    )
    parser.add_argument(
        "--hold-actions",
        action="store_true",
        help="Enable hold/swap as part of the macro-action space (new training run).",
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
        help=(
            "Reward shaping profile. "
            "'tetris' matches existing behavior; "
            "'survival' penalizes dying more and reduces the Tetris jackpot; "
            "'balanced' keeps the strong death penalty but restores the Tetris jackpot."
        ),
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Total training timesteps target (placements).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--name-prefix", type=str, default="dqn_tetris_macro")
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Optional path to an SB3 DQN .zip checkpoint to resume from.",
    )
    parser.add_argument(
        "--resume-eps-start",
        type=float,
        default=0.05,
        help=(
            "If resuming, start epsilon for exploration. "
            "Increase (e.g. 0.15-0.25) to help break plateaus."
        ),
    )
    parser.add_argument(
        "--resume-eps-final",
        type=float,
        default=0.05,
        help=(
            "If resuming with --resume-eps-decay-steps > 0, linearly decay epsilon "
            "toward this final value."
        ),
    )
    parser.add_argument(
        "--resume-eps-decay-steps",
        type=int,
        default=0,
        help=(
            "If resuming, and > 0, linearly decay epsilon from --resume-eps-start to "
            "--resume-eps-final over this many timesteps after resume. If 0, keep epsilon "
            "constant at --resume-eps-start."
        ),
    )
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    train_env = DummyVecEnv(
        [
            lambda: build_env(
                args.env_id,
                seed=args.seed,
                obs_mode=args.obs,
                include_hold=args.hold_actions,
                next_n=args.next_n,
                reward_profile=args.reward_profile,
            )
        ]
    )

    n_envs = int(getattr(train_env, "num_envs", 1))
    save_freq = max(int(args.checkpoint_freq // n_envs), 1)

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(models_dir),
        name_prefix=args.name_prefix,
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    if args.resume_from:
        model = DQN.load(args.resume_from, env=train_env, device="auto")
        model.verbose = 1
        # Ensure TB log path is configured for continued runs
        model.tensorboard_log = str(log_dir)

        already = int(getattr(model, "num_timesteps", 0) or 0)
        remaining = max(int(args.timesteps) - already, 0)
        print(f"Resuming from {args.resume_from} (num_timesteps={already}). Remaining={remaining}.")

        start_eps = float(args.resume_eps_start)
        final_eps = float(args.resume_eps_final)
        decay_steps = int(args.resume_eps_decay_steps)

        if remaining > 0 and decay_steps > 0:
            # Decay epsilon over the next `decay_steps` AFTER resume.
            # We anchor the window at the model's current timestep counter so it works even
            # when resuming late in training (reset_num_timesteps=False).
            end_t = int(min(already + int(decay_steps), int(args.timesteps)))

            model.exploration_initial_eps = float(start_eps)
            model.exploration_final_eps = float(final_eps)
            # exploration_fraction is used by SB3's default schedule; we use a custom schedule.
            model.exploration_fraction = 1.0
            model.exploration_schedule = ResumeLinearSchedule(
                start_eps,
                final_eps,
                start_timestep=int(already),
                end_timestep=int(end_t),
                total_timesteps=int(args.timesteps),
            )

            print(
                f"Set resume exploration schedule: eps {start_eps} -> {final_eps} from t={already} to t={end_t} (decay_steps={decay_steps})"
            )
        else:
            # Constant epsilon
            model.exploration_initial_eps = float(start_eps)
            model.exploration_final_eps = float(start_eps)
            model.exploration_fraction = 1.0
            model.exploration_schedule = ResumeLinearSchedule(
                start_eps,
                start_eps,
                start_timestep=int(already),
                end_timestep=int(already),
                total_timesteps=int(args.timesteps),
            )

            print(f"Fixed exploration schedule to constant {start_eps}")

        if remaining > 0:
            model.learn(
                total_timesteps=int(remaining),
                callback=checkpoint_callback,
                tb_log_name=args.name_prefix,
                reset_num_timesteps=False,
                progress_bar=False,
            )
    else:
        model = DQN(
            policy="MlpPolicy",
            env=train_env,
            verbose=1,
            seed=args.seed,
            device="auto",
            tensorboard_log=str(log_dir),
            learning_rate=1e-4,
            buffer_size=500_000,
            learning_starts=10_000,
            batch_size=64,
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1_000,
            exploration_fraction=0.10,
            exploration_final_eps=0.05,
        )

        model.learn(
            total_timesteps=int(args.timesteps),
            callback=checkpoint_callback,
            tb_log_name=args.name_prefix,
            progress_bar=False,
        )

    model.save(str(models_dir / f"{args.name_prefix}_final"))
    train_env.close()


if __name__ == "__main__":
    main()

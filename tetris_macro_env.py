"""Macro-action Tetris environment for RL.

This module turns the primitive per-frame action space (left/right/rotate/etc.) into
"placement" actions:
  - choose a rotation (0..3)
  - choose a target x position
  - (optional) choose whether to use hold/swap

One wrapper step executes the placement and commits the piece via hard_drop.

Why:
- Removes long action sequences and delayed credit assignment.
- Makes value-based methods (DQN) much more reliable.

Observation modes:
- "features": compact vector [agg_height, holes, bumpiness, max_height] (normalized)
- "board": flattened 20x10 locked-board occupancy (0/1) plus optional helper features

Both modes append one-hot(current_piece) and (optionally) one-hot(next_n_pieces).

Rewards are shaped per placement:
  + line clear bonus
  + small survive bonus
  + bonuses for reducing holes/bumpiness
  - penalties for *increasing* holes/height/bumpiness/max_height (gated by stack height)
  - invalid-action penalty
  - game-over penalty

NOTE: This wrapper assumes tetris_gymnasium's dict observation format with
'board' and 'active_tetromino_mask'.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np


def _locked_playfield_01(env: gym.Env, obs: Dict[str, Any]) -> np.ndarray:
    """Return binary occupancy of the *locked* 20x10 playfield.

    - Removes the active tetromino via active_tetromino_mask
    - Crops away padding (left/right and bottom bedrock) using env.unwrapped.padding
    """

    board = np.asarray(obs["board"], dtype=np.uint8)
    mask = np.asarray(obs.get("active_tetromino_mask", 0), dtype=np.uint8)

    locked = board.copy()
    locked[mask != 0] = 0

    # Crop padding: board is (height+pad, width+2*pad)
    p = int(getattr(env.unwrapped, "padding", 0) or 0)
    if p > 0:
        locked = locked[0:-p, p:-p]

    return (locked != 0).astype(np.uint8)


def compute_features(board_01: np.ndarray) -> Tuple[int, int, int, int]:
    """Compute (aggregate_height, holes, bumpiness, max_height) on a 0/1 board."""

    h, w = board_01.shape
    heights = np.zeros(w, dtype=np.int32)
    holes = 0

    for x in range(w):
        col = board_01[:, x]
        filled_idxs = np.flatnonzero(col)
        if filled_idxs.size == 0:
            heights[x] = 0
            continue

        top = int(filled_idxs[0])
        heights[x] = h - top
        holes += int(np.sum(col[top:] == 0))

    aggregate_height = int(np.sum(heights))
    bumpiness = int(np.sum(np.abs(np.diff(heights)))) if w > 1 else 0
    max_height = int(np.max(heights)) if w > 0 else 0

    return aggregate_height, holes, bumpiness, max_height


def _one_hot(idx: int, size: int) -> np.ndarray:
    v = np.zeros(size, dtype=np.float32)
    if 0 <= idx < size:
        v[idx] = 1.0
    return v


@dataclass(frozen=True)
class PlacementAction:
    hold: int
    rot_cw: int
    x: int


class TetrisPlacementEnv(gym.Wrapper):
    """Wrap tetris_gymnasium to expose placement macro-actions for DQN."""

    # Reward (tune as needed)
    # Original V1 rewards
    LINE_CLEAR_REWARDS = {0: 0.0, 1: 1.00, 2: 3.00, 3: 6.00, 4: 12.00}

    # Encourage survival (longer games -> more opportunities to clear lines)
    SURVIVE_BONUS = 0.04
    GAME_OVER_PENALTY = 2.00
    INVALID_ACTION_PENALTY = 0.20

    # Structural penalties (applied every placement, scaled by stack height)
    # The old behavior gated these penalties behind a hard max-height threshold, which
    # let the agent create holes early and then inevitably die later.
    HOLE_CREATED_PENALTY = 0.10
    HEIGHT_INCREASE_PENALTY = 0.006
    BUMP_INCREASE_PENALTY = 0.004
    MAX_HEIGHT_INCREASE_PENALTY = 0.020

    # Bonuses for improving board quality
    HOLE_REDUCTION_BONUS = 0.03
    BUMP_REDUCTION_BONUS = 0.003

    # Always-on penalty scaling: at low stacks, only a fraction of structural penalties apply.
    # As the stack grows, penalties ramp up smoothly toward full strength.
    PENALTY_SCALE_BASE = 0.10

    def _apply_reward_profile(self, reward_profile: str) -> None:
        rp = str(reward_profile or "tetris").lower().strip()
        if rp in {"tetris", "default"}:
            # Existing behavior (tetris-biased).
            self.LINE_CLEAR_REWARDS = {0: 0.0, 1: 1.00, 2: 3.00, 3: 6.00, 4: 12.00}
            self.GAME_OVER_PENALTY = 2.00
            return

        if rp in {"survival"}:
            # Survival-focused: make dying expensive and reduce the Tetris jackpot so the
            # agent doesn't over-commit to deep wells / risky stacking.
            self.LINE_CLEAR_REWARDS = {0: 0.0, 1: 1.00, 2: 3.00, 3: 6.00, 4: 8.00}
            self.GAME_OVER_PENALTY = 15.00
            return

        if rp in {"balanced"}:
            # Balanced: keep the strong death penalty from "survival" but restore the
            # classic Tetris jackpot so high-scoring play is still attractive.
            self.LINE_CLEAR_REWARDS = {0: 0.0, 1: 1.00, 2: 3.00, 3: 6.00, 4: 12.00}
            self.GAME_OVER_PENALTY = 15.00
            return

        raise ValueError(
            f"Unknown reward_profile: {reward_profile!r}. Expected one of: 'tetris', 'survival', 'balanced'."
        )

    def __init__(
        self,
        env: gym.Env,
        *,
        include_hold: bool = False,
        use_next_piece: bool = True,
        next_n: Optional[int] = None,
        reward_profile: str = "tetris",
        observation_mode: str = "features",
    ):
        super().__init__(env)

        self.reward_profile = str(reward_profile or "tetris")
        self._apply_reward_profile(self.reward_profile)

        self.include_hold = bool(include_hold)

        if next_n is None:
            self.next_n = 1 if bool(use_next_piece) else 0
        else:
            self.next_n = int(next_n)
        if self.next_n < 0:
            raise ValueError(f"next_n must be >= 0, got: {next_n!r}")

        # Back-compat: older configs pass use_next_piece=True/False.
        self.use_next_piece = bool(self.next_n > 0)

        self.observation_mode = str(observation_mode).lower().strip()
        if self.observation_mode not in {"features", "board"}:
            raise ValueError(f"observation_mode must be 'features' or 'board', got: {observation_mode!r}")

        # Action encoding
        u = self.env.unwrapped
        self._x_count = int(u.width_padded - u.padding + 1)
        self._rots = 4
        self._hold_opts = 2 if self.include_hold else 1
        n_actions = self._hold_opts * self._rots * self._x_count
        self.action_space = gym.spaces.Discrete(int(n_actions))

        piece_dim = 7
        extra = piece_dim  # current piece
        if self.next_n > 0:
            extra += piece_dim * int(self.next_n)
        if self.include_hold:
            # Held piece (for swap/hold decision)
            extra += piece_dim

        # Track the currently held piece (0..6), or -1 if hold slot is empty.
        self._held_piece_idx: int = -1

        # Derive the (cropped) playfield size from the base env's observation space.
        self._board_h = 20
        self._board_w = 10
        try:
            space = getattr(self.env, "observation_space", None)
            if isinstance(space, gym.spaces.Dict) and "board" in space.spaces:
                raw_shape = tuple(int(x) for x in space.spaces["board"].shape)
                p = int(getattr(u, "padding", 0) or 0)
                if len(raw_shape) == 2:
                    h_raw, w_raw = raw_shape
                    h = h_raw - p if p > 0 else h_raw
                    w = w_raw - (2 * p) if p > 0 else w_raw
                    if h > 0 and w > 0:
                        self._board_h = int(h)
                        self._board_w = int(w)
        except Exception:
            pass

        if self.observation_mode == "features":
            feat_dim = 4
            obs_dim = feat_dim + extra
        else:
            board_dim = int(self._board_h * self._board_w)
            # Add helper features too (helps learning): [agg_h, holes, bump, max_h]
            feat_dim = 4
            obs_dim = board_dim + feat_dim + extra

        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(int(obs_dim),),
            dtype=np.float32,
        )

        # Cached previous features
        self._prev_agg_h: Optional[int] = None
        self._prev_holes: Optional[int] = None
        self._prev_bump: Optional[int] = None
        self._prev_max_h: Optional[int] = None

    def decode_action(self, a: int) -> PlacementAction:
        a = int(a)
        per_hold = self._rots * self._x_count
        hold = 0
        if self.include_hold:
            hold = a // per_hold
            a = a % per_hold
        rot = a // self._x_count
        x = a % self._x_count
        return PlacementAction(hold=int(hold), rot_cw=int(rot), x=int(x))

    def _tetromino_id_offset(self) -> int:
        """tetris_gymnasium uses raw tetromino ids offset by len(base_pixels) (usually 2)."""
        u = self.env.unwrapped
        try:
            return int(len(getattr(u, "base_pixels", [])) or 2)
        except Exception:
            return 2

    def _raw_tetromino_id_to_index(self, raw_id: int) -> int:
        idx = int(raw_id) - int(self._tetromino_id_offset())
        return idx if 0 <= idx < 7 else -1

    def _current_piece_index(self) -> int:
        u = self.env.unwrapped
        return self._raw_tetromino_id_to_index(int(getattr(u.active_tetromino, "id", 0)))

    def _next_piece_indices(self) -> list[int]:
        if self.next_n <= 0:
            return []

        u = self.env.unwrapped
        try:
            q = u.queue.get_queue()
        except Exception:
            q = []

        idxs: list[int] = []
        for i in range(int(self.next_n)):
            if i < len(q):
                idx = int(q[i])
                idxs.append(idx if 0 <= idx < 7 else -1)
            else:
                idxs.append(-1)
        return idxs

    def _next_piece_index(self) -> int:
        """Back-compat helper: return the first upcoming piece (or -1)."""
        idxs = self._next_piece_indices()
        return int(idxs[0]) if idxs else -1

    def _obs_from_raw(self, raw_obs: Dict[str, Any]) -> np.ndarray:
        board_01 = _locked_playfield_01(self.env, raw_obs)
        agg_h, holes, bump, max_h = compute_features(board_01)

        # Normalize to [0,1] with rough maxima
        # width=10, height=20 => agg_height/200, holes/200, bump/200, max_h/20
        feat = np.array(
            [
                agg_h / 200.0,
                holes / 200.0,
                bump / 200.0,
                max_h / 20.0,
            ],
            dtype=np.float32,
        )

        cur = _one_hot(self._current_piece_index(), 7)
        nxt_vecs = [_one_hot(i, 7) for i in self._next_piece_indices()] if self.next_n > 0 else []
        held = _one_hot(self._held_piece_idx, 7) if self.include_hold else None

        if self.observation_mode == "features":
            parts = [feat, cur]
            parts.extend(nxt_vecs)
            if held is not None:
                parts.append(held)
            return np.concatenate(parts).astype(np.float32)

        # "board" mode: flattened 20x10 occupancy + helper features + piece ids
        board_flat = board_01.astype(np.float32).reshape(-1)
        parts = [board_flat, feat, cur]
        parts.extend(nxt_vecs)
        if held is not None:
            parts.append(held)
        return np.concatenate(parts).astype(np.float32)

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        raw_obs, info = self.env.reset(**kwargs)
        board_01 = _locked_playfield_01(self.env, raw_obs)
        agg_h, holes, bump, max_h = compute_features(board_01)
        self._prev_agg_h = agg_h
        self._prev_holes = holes
        self._prev_bump = bump
        self._prev_max_h = max_h

        # Reset held-piece tracking. (If include_hold=False, this is unused.)
        self._held_piece_idx = -1
        if self.include_hold:
            try:
                state = self.env.unwrapped.get_state()
                held = state.holder.get_tetrominoes()
                if held:
                    self._held_piece_idx = self._raw_tetromino_id_to_index(int(getattr(held[0], "id", -1)))
            except Exception:
                pass

        return self._obs_from_raw(raw_obs), info

    def step(self, action: int):
        u = self.env.unwrapped
        act = self.decode_action(action)

        invalid = False

        # Optional hold/swap
        if self.include_hold and act.hold == 1:
            # After a swap, the held slot always becomes the previously active piece.
            prev_active_idx = self._current_piece_index()

            # swap action respects has_swapped internally
            _o, _r, terminated, truncated, _info = self.env.step(int(u.actions.swap))
            if terminated or truncated:
                # game ended during swap (rare)
                raw_obs = _o
                info = dict(_info)
                obs = self._obs_from_raw(raw_obs)
                return obs, float(-self.GAME_OVER_PENALTY), bool(terminated), bool(truncated), info

            self._held_piece_idx = int(prev_active_idx)

        # Try to set pose directly (rotation + x at y=0)
        state = u.get_state()
        tet = state.active_tetromino
        for _ in range(int(act.rot_cw) % 4):
            tet = u.rotate(tet, True)
        state.active_tetromino = tet
        state.x = int(act.x)
        state.y = 0

        if u.collision(state.active_tetromino, state.x, state.y):
            invalid = True
        else:
            u.set_state(state)

        # Commit the piece
        raw_obs, _env_reward, terminated, truncated, info = self.env.step(int(u.actions.hard_drop))
        info = dict(info)

        # Compute feature deltas
        board_01 = _locked_playfield_01(self.env, raw_obs)
        agg_h, holes, bump, max_h = compute_features(board_01)

        prev_agg_h = int(self._prev_agg_h if self._prev_agg_h is not None else agg_h)
        prev_holes = int(self._prev_holes if self._prev_holes is not None else holes)
        prev_bump = int(self._prev_bump if self._prev_bump is not None else bump)
        prev_max_h = int(self._prev_max_h if self._prev_max_h is not None else max_h)

        delta_holes = holes - prev_holes
        delta_height = agg_h - prev_agg_h
        delta_bump = bump - prev_bump
        delta_max_h = max_h - prev_max_h

        # Update prev
        self._prev_agg_h = agg_h
        self._prev_holes = holes
        self._prev_bump = bump
        self._prev_max_h = max_h

        # Lines cleared from info (tetris_gymnasium provides this per step)
        lines_cleared = int(info.get("lines_cleared", 0) or 0)

        # Reward shaping (per placement)
        reward = 0.0

        # Encourage survival (one piece placed without topping out)
        if not (terminated or truncated):
            reward += self.SURVIVE_BONUS

        # Line clears
        reward += float(self.LINE_CLEAR_REWARDS.get(lines_cleared, 0.0))

        # Bonuses for making the board cleaner/flatter
        if delta_holes < 0:
            reward += self.HOLE_REDUCTION_BONUS * float(-delta_holes)
        if delta_bump < 0:
            reward += self.BUMP_REDUCTION_BONUS * float(-delta_bump)

        # Structural penalties: always on, but scale up as the stack grows.
        # This pushes the agent to avoid creating holes early (long-term survivability)
        # without being overly harsh when the board is still empty.
        h_ref = float(board_01.shape[0] or 20)
        height_ratio = float(max_h) / h_ref if h_ref > 0 else 0.0
        height_ratio = float(np.clip(height_ratio, 0.0, 1.0))
        penalty_scale = float(self.PENALTY_SCALE_BASE + (1.0 - self.PENALTY_SCALE_BASE) * height_ratio)

        if delta_holes > 0:
            reward -= self.HOLE_CREATED_PENALTY * penalty_scale * float(delta_holes)
        if delta_height > 0:
            reward -= self.HEIGHT_INCREASE_PENALTY * penalty_scale * float(delta_height)
        if delta_bump > 0:
            reward -= self.BUMP_INCREASE_PENALTY * penalty_scale * float(delta_bump)
        if delta_max_h > 0:
            reward -= self.MAX_HEIGHT_INCREASE_PENALTY * penalty_scale * float(delta_max_h)

        if invalid:
            reward -= self.INVALID_ACTION_PENALTY

        if terminated or truncated:
            reward -= self.GAME_OVER_PENALTY

        # Attach diagnostics
        info["custom_reward"] = float(reward)
        info["lines_cleared"] = int(lines_cleared)
        info["delta_holes"] = int(delta_holes)
        info["delta_height"] = int(delta_height)
        info["delta_bump"] = int(delta_bump)
        info["delta_max_height"] = int(delta_max_h)
        info["holes"] = int(holes)
        info["aggregate_height"] = int(agg_h)
        info["bumpiness"] = int(bump)
        info["max_height"] = int(max_h)
        info["invalid_macro_action"] = bool(invalid)
        info["macro_action_hold"] = int(act.hold)
        info["macro_action_rot"] = int(act.rot_cw)
        info["macro_action_x"] = int(act.x)

        obs = self._obs_from_raw(raw_obs)
        return obs, float(reward), bool(terminated), bool(truncated), info

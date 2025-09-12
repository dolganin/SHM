"""VizDoom environment wrapper.

This wrapper creates a minimal interface around ``vizdoom`` providing the
standard Gymnasium ``reset`` and ``step`` methods.  Observations are
returned in ``float32`` channel-first format.
"""
from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import gymnasium as gym
from vizdoom import DoomGame, ScreenResolution


class VizdoomEnv:
    """Simple wrapper for VizDoom environments."""

    def __init__(self, scenario: str, config: str, frame_skip: int = 4,
                 resolution: str = "RES_160X120") -> None:
        self._game = DoomGame()
        self._game.load_config(config)
        self._game.set_doom_scenario_path(scenario)
        self._game.set_screen_resolution(getattr(ScreenResolution, resolution))
        self._game.init()

        self.frame_skip = frame_skip
        # Define observation and action spaces for interoperability
        c = self._game.get_screen_channels()
        h = self._game.get_screen_height()
        w = self._game.get_screen_width()
        self.observation_space = gym.spaces.Box(0, 255, (c, h, w), dtype=np.float32)
        self.action_space = gym.spaces.MultiBinary(self._game.get_available_buttons_size())

    def reset(self) -> Tuple[np.ndarray, dict]:
        self._game.new_episode()
        state = self._game.get_state()
        obs = np.copy(state.screen_buffer)
        return obs.astype(np.float32), {}

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, dict]:
        reward = self._game.make_action(action.tolist(), self.frame_skip)
        done = self._game.is_episode_finished()
        if not done:
            obs = self._game.get_state().screen_buffer
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
        info = {}
        return obs.astype(np.float32), reward, done, False, info

    def close(self) -> None:
        self._game.close()

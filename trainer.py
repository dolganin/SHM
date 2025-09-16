"""Simplified PPO trainer with success-rate logging and GIF recording (SHM paths)."""
from __future__ import annotations

import os
import time
from collections import deque
from typing import Any, Dict, List

import imageio
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from buffer import Buffer
from model import ActorCriticModel
from utils import create_env_from_config, polynomial_decay


class PPOTrainer:
    def __init__(self, config: Dict[str, Any], run_id: str, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.run_id = run_id

        # --- SHM paths for logs and videos ---
        base_log_dir = "../logs/shm/"
        base_video_dir = "../logs/videos_shm"
        os.makedirs(base_log_dir, exist_ok=True)
        os.makedirs(base_video_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_path = os.path.join(base_log_dir, run_id, timestamp)
        video_path = os.path.join(base_video_dir, run_id)
        os.makedirs(log_path, exist_ok=True)
        os.makedirs(video_path, exist_ok=True)

        self.writer = SummaryWriter(log_path)
        self.video_dir = video_path

        # --- environments ---
        self.envs = [create_env_from_config(config["environment"]) for _ in range(config.get("n_workers", 1))]
        obs_shape = self.envs[0].observation_space.shape
        self.action_space = self.envs[0].action_space

        # --- model ---
        hidden_size = config.get("hidden_layer_size", 256)
        self.recurrence = config.get("recurrence", {"layer_type": "gru", "hidden_state_size": hidden_size})
        self.model = ActorCriticModel(obs_shape, self.action_space, hidden_size, self.recurrence).to(device)

        # --- buffer ---
        gamma = config.get("gamma", 0.99)
        lamda = config.get("lamda", 0.95)
        self.buffer = Buffer(gamma=gamma, lam=lamda)

        # --- schedules ---
        self.lr_schedule = config.get(
            "learning_rate_schedule",
            {"initial": 3e-4, "final": 3e-4, "max_decay_steps": config.get("updates", 1), "power": 1.0},
        )
        self.beta_schedule = config.get(
            "beta_schedule",
            {"initial": 0.0, "final": 0.0, "max_decay_steps": config.get("updates", 1), "power": 1.0},
        )
        self.cr_schedule = config.get(
            "clip_range_schedule",
            {"initial": 0.2, "final": 0.2, "max_decay_steps": config.get("updates", 1), "power": 1.0},
        )

    # ------------------------------------------------------------------
    def run_training(self) -> None:
        """Runs the entire training logic from sampling data to optimizing the model."""
        print("Step 6: Starting training")
        episode_infos = deque(maxlen=100)

        for update in range(self.config["updates"]):
            learning_rate = polynomial_decay(
                self.lr_schedule["initial"], self.lr_schedule["final"],
                self.lr_schedule["max_decay_steps"], self.lr_schedule["power"], update
            )
            beta = polynomial_decay(
                self.beta_schedule["initial"], self.beta_schedule["final"],
                self.beta_schedule["max_decay_steps"], self.beta_schedule["power"], update
            )
            clip_range = polynomial_decay(
                self.cr_schedule["initial"], self.cr_schedule["final"],
                self.cr_schedule["max_decay_steps"], self.cr_schedule["power"], update
            )

            sampled_episode_info = self._sample_training_data()
            self.buffer.prepare_batch_dict()
            training_stats = self._train_epochs(learning_rate, clip_range, beta)
            training_stats = np.mean(training_stats, axis=0)

            episode_infos.extend(sampled_episode_info)
            episode_result = self._process_episode_info(episode_infos)

            rm = episode_result.get("reward_mean", 0.0)
            rs = episode_result.get("reward_std", 0.0)
            lm = episode_result.get("length_mean", 0.0)
            ls = episode_result.get("length_std", 0.0)
            sp = episode_result.get("success_percent", None)

            if sp is not None:
                result = (
                    "{:4} reward={:.2f} std={:.2f} length={:.1f} std={:.2f} success={:.2f} "
                    "pi_loss={:3f} v_loss={:3f} entropy={:.3f} loss={:3f} value={:.3f} advantage={:.3f}"
                ).format(
                    update, rm, rs, lm, ls, sp,
                    training_stats[0], training_stats[1], training_stats[3], training_stats[2],
                    torch.mean(self.buffer.values), torch.mean(self.buffer.advantages)
                )
            else:
                result = (
                    "{:4} reward={:.2f} std={:.2f} length={:.1f} std={:.2f} "
                    "pi_loss={:3f} v_loss={:3f} entropy={:.3f} loss={:3f} value={:.3f} advantage={:.3f}"
                ).format(
                    update, rm, rs, lm, ls,
                    training_stats[0], training_stats[1], training_stats[3], training_stats[2],
                    torch.mean(self.buffer.values), torch.mean(self.buffer.advantages)
                )

            print(result)
            self._write_training_summary(update, training_stats, episode_result)
            if update % self.config.get("video_every", 10) == 0:
                self._record_eval_episode(update)

            del self.buffer.samples_flat
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        self._save_model()

    # ------------------------------------------------------------------
    def _sample_training_data(self) -> List[Dict[str, Any]]:
        """Collects trajectories from environments."""
        self.buffer.reset()
        episode_infos: List[Dict[str, Any]] = []
        for env in self.envs:
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0
            ep_len = 0
            success = 0.0
            for _ in range(self.config.get("worker_steps", 32)):
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs).unsqueeze(0)
                    pol, value, _ = self.model(obs_t, None, self.device)
                    action = pol[0].sample().item()
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                self.buffer.add(obs, action, reward, done, value.item())
                ep_reward += reward
                ep_len += 1
                success = info.get("success", success)
                obs = next_obs
                if done:
                    episode_infos.append({"reward": ep_reward, "length": ep_len, "success": success})
                    obs, _ = env.reset()
                    ep_reward = 0.0
                    ep_len = 0
                    success = 0.0
        return episode_infos

    # ------------------------------------------------------------------
    def _train_epochs(self, lr: float, clip_range: float, beta: float) -> np.ndarray:
        """Dummy training step returning placeholder statistics for each epoch."""
        epochs = self.config.get("epochs", 1)
        stats = []
        for _ in range(epochs):
            pi_loss = np.random.random()
            v_loss = np.random.random()
            loss = pi_loss + v_loss
            entropy = np.random.random()
            stats.append([pi_loss, v_loss, loss, entropy])
        return np.array(stats)

    # ------------------------------------------------------------------
    def _process_episode_info(self, infos: deque) -> Dict[str, float]:
        rewards = [i.get("reward", 0.0) for i in infos]
        lengths = [i.get("length", 0) for i in infos]
        successes = [i.get("success", 0.0) for i in infos if "success" in i]
        result = {
            "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
            "reward_std": float(np.std(rewards)) if rewards else 0.0,
            "length_mean": float(np.mean(lengths)) if lengths else 0.0,
            "length_std": float(np.std(lengths)) if lengths else 0.0,
        }
        if successes:
            result["success_percent"] = 100.0 * float(np.mean(successes))
        return result

    # ------------------------------------------------------------------
    def _write_training_summary(self, update: int, stats: np.ndarray, episode_result: Dict[str, float]) -> None:
        self.writer.add_scalar("loss/policy", stats[0], update)
        self.writer.add_scalar("loss/value", stats[1], update)
        self.writer.add_scalar("loss/total", stats[2], update)
        self.writer.add_scalar("stats/entropy", stats[3], update)
        for k, v in episode_result.items():
            self.writer.add_scalar(f"episode/{k}", v, update)

    # ------------------------------------------------------------------
    def _to_hwc_uint8(self, frame) -> np.ndarray:
        """Normalizes frame to HWC uint8 RGB."""
        if torch.is_tensor(frame):
            frame = frame.detach().cpu().numpy()
        arr = np.asarray(frame)
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] > 3:
            arr = arr[:, :, :3]
        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32)
            a_min, a_max = float(arr.min()), float(arr.max())
            if 0.0 <= a_min and a_max <= 1.0:
                arr = (arr * 255.0).round()
            else:
                rng = max(1e-6, a_max - a_min)
                arr = ((arr - a_min) / rng * 255.0).round()
            arr = arr.clip(0, 255).astype(np.uint8)
        return arr

    # ------------------------------------------------------------------
    def _record_eval_episode(self, step: int) -> None:
        print(f"[eval] Recording evaluation gif for step {step}...")
        old_n_workers = self.config["n_workers"]
        self.config["n_workers"] = 1
        env = create_env_from_config(self.config["environment"])
        self.config["n_workers"] = old_n_workers

        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, (tuple, list)) and len(reset_out) >= 1 else reset_out

        frames = []
        hxs, cxs = self.model.init_recurrent_cell_states(1, self.device)
        rc = hxs if self.recurrence["layer_type"] == "gru" else (hxs, cxs)

        raw = env.render()
        first = self._to_hwc_uint8(raw)
        if self.config.get("downscale_eval_frames", True):
            target_h, target_w = first.shape[0] // 2, first.shape[1] // 2
        else:
            target_h, target_w = first.shape[0], first.shape[1]
        frame = first[:: first.shape[0] // target_h or 1, :: first.shape[1] // target_w or 1]
        frames.append(frame)

        done, steps = False, 0
        max_steps = int(self.config.get("eval_max_steps", 100))

        while not done and steps < max_steps:
            with torch.no_grad():
                obs_tensor = torch.as_tensor(np.asarray(obs)).unsqueeze(0).to(self.device)
                pol, _, rc = self.model(obs_tensor, rc, self.device)
                act = np.array([p.probs.argmax(-1).item() for p in pol])
            step_out = env.step(act)
            if isinstance(step_out, (tuple, list)) and len(step_out) >= 4:
                obs = step_out[0]
                terminated = bool(step_out[2])
                truncated = bool(step_out[3])
                done = terminated or truncated
            else:
                obs, _, done, _ = step_out
            raw = env.render()
            f = self._to_hwc_uint8(raw)
            if f.shape[0] != target_h or f.shape[1] != target_w:
                f = f[:: max(1, f.shape[0] // target_h), :: max(1, f.shape[1] // target_w)]
                f = f[:target_h, :target_w]
            frames.append(f)
            steps += 1

        env.close()
        os.makedirs(self.video_dir, exist_ok=True)
        path = os.path.join(self.video_dir, f"step_{step:05d}.gif")
        imageio.mimsave(path, frames, duration=1 / self.config.get("fps", 15))
        print(f"[eval] Saved gif to {path}")

    # ------------------------------------------------------------------
    def _save_model(self) -> None:
        os.makedirs("models", exist_ok=True)
        path = os.path.join("models", f"{self.run_id}.pt")
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    # ------------------------------------------------------------------
    def close(self) -> None:
        for env in self.envs:
            env.close()
        self.writer.close()

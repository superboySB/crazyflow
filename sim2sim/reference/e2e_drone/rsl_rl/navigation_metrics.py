"""Evaluation-only navigation measurements taken before automatic episode resets."""

from __future__ import annotations

import math
import statistics

import torch


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def _distribution(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values) if values else None,
        "median": _percentile(values, 0.5),
        "p10": _percentile(values, 0.1),
        "p90": _percentile(values, 0.9),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


class NavigationMetrics:
    """Observe a task without changing its rewards, actions, resets, or termination rules."""

    def __init__(self, task_env, episodes_per_env: int):
        if episodes_per_env <= 0:
            raise ValueError("episodes_per_env must be positive")
        self.env = task_env
        self.episodes_per_env = episodes_per_env
        self.dt = float(task_env.step_dt)
        self.episodes: list[dict] = []
        self._started = False
        self._snapshot = None

    def start(self):
        """Start after resetting the environment and before its first evaluation step."""
        if self._started:
            raise RuntimeError("Navigation measurements already started")
        self.episodes.clear()
        positions = self.env._robot.data.root_pos_w.detach().clone()
        self._previous_positions = positions
        self._counts = torch.zeros(self.env.num_envs, dtype=torch.long, device=positions.device)
        self._steps = torch.zeros_like(self._counts)
        self._paths = torch.zeros_like(positions[:, 0])
        self._speed_integrals = torch.zeros_like(self._paths)
        self._max_speeds = torch.zeros_like(self._paths)
        self._start_distances = torch.linalg.vector_norm(self.env._desired_pos_w - positions, dim=1)
        self._commanded_speeds = self.env._desired_speed_init.detach().reshape(-1).clone()
        self._episode_speeds = [[] for _ in range(self.env.num_envs)]
        self._episode_cruise_speeds = [[] for _ in range(self.env.num_envs)]
        self._speed_samples: list[float] = []
        self._cruise_speed_samples: list[float] = []
        self._snapshot = None
        self._had_instance_method = "_get_dones" in vars(self.env)
        self._original_dones = self.env._get_dones

        def capture_dones():
            terminated, timeout = self._original_dones()
            data = self.env._robot.data
            self._snapshot = {
                "positions": data.root_pos_w.detach().clone(),
                "speeds": torch.linalg.vector_norm(data.root_lin_vel_w, dim=1).detach(),
                "goal_distances": torch.linalg.vector_norm(self.env._desired_pos_w - data.root_pos_w, dim=1).detach(),
                "terminated": terminated.clone(),
                "timeout": timeout.clone(),
                "success": self.env._is_success.clone(),
                "collision": self.env._is_contact.clone(),
                "numerical_instability": self.env._numerical_instability.clone(),
            }
            return terminated, timeout

        self.env._get_dones = capture_dones
        self._started = True

    def update(self, active_mask):
        """Consume one step before the evaluator updates its episode quotas."""
        if self._snapshot is None:
            raise RuntimeError("No completed environment step to measure")
        sample = self._snapshot
        self._snapshot = None
        active = active_mask.to(device=self._counts.device, dtype=torch.bool)
        active = active & (self._counts < self.episodes_per_env)
        displacement = torch.linalg.vector_norm(sample["positions"] - self._previous_positions, dim=1)
        self._steps[active] += 1
        self._paths[active] += displacement[active]
        self._speed_integrals[active] += sample["speeds"][active] * self.dt
        self._max_speeds[active] = torch.maximum(self._max_speeds[active], sample["speeds"][active])

        active_ids = active.nonzero(as_tuple=False).flatten().tolist()
        speeds = sample["speeds"][active].cpu().tolist()
        cruise = (sample["goal_distances"][active] > 1.0).cpu().tolist()
        for env_id, speed, is_cruising in zip(active_ids, speeds, cruise):
            self._episode_speeds[env_id].append(speed)
            if is_cruising:
                self._episode_cruise_speeds[env_id].append(speed)

        completed = active & (sample["terminated"] | sample["timeout"])
        completed_ids = completed.nonzero(as_tuple=False).flatten().tolist()
        for env_id in completed_ids:
            self._finish_episode(env_id, sample)

        # The task has already reset terminated drones. Seed the next displacement
        # from their new positions so that reset teleports are never counted.
        positions = self.env._robot.data.root_pos_w.detach()
        self._previous_positions.copy_(positions)
        self._steps[completed] = 0
        self._paths[completed] = 0
        self._speed_integrals[completed] = 0
        self._max_speeds[completed] = 0
        self._start_distances[completed] = torch.linalg.vector_norm(
            self.env._desired_pos_w[completed] - positions[completed], dim=1
        )
        self._commanded_speeds[completed] = self.env._desired_speed_init[completed].reshape(-1)

    def _finish_episode(self, env_id: int, sample: dict):
        flags = {name: bool(sample[name][env_id]) for name in (
            "success", "collision", "numerical_instability", "timeout"
        )}
        outcome = next((name for name, value in flags.items() if value), "other_death")
        duration = int(self._steps[env_id]) * self.dt
        self.episodes.append({
            "env_id": env_id,
            "episode_index": int(self._counts[env_id]) + 1,
            **flags,
            "outcome": outcome,
            "duration_s": duration,
            "path_length_m": float(self._paths[env_id]),
            "mean_speed_m_s": float(self._speed_integrals[env_id]) / duration,
            "max_speed_m_s": float(self._max_speeds[env_id]),
            "start_goal_distance_m": float(self._start_distances[env_id]),
            "commanded_speed_m_s": float(self._commanded_speeds[env_id]),
        })
        self._counts[env_id] += 1
        self._speed_samples.extend(self._episode_speeds[env_id])
        self._cruise_speed_samples.extend(self._episode_cruise_speeds[env_id])
        self._episode_speeds[env_id].clear()
        self._episode_cruise_speeds[env_id].clear()

    def close(self):
        """Restore the task method even if evaluation was interrupted."""
        if not self._started:
            return
        if self._had_instance_method:
            self.env._get_dones = self._original_dones
        else:
            del self.env._get_dones
        self._started = False
        self._snapshot = None

    def summary(self) -> dict:
        """Return JSON-compatible statistics for completed, accepted episodes only."""
        count = len(self.episodes)
        outcomes = {name: sum(row["outcome"] == name for row in self.episodes) for name in (
            "success", "collision", "numerical_instability", "timeout", "other_death"
        )}
        result = {
            "num_episodes": count,
            "num_envs": self.env.num_envs,
            "episodes_per_env": self.episodes_per_env,
            "outcome_counts": outcomes,
            "sampling_frequency_hz": 1.0 / self.dt,
        }
        for name in ("success", "collision", "numerical_instability", "timeout"):
            total = sum(row[name] for row in self.episodes)
            result[name + "_count"] = total
            result[name + "_rate"] = total / count if count else 0.0
        result["speed_m_s"] = self._speed_distribution(self._speed_samples)
        result["cruising_speed_m_s"] = self._speed_distribution(self._cruise_speed_samples)
        for name in ("path_length_m", "duration_s", "start_goal_distance_m", "commanded_speed_m_s"):
            values = [row[name] for row in self.episodes]
            result[name] = _distribution(values)
            result[name]["total"] = sum(values)
        result["definitions"] = {
            "accepted_episodes": "Completed episodes within each environment's quota; unfinished episodes are excluded.",
            "speed": "3D linear speed sampled before reset; time-weighted over accepted episode steps.",
            "cruising_speed": "Speed samples with more than 1 m remaining to the current goal.",
            "path_length": "3D displacement summed at the control frequency; reset teleports excluded.",
            "collision": "Any task-detected body contact; raw collision rate includes simultaneous success.",
            "outcome": "Exclusive priority: success, collision, numerical instability, timeout, other death.",
            "episode_index": "One-based episode number within each environment.",
        }
        return result

    @staticmethod
    def _speed_distribution(values: list[float]) -> dict:
        return {
            "samples": len(values),
            "time_weighted_mean": statistics.fmean(values) if values else None,
            "median": _percentile(values, 0.5),
            "p95": _percentile(values, 0.95),
            "max": max(values) if values else None,
        }

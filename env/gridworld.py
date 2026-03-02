import numpy as np
from typing import List, Tuple, Dict, Optional


class GridWorld:
    def __init__(
        self,
        grid_size: int = 10,
        num_agents: int = 2,
        max_steps: int = 100,
        resource_spawn_rate: int = 10,
        max_resources: int = 5,
        shared_reward: bool = False,
    ):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.resource_spawn_rate = resource_spawn_rate
        self.max_resources = max_resources
        self.shared_reward = shared_reward

        self.agent_positions: List[np.ndarray] = []
        self.resources: List[np.ndarray] = []
        self.step_count = 0

    def reset(self) -> List[np.ndarray]:
        self.step_count = 0
        self.agent_positions = []
        occupied = set()

        for _ in range(self.num_agents):
            while True:
                pos = np.array([
                    np.random.randint(0, self.grid_size),
                    np.random.randint(0, self.grid_size),
                ])
                key = (pos[0], pos[1])
                if key not in occupied:
                    occupied.add(key)
                    self.agent_positions.append(pos)
                    break

        self.resources = []
        self._spawn_resources()

        return self._get_observations()

    def _spawn_resources(self):
        occupied = set()
        for pos in self.agent_positions:
            occupied.add((pos[0], pos[1]))
        for pos in self.resources:
            occupied.add((pos[0], pos[1]))

        while len(self.resources) < self.max_resources:
            pos = np.array([
                np.random.randint(0, self.grid_size),
                np.random.randint(0, self.grid_size),
            ])
            key = (pos[0], pos[1])
            if key not in occupied:
                occupied.add(key)
                self.resources.append(pos)

    def _get_observations(self) -> List[np.ndarray]:
        observations = []
        for i in range(self.num_agents):
            obs = self._get_agent_observation(i)
            observations.append(obs)
        return observations

    def _get_agent_observation(self, agent_idx: int) -> np.ndarray:
        own_pos = self.agent_positions[agent_idx].astype(np.float32)
        other_idx = 1 - agent_idx
        other_pos = self.agent_positions[other_idx].astype(np.float32)

        resource_obs = np.zeros(self.max_resources * 2, dtype=np.float32)
        for r_idx, r_pos in enumerate(self.resources):
            resource_obs[r_idx * 2] = r_pos[0]
            resource_obs[r_idx * 2 + 1] = r_pos[1]

        obs = np.concatenate([own_pos, other_pos, resource_obs])
        return obs

    @property
    def obs_dim(self) -> int:
        return 2 + 2 + self.max_resources * 2

    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], bool, Dict]:
        self.step_count += 1
        rewards = [0.0] * self.num_agents

        action_map = {
            0: np.array([0, 0]),
            1: np.array([-1, 0]),
            2: np.array([1, 0]),
            3: np.array([0, -1]),
            4: np.array([0, 1]),
        }

        new_positions = []
        for i in range(self.num_agents):
            delta = action_map[actions[i]]
            new_pos = self.agent_positions[i] + delta
            new_pos = np.clip(new_pos, 0, self.grid_size - 1)
            new_positions.append(new_pos)

        if np.array_equal(new_positions[0], new_positions[1]):
            new_positions = [pos.copy() for pos in self.agent_positions]

        self.agent_positions = new_positions

        for i in range(self.num_agents):
            collected = []
            for r_idx, r_pos in enumerate(self.resources):
                if np.array_equal(self.agent_positions[i], r_pos):
                    collected.append(r_idx)
                    rewards[i] += 1.0
            for idx in sorted(collected, reverse=True):
                self.resources.pop(idx)

        if self.shared_reward:
            total = sum(rewards)
            rewards = [total] * self.num_agents

        if self.step_count % self.resource_spawn_rate == 0:
            self._spawn_resources()

        done = self.step_count >= self.max_steps
        observations = self._get_observations()

        info = {
            "resources_collected": sum(1 for r in rewards if r > 0),
            "step": self.step_count,
        }

        return observations, rewards, done, info

    def get_state(self) -> Dict:
        return {
            "agent_positions": [pos.copy() for pos in self.agent_positions],
            "resources": [pos.copy() for pos in self.resources],
            "step": self.step_count,
        }

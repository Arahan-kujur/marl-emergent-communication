import numpy as np
from typing import List, Tuple, Dict, Optional


class GridWorld:
    """
    Cooperative gridworld where agents collect shared resources.
    Partial observability: each agent only sees within a limited vision radius.
    Communication becomes valuable when agents can't see each other or distant resources.
    """

    def __init__(
        self,
        grid_size: int = 10,
        num_agents: int = 2,
        max_steps: int = 100,
        resource_spawn_rate: int = 10,
        max_resources: int = 5,
        shared_reward: bool = True,
        vision_radius: int = 3,
    ):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.resource_spawn_rate = resource_spawn_rate
        self.max_resources = max_resources
        self.shared_reward = shared_reward
        self.vision_radius = vision_radius

        self.agent_positions: List[np.ndarray] = []
        self.resources: List[np.ndarray] = []
        self.step_count = 0
        self._coordination_events = 0
        self._total_collections = 0
        self._duplicate_targets = 0

    def reset(self) -> List[np.ndarray]:
        self.step_count = 0
        self._coordination_events = 0
        self._total_collections = 0
        self._duplicate_targets = 0

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

    def _in_vision(self, agent_pos: np.ndarray, target_pos: np.ndarray) -> bool:
        return np.max(np.abs(agent_pos - target_pos)) <= self.vision_radius

    def _get_observations(self) -> List[np.ndarray]:
        observations = []
        for i in range(self.num_agents):
            obs = self._get_agent_observation(i)
            observations.append(obs)
        return observations

    def _get_agent_observation(self, agent_idx: int) -> np.ndarray:
        own_pos = self.agent_positions[agent_idx].astype(np.float32) / self.grid_size

        other_idx = 1 - agent_idx
        other_pos_raw = self.agent_positions[other_idx]
        if self._in_vision(self.agent_positions[agent_idx], other_pos_raw):
            other_pos = other_pos_raw.astype(np.float32) / self.grid_size
            other_visible = 1.0
        else:
            other_pos = np.zeros(2, dtype=np.float32)
            other_visible = 0.0

        resource_obs = np.zeros(self.max_resources * 3, dtype=np.float32)
        visible_count = 0
        for r_pos in self.resources:
            if visible_count >= self.max_resources:
                break
            if self._in_vision(self.agent_positions[agent_idx], r_pos):
                offset = visible_count * 3
                resource_obs[offset] = r_pos[0] / self.grid_size
                resource_obs[offset + 1] = r_pos[1] / self.grid_size
                resource_obs[offset + 2] = 1.0  # visibility flag
                visible_count += 1

        obs = np.concatenate([
            own_pos,                          # 2: own position (normalized)
            other_pos,                        # 2: other agent position (if visible)
            np.array([other_visible]),        # 1: whether other agent is visible
            resource_obs,                     # max_resources * 3: (x, y, visible) per slot
        ])
        return obs

    @property
    def obs_dim(self) -> int:
        return 2 + 2 + 1 + self.max_resources * 3

    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], bool, Dict]:
        self.step_count += 1
        rewards = [0.0] * self.num_agents

        action_map = {
            0: np.array([0, 0]),   # stay
            1: np.array([-1, 0]),  # up
            2: np.array([1, 0]),   # down
            3: np.array([0, -1]),  # left
            4: np.array([0, 1]),   # right
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

        step_collected_by = {}
        for i in range(self.num_agents):
            for r_idx, r_pos in enumerate(self.resources):
                if np.array_equal(self.agent_positions[i], r_pos):
                    if r_idx not in step_collected_by:
                        step_collected_by[r_idx] = []
                    step_collected_by[r_idx].append(i)

        collected_indices = sorted(step_collected_by.keys(), reverse=True)
        for r_idx in collected_indices:
            agents_collecting = step_collected_by[r_idx]
            rewards[agents_collecting[0]] += 1.0
            self.resources.pop(r_idx)
            self._total_collections += 1

        if len(collected_indices) > 0:
            collecting_agents = set()
            for agents in step_collected_by.values():
                collecting_agents.update(agents)
            if len(collecting_agents) > 1 and len(collected_indices) >= 2:
                self._coordination_events += 1

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

    def get_episode_stats(self) -> Dict:
        return {
            "total_collections": self._total_collections,
            "coordination_events": self._coordination_events,
        }

    def get_state(self) -> Dict:
        return {
            "agent_positions": [pos.copy() for pos in self.agent_positions],
            "resources": [pos.copy() for pos in self.resources],
            "step": self.step_count,
        }

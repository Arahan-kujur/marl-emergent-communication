import os
import csv
from typing import Dict, Optional

from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir: str = "runs", csv_path: Optional[str] = None):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.episode = 0

        self.csv_path = csv_path
        self._csv_file = None
        self._csv_writer = None
        self._csv_fields = [
            "episode",
            "episode_reward",
            "resources_collected",
            "coordination_events",
            "coordination_rate",
            "communication_entropy",
            "message_diversity",
            "episode_length",
            "policy_loss",
            "value_loss",
            "entropy",
        ]
        if csv_path:
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            self._csv_file = open(csv_path, "w", newline="")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._csv_fields)
            self._csv_writer.writeheader()

    def log_episode(
        self,
        episode_reward: float,
        resources_collected: int,
        communication_entropy: float,
        episode_length: int,
        coordination_events: int = 0,
        coordination_rate: float = 0.0,
        message_diversity: float = 0.0,
        policy_loss: float = 0.0,
        value_loss: float = 0.0,
        entropy: float = 0.0,
    ):
        self.writer.add_scalar("episode_reward", episode_reward, self.episode)
        self.writer.add_scalar("resources_collected", resources_collected, self.episode)
        self.writer.add_scalar("communication_entropy", communication_entropy, self.episode)
        self.writer.add_scalar("episode_length", episode_length, self.episode)
        self.writer.add_scalar("coordination_events", coordination_events, self.episode)
        self.writer.add_scalar("coordination_rate", coordination_rate, self.episode)
        self.writer.add_scalar("message_diversity", message_diversity, self.episode)
        self.writer.add_scalar("policy_loss", policy_loss, self.episode)
        self.writer.add_scalar("value_loss", value_loss, self.episode)

        if self._csv_writer is not None:
            self._csv_writer.writerow({
                "episode": self.episode,
                "episode_reward": round(episode_reward, 4),
                "resources_collected": resources_collected,
                "coordination_events": coordination_events,
                "coordination_rate": round(coordination_rate, 4),
                "communication_entropy": round(communication_entropy, 4),
                "message_diversity": round(message_diversity, 4),
                "episode_length": episode_length,
                "policy_loss": round(policy_loss, 6),
                "value_loss": round(value_loss, 6),
                "entropy": round(entropy, 6),
            })
            self._csv_file.flush()

        self.episode += 1

    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        if step is None:
            step = self.episode
        self.writer.add_scalar(tag, value, step)

    def close(self):
        self.writer.close()
        if self._csv_file is not None:
            self._csv_file.close()

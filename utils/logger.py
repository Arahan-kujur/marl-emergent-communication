from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional
import os


class Logger:
    def __init__(self, log_dir: str = "runs"):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.episode = 0

    def log_episode(
        self,
        episode_reward: float,
        resources_collected: int,
        communication_entropy: float,
        episode_length: int,
    ):
        self.writer.add_scalar("episode_reward", episode_reward, self.episode)
        self.writer.add_scalar("resources_collected", resources_collected, self.episode)
        self.writer.add_scalar("communication_entropy", communication_entropy, self.episode)
        self.writer.add_scalar("episode_length", episode_length, self.episode)
        self.episode += 1

    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        if step is None:
            step = self.episode
        self.writer.add_scalar(tag, value, step)

    def close(self):
        self.writer.close()

import numpy as np
import torch
from typing import Dict, Optional

from env.gridworld import GridWorld
from agents.agent import AgentNetwork
from communication.channel import CommunicationChannel
from training.ppo import PPO, RolloutBuffer
from utils.logger import Logger
from analysis.metrics import compute_communication_entropy


class Trainer:
    def __init__(self, config: Dict):
        self.config = config

        self.comm_mode = config.get("communication_mode", "no_comm")
        self.resource_spawn_rate = config.get("resource_spawn_rate", 10)
        self.shared_reward = config.get("shared_reward", False)
        self.num_episodes = config.get("num_episodes", 1000)
        self.max_steps = config.get("max_steps", 100)
        self.grid_size = config.get("grid_size", 10)
        self.num_agents = config.get("num_agents", 2)
        self.max_resources = config.get("max_resources", 5)

        self.env = GridWorld(
            grid_size=self.grid_size,
            num_agents=self.num_agents,
            max_steps=self.max_steps,
            resource_spawn_rate=self.resource_spawn_rate,
            max_resources=self.max_resources,
            shared_reward=self.shared_reward,
        )

        self.channel = CommunicationChannel(
            vocab_size=8,
            message_length=self._get_message_length(),
            comm_mode=self.comm_mode,
        )

        self.network = AgentNetwork(
            obs_dim=self.env.obs_dim,
            num_actions=5,
            vocab_size=8,
            message_length=self.channel.message_length,
            comm_enabled=self.channel.enabled,
        )

        self.ppo = PPO(
            network=self.network,
            learning_rate=config.get("learning_rate", 3e-4),
            gamma=config.get("gamma", 0.99),
            gae_lambda=config.get("gae_lambda", 0.95),
            clip_ratio=config.get("clip_ratio", 0.2),
            update_epochs=config.get("update_epochs", 4),
            batch_size=config.get("batch_size", 256),
            comm_enabled=self.channel.enabled,
        )

        log_dir = config.get("log_dir", "runs")
        self.logger = Logger(log_dir=f"{log_dir}/{self.comm_mode}")

    def _get_message_length(self) -> int:
        if self.comm_mode == "no_comm":
            return 0
        elif self.comm_mode == "limited_comm":
            return 1
        elif self.comm_mode == "bandwidth_comm":
            return 3
        return 0

    def train(self):
        print(f"Starting training with communication mode: {self.comm_mode}")
        print(f"Number of episodes: {self.num_episodes}")

        for episode in range(self.num_episodes):
            buffers, episode_info = self._run_episode()

            update_info = self.ppo.update(buffers)

            self.logger.log_episode(
                episode_reward=episode_info["total_reward"],
                resources_collected=episode_info["resources_collected"],
                communication_entropy=episode_info["comm_entropy"],
                episode_length=episode_info["episode_length"],
            )

            if episode % 10 == 0:
                print(
                    f"Episode {episode} | "
                    f"Reward: {episode_info['total_reward']:.2f} | "
                    f"Resources: {episode_info['resources_collected']} | "
                    f"Comm Entropy: {episode_info['comm_entropy']:.4f} | "
                    f"Policy Loss: {update_info['policy_loss']:.4f}"
                )

        self.logger.close()
        print("Training complete.")

    def _run_episode(self):
        observations = self.env.reset()
        buffers = [RolloutBuffer() for _ in range(self.num_agents)]

        total_reward = 0.0
        resources_collected = 0
        all_messages = []

        prev_messages = [self.channel.get_empty_message() for _ in range(self.num_agents)]

        for step in range(self.max_steps):
            received_messages = self.channel.exchange_messages(prev_messages)

            actions = []
            new_messages = []

            for i in range(self.num_agents):
                obs_tensor = torch.tensor(observations[i], dtype=torch.float32).unsqueeze(0)

                msg_tensor = None
                if self.channel.enabled and len(received_messages[i]) > 0:
                    msg_tensor = torch.tensor(received_messages[i], dtype=torch.long).unsqueeze(0)

                action, log_prob, value, msg_tokens, msg_log_probs = self.network.get_action(
                    obs_tensor, msg_tensor,
                )

                actions.append(action)

                if msg_tokens is not None:
                    new_messages.append(msg_tokens)
                    all_messages.append(msg_tokens)
                else:
                    new_messages.append(self.channel.get_empty_message())

                buffers[i].add(
                    obs=observations[i],
                    action=action,
                    log_prob=log_prob,
                    reward=0.0,
                    value=value,
                    done=False,
                    message_received=received_messages[i] if self.channel.enabled else None,
                    message_sent=msg_tokens,
                    message_log_prob=msg_log_probs,
                )

            prev_messages = new_messages

            next_observations, rewards, done, info = self.env.step(actions)

            for i in range(self.num_agents):
                buffers[i].rewards[-1] = rewards[i]
                buffers[i].dones[-1] = done

            total_reward += sum(rewards)
            resources_collected += info["resources_collected"]
            observations = next_observations

            if done:
                break

        comm_entropy = 0.0
        if self.channel.enabled and len(all_messages) > 0:
            comm_entropy = compute_communication_entropy(all_messages, self.channel.vocab_size)

        episode_info = {
            "total_reward": total_reward,
            "resources_collected": resources_collected,
            "comm_entropy": comm_entropy,
            "episode_length": self.env.step_count,
        }

        return buffers, episode_info

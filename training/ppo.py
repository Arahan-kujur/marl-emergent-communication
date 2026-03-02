import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Optional, Tuple


class RolloutBuffer:
    def __init__(self):
        self.observations: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []
        self.messages_received: List[Optional[np.ndarray]] = []
        self.messages_sent: List[Optional[np.ndarray]] = []
        self.message_log_probs: List[Optional[torch.Tensor]] = []

    def add(
        self,
        obs: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
        message_received: Optional[np.ndarray] = None,
        message_sent: Optional[np.ndarray] = None,
        message_log_prob: Optional[torch.Tensor] = None,
    ):
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.messages_received.append(message_received)
        self.messages_sent.append(message_sent)
        self.message_log_probs.append(message_log_prob)

    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.messages_received.clear()
        self.messages_sent.clear()
        self.message_log_probs.clear()

    def __len__(self):
        return len(self.observations)


def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    next_value: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    returns = np.zeros(n, dtype=np.float32)

    gae = 0.0
    for t in reversed(range(n)):
        if t == n - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]

        next_non_terminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]

    return advantages, returns


class PPO:
    def __init__(
        self,
        network: nn.Module,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        update_epochs: int = 4,
        batch_size: int = 256,
        comm_enabled: bool = False,
    ):
        self.network = network
        self.optimizer = optim.Adam(network.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.comm_enabled = comm_enabled

    def update(self, buffers: List[RolloutBuffer]) -> Dict[str, float]:
        all_obs = []
        all_actions = []
        all_old_log_probs = []
        all_advantages = []
        all_returns = []
        all_messages_received = []
        all_messages_sent = []

        for buffer in buffers:
            advantages, returns = compute_gae(
                buffer.rewards,
                buffer.values,
                buffer.dones,
                self.gamma,
                self.gae_lambda,
            )

            all_obs.extend(buffer.observations)
            all_actions.extend(buffer.actions)
            all_old_log_probs.extend(buffer.log_probs)
            all_advantages.append(advantages)
            all_returns.append(returns)
            all_messages_received.extend(buffer.messages_received)
            all_messages_sent.extend(buffer.messages_sent)

        all_advantages = np.concatenate(all_advantages)
        all_returns = np.concatenate(all_returns)

        adv_mean = all_advantages.mean()
        adv_std = all_advantages.std() + 1e-8
        all_advantages = (all_advantages - adv_mean) / adv_std

        obs_tensor = torch.tensor(np.array(all_obs), dtype=torch.float32)
        actions_tensor = torch.tensor(all_actions, dtype=torch.long)
        old_log_probs_tensor = torch.tensor(all_old_log_probs, dtype=torch.float32)
        advantages_tensor = torch.tensor(all_advantages, dtype=torch.float32)
        returns_tensor = torch.tensor(all_returns, dtype=torch.float32)

        msg_recv_tensor = None
        msg_sent_tensor = None
        if self.comm_enabled:
            has_messages = any(m is not None and len(m) > 0 for m in all_messages_received)
            if has_messages:
                msg_recv_tensor = torch.tensor(
                    np.array([m if m is not None else np.zeros(1, dtype=np.int64) for m in all_messages_received]),
                    dtype=torch.long,
                )
                msg_sent_tensor = torch.tensor(
                    np.array([m if m is not None else np.zeros(1, dtype=np.int64) for m in all_messages_sent]),
                    dtype=torch.long,
                )

        n = len(all_obs)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for _ in range(self.update_epochs):
            indices = np.random.permutation(n)

            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                batch_idx = indices[start:end]

                batch_obs = obs_tensor[batch_idx]
                batch_actions = actions_tensor[batch_idx]
                batch_old_log_probs = old_log_probs_tensor[batch_idx]
                batch_advantages = advantages_tensor[batch_idx]
                batch_returns = returns_tensor[batch_idx]

                batch_msg_recv = None
                batch_msg_sent = None
                if msg_recv_tensor is not None:
                    batch_msg_recv = msg_recv_tensor[batch_idx]
                    batch_msg_sent = msg_sent_tensor[batch_idx]

                new_log_probs, values, entropy, msg_log_probs = self.network.evaluate_actions(
                    batch_obs, batch_actions, batch_msg_recv, batch_msg_sent,
                )

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(values, batch_returns)

                entropy_loss = -entropy.mean()

                loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

                if msg_log_probs is not None:
                    msg_policy_loss = -msg_log_probs.mean()
                    loss = loss + 0.1 * msg_policy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        return {
            "policy_loss": total_policy_loss / max(num_updates, 1),
            "value_loss": total_value_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
        }

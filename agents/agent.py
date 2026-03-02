import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Tuple, Optional


class AgentNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        num_actions: int = 5,
        vocab_size: int = 8,
        message_length: int = 0,
        comm_enabled: bool = False,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.vocab_size = vocab_size
        self.message_length = message_length
        self.comm_enabled = comm_enabled

        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
        )

        if comm_enabled and message_length > 0:
            self.comm_embedding = nn.Embedding(vocab_size, 16)
            self.comm_encoder = nn.Sequential(
                nn.Linear(message_length * 16, 32),
                nn.ReLU(),
            )
            combined_dim = 64 + 32
        else:
            self.comm_embedding = None
            self.comm_encoder = None
            combined_dim = 64

        self.policy_head = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

        self.value_head = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        if comm_enabled and message_length > 0:
            self.message_head = nn.Linear(combined_dim, message_length * vocab_size)
        else:
            self.message_head = None

    def forward(
        self,
        obs: torch.Tensor,
        message: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        obs_enc = self.obs_encoder(obs)

        if self.comm_enabled and self.comm_encoder is not None and message is not None:
            msg_embedded = self.comm_embedding(message)
            msg_flat = msg_embedded.reshape(msg_embedded.shape[0], -1)
            comm_enc = self.comm_encoder(msg_flat)
            combined = torch.cat([obs_enc, comm_enc], dim=-1)
        else:
            combined = obs_enc

        action_logits = self.policy_head(combined)
        value = self.value_head(combined).squeeze(-1)

        message_logits = None
        if self.message_head is not None:
            message_logits = self.message_head(combined)

        return action_logits, value, message_logits

    def get_action(
        self,
        obs: torch.Tensor,
        message: Optional[torch.Tensor] = None,
    ) -> Tuple[int, float, float, Optional[np.ndarray], Optional[torch.Tensor]]:
        with torch.no_grad():
            action_logits, value, message_logits = self.forward(obs, message)

        dist = Categorical(logits=action_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        msg_tokens = None
        msg_log_probs = None
        if message_logits is not None:
            msg_tokens_list = []
            msg_log_probs_list = []
            for t in range(self.message_length):
                token_logits = message_logits[:, t * self.vocab_size:(t + 1) * self.vocab_size]
                token_dist = Categorical(logits=token_logits)
                token = token_dist.sample()
                token_lp = token_dist.log_prob(token)
                msg_tokens_list.append(token.item())
                msg_log_probs_list.append(token_lp)
            msg_tokens = np.array(msg_tokens_list, dtype=np.int64)
            msg_log_probs = torch.stack(msg_log_probs_list, dim=-1)

        return (
            action.item(),
            log_prob.item(),
            value.item(),
            msg_tokens,
            msg_log_probs,
        )

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        messages_received: Optional[torch.Tensor] = None,
        messages_sent: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        action_logits, values, message_logits = self.forward(obs, messages_received)

        dist = Categorical(logits=action_logits)
        action_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        msg_log_probs = None
        if message_logits is not None and messages_sent is not None:
            msg_log_probs_list = []
            for t in range(self.message_length):
                token_logits = message_logits[:, t * self.vocab_size:(t + 1) * self.vocab_size]
                token_dist = Categorical(logits=token_logits)
                token_lp = token_dist.log_prob(messages_sent[:, t])
                msg_log_probs_list.append(token_lp)
            msg_log_probs = torch.stack(msg_log_probs_list, dim=-1).sum(dim=-1)

        return action_log_probs, values, entropy, msg_log_probs

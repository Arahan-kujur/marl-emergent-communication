import numpy as np
import torch
from typing import List, Optional


class CommunicationChannel:
    def __init__(
        self,
        vocab_size: int = 8,
        message_length: int = 1,
        comm_mode: str = "no_comm",
    ):
        self.vocab_size = vocab_size
        self.message_length = message_length
        self.comm_mode = comm_mode

        if comm_mode == "no_comm":
            self.enabled = False
            self.message_length = 0
        elif comm_mode == "limited_comm":
            self.enabled = True
            self.message_length = 1
        elif comm_mode == "bandwidth_comm":
            self.enabled = True
            self.message_length = 3
        else:
            raise ValueError(f"Unknown communication mode: {comm_mode}")

    def encode_message(self, message_logits: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return torch.zeros(0, dtype=torch.long)

        if message_logits.dim() == 1:
            message_logits = message_logits.unsqueeze(0)

        messages = []
        for t in range(self.message_length):
            token_logits = message_logits[:, t * self.vocab_size:(t + 1) * self.vocab_size]
            probs = torch.softmax(token_logits, dim=-1)
            token = torch.multinomial(probs, 1).squeeze(-1)
            messages.append(token)

        return torch.stack(messages, dim=-1)

    def get_empty_message(self) -> np.ndarray:
        if not self.enabled:
            return np.zeros(0, dtype=np.int64)
        return np.zeros(self.message_length, dtype=np.int64)

    def exchange_messages(
        self,
        messages: List[np.ndarray],
    ) -> List[np.ndarray]:
        if not self.enabled:
            return [self.get_empty_message() for _ in messages]

        num_agents = len(messages)
        received = []
        for i in range(num_agents):
            other_idx = 1 - i
            received.append(messages[other_idx].copy())

        return received

    @property
    def comm_dim(self) -> int:
        if not self.enabled:
            return 0
        return self.message_length

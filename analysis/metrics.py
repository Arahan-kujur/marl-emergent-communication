import numpy as np
from typing import List, Optional


def compute_communication_entropy(
    messages: List[np.ndarray],
    vocab_size: int = 8,
) -> float:
    if len(messages) == 0:
        return 0.0

    all_tokens = []
    for msg in messages:
        for token in msg:
            all_tokens.append(int(token))

    if len(all_tokens) == 0:
        return 0.0

    counts = np.zeros(vocab_size, dtype=np.float64)
    for token in all_tokens:
        if 0 <= token < vocab_size:
            counts[token] += 1

    total = counts.sum()
    if total == 0:
        return 0.0

    probs = counts / total
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * np.log2(p)

    return float(entropy)


def compute_average_reward(episode_rewards: List[float]) -> float:
    if len(episode_rewards) == 0:
        return 0.0
    return float(np.mean(episode_rewards))


def compute_resource_efficiency(
    resources_collected: List[int],
    max_possible: int,
) -> float:
    if max_possible == 0 or len(resources_collected) == 0:
        return 0.0
    return float(np.mean(resources_collected)) / max_possible

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


def smooth(values: List[float], window: int = 25) -> np.ndarray:
    """Simple moving average for smoother curves."""
    if len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def load_csv_log(csv_path: str) -> Dict[str, List[float]]:
    data: Dict[str, List[float]] = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                if key not in data:
                    data[key] = []
                data[key].append(float(val))
    return data


def plot_reward_curves(
    results: Dict[str, Dict[str, List[float]]],
    save_path: str,
    window: int = 25,
):
    """Plot reward vs training episode for each condition."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"no_comm": "#d62728", "limited_comm": "#2ca02c", "bandwidth_comm": "#1f77b4"}
    labels = {"no_comm": "No Communication", "limited_comm": "With Communication (1 token)", "bandwidth_comm": "With Communication (3 tokens)"}

    for mode, data in results.items():
        rewards = data["episode_reward"]
        raw_episodes = list(range(len(rewards)))

        ax.plot(raw_episodes, rewards, alpha=0.15, color=colors.get(mode, "gray"))

        smoothed = smooth(rewards, window)
        smoothed_episodes = list(range(window - 1, window - 1 + len(smoothed)))
        ax.plot(smoothed_episodes, smoothed, linewidth=2.5,
                color=colors.get(mode, "gray"), label=labels.get(mode, mode))

    ax.set_xlabel("Episode", fontsize=13)
    ax.set_ylabel("Episode Reward", fontsize=13)
    ax.set_title("Reward vs Training Episodes", fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved reward curve plot to {save_path}")


def plot_success_rate_comparison(
    results: Dict[str, Dict[str, List[float]]],
    save_path: str,
    threshold: float = 5.0,
    window: int = 50,
):
    """
    Plot success rate (fraction of episodes where collections >= threshold)
    as a rolling average for each condition.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"no_comm": "#d62728", "limited_comm": "#2ca02c", "bandwidth_comm": "#1f77b4"}
    labels = {"no_comm": "No Communication", "limited_comm": "With Communication (1 token)", "bandwidth_comm": "With Communication (3 tokens)"}

    for mode, data in results.items():
        collections = data["resources_collected"]
        successes = [1.0 if c >= threshold else 0.0 for c in collections]

        smoothed = smooth(successes, window)
        episodes = list(range(window - 1, window - 1 + len(smoothed)))
        ax.plot(episodes, smoothed, linewidth=2.5,
                color=colors.get(mode, "gray"), label=labels.get(mode, mode))

    ax.set_xlabel("Episode", fontsize=13)
    ax.set_ylabel(f"Success Rate (>={int(threshold)} resources)", fontsize=13)
    ax.set_title("Coordination Success Rate Comparison", fontsize=15)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved success rate plot to {save_path}")


def plot_message_stats(
    results: Dict[str, Dict[str, List[float]]],
    save_path: str,
    window: int = 25,
):
    """Plot communication entropy and message diversity for comm conditions."""
    comm_results = {k: v for k, v in results.items() if k != "no_comm"}
    if not comm_results:
        print("No communication conditions to plot message stats.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"limited_comm": "#2ca02c", "bandwidth_comm": "#1f77b4"}
    labels = {"limited_comm": "With Comm (1 token)", "bandwidth_comm": "With Comm (3 tokens)"}

    for mode, data in comm_results.items():
        entropy = data.get("communication_entropy", [])
        if entropy:
            smoothed = smooth(entropy, window)
            episodes = list(range(window - 1, window - 1 + len(smoothed)))
            ax1.plot(episodes, smoothed, linewidth=2,
                     color=colors.get(mode, "gray"), label=labels.get(mode, mode))

    ax1.set_xlabel("Episode", fontsize=12)
    ax1.set_ylabel("Communication Entropy (bits)", fontsize=12)
    ax1.set_title("Message Entropy Over Training", fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    for mode, data in comm_results.items():
        diversity = data.get("message_diversity", [])
        if diversity:
            smoothed = smooth(diversity, window)
            episodes = list(range(window - 1, window - 1 + len(smoothed)))
            ax2.plot(episodes, smoothed, linewidth=2,
                     color=colors.get(mode, "gray"), label=labels.get(mode, mode))

    ax2.set_xlabel("Episode", fontsize=12)
    ax2.set_ylabel("Message Diversity (unique / total)", fontsize=12)
    ax2.set_title("Message Diversity Over Training", fontsize=13)
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved message stats plot to {save_path}")


def plot_summary_bar(
    results: Dict[str, Dict[str, List[float]]],
    save_path: str,
    last_n: int = 100,
):
    """Bar chart comparing final performance across conditions."""
    modes = list(results.keys())
    colors = {"no_comm": "#d62728", "limited_comm": "#2ca02c", "bandwidth_comm": "#1f77b4"}
    labels = {"no_comm": "No Comm", "limited_comm": "Comm (1 tok)", "bandwidth_comm": "Comm (3 tok)"}

    avg_rewards = []
    avg_collections = []
    for mode in modes:
        rewards = results[mode]["episode_reward"][-last_n:]
        collections = results[mode]["resources_collected"][-last_n:]
        avg_rewards.append(np.mean(rewards))
        avg_collections.append(np.mean(collections))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(modes))
    bar_colors = [colors.get(m, "gray") for m in modes]
    bar_labels = [labels.get(m, m) for m in modes]

    bars1 = ax1.bar(x, avg_rewards, color=bar_colors, width=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(bar_labels, fontsize=11)
    ax1.set_ylabel("Mean Reward (last 100 ep)", fontsize=12)
    ax1.set_title("Final Reward Comparison", fontsize=13)
    ax1.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars1, avg_rewards):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f"{val:.1f}", ha="center", fontsize=11)

    bars2 = ax2.bar(x, avg_collections, color=bar_colors, width=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(bar_labels, fontsize=11)
    ax2.set_ylabel("Mean Resources (last 100 ep)", fontsize=12)
    ax2.set_title("Final Collection Comparison", fontsize=13)
    ax2.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars2, avg_collections):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f"{val:.1f}", ha="center", fontsize=11)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved summary bar plot to {save_path}")

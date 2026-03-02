import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict, Optional


def render_gridworld(
    grid_size: int,
    agent_positions: List[np.ndarray],
    resources: List[np.ndarray],
    step: int = 0,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    ax.clear()
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    ax.set_title(f"Step {step}")

    colors = ["blue", "red"]
    for i, pos in enumerate(agent_positions):
        ax.plot(pos[1], pos[0], "o", color=colors[i % len(colors)],
                markersize=15, label=f"Agent {i}")

    for pos in resources:
        ax.plot(pos[1], pos[0], "s", color="green", markersize=10)

    ax.legend(loc="upper right")
    return ax


def display_agent_paths(
    grid_size: int,
    episode_states: List[Dict],
    save_path: Optional[str] = None,
):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    ax.set_title("Agent Paths Over Episode")

    colors = ["blue", "red"]
    num_agents = len(episode_states[0]["agent_positions"])

    for agent_idx in range(num_agents):
        path_x = []
        path_y = []
        for state in episode_states:
            pos = state["agent_positions"][agent_idx]
            path_x.append(pos[1])
            path_y.append(pos[0])

        ax.plot(path_x, path_y, "-", color=colors[agent_idx % len(colors)],
                alpha=0.5, linewidth=2, label=f"Agent {agent_idx}")

        ax.plot(path_x[0], path_y[0], "^", color=colors[agent_idx % len(colors)],
                markersize=12)
        ax.plot(path_x[-1], path_y[-1], "v", color=colors[agent_idx % len(colors)],
                markersize=12)

    ax.legend(loc="upper right")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved path visualization to {save_path}")

    plt.close(fig)
    return fig


def render_episode(
    grid_size: int,
    episode_states: List[Dict],
    save_path: Optional[str] = None,
):
    n_steps = len(episode_states)
    cols = min(5, n_steps)
    rows = (n_steps + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for idx, state in enumerate(episode_states):
        r, c = idx // cols, idx % cols
        render_gridworld(
            grid_size=grid_size,
            agent_positions=state["agent_positions"],
            resources=state["resources"],
            step=state["step"],
            ax=axes[r, c],
        )

    for idx in range(n_steps, rows * cols):
        r, c = idx // cols, idx % cols
        axes[r, c].axis("off")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved episode rendering to {save_path}")

    plt.close(fig)
    return fig

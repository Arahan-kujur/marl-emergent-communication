"""
Run the full experiment: train agents with and without communication,
save logs to results/logs/, generate comparison plots in results/plots/.

Usage:
    python run_experiment.py
    python run_experiment.py --episodes 500
    python run_experiment.py --modes no_comm limited_comm bandwidth_comm
"""

import argparse
import os
import sys
import yaml
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.trainer import Trainer
from analysis.plotting import (
    load_csv_log,
    plot_reward_curves,
    plot_success_rate_comparison,
    plot_message_stats,
    plot_summary_bar,
)


RESULTS_DIR = "results"
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")


def load_base_config(mode: str) -> dict:
    config_map = {
        "no_comm": "configs/no_comm.yaml",
        "limited_comm": "configs/with_comm.yaml",
        "bandwidth_comm": "configs/bandwidth_comm.yaml",
    }
    path = config_map.get(mode)
    if path and os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    return {
        "communication_mode": mode,
        "resource_spawn_rate": 10,
        "shared_reward": True,
        "grid_size": 10,
        "num_agents": 2,
        "max_steps": 100,
        "max_resources": 5,
        "vision_radius": 3,
        "num_episodes": 500,
        "learning_rate": 0.0003,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_ratio": 0.2,
        "update_epochs": 4,
        "batch_size": 256,
        "log_dir": "runs",
    }


def run_experiment(modes: list, num_episodes: int = 500, seed: int = 42):
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    import numpy as np
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)

    all_results = {}

    for mode in modes:
        print("\n" + "=" * 70)
        print(f"  EXPERIMENT: {mode}")
        print("=" * 70)

        config = load_base_config(mode)
        config["num_episodes"] = num_episodes
        config["csv_path"] = os.path.join(LOGS_DIR, f"{mode}.csv")
        config["log_dir"] = "runs"

        start_time = time.time()
        trainer = Trainer(config)
        trainer.train()
        elapsed = time.time() - start_time

        print(f"  Finished {mode} in {elapsed:.1f}s")

    print("\n" + "=" * 70)
    print("  GENERATING PLOTS")
    print("=" * 70)

    for mode in modes:
        csv_path = os.path.join(LOGS_DIR, f"{mode}.csv")
        if os.path.exists(csv_path):
            all_results[mode] = load_csv_log(csv_path)
        else:
            print(f"  WARNING: CSV log not found for {mode} at {csv_path}")

    if not all_results:
        print("  No results to plot. Exiting.")
        return

    plot_reward_curves(
        all_results,
        save_path=os.path.join(PLOTS_DIR, "reward_curves.png"),
    )

    plot_success_rate_comparison(
        all_results,
        save_path=os.path.join(PLOTS_DIR, "success_rate.png"),
    )

    plot_message_stats(
        all_results,
        save_path=os.path.join(PLOTS_DIR, "message_stats.png"),
    )

    plot_summary_bar(
        all_results,
        save_path=os.path.join(PLOTS_DIR, "summary_comparison.png"),
    )

    print("\n" + "=" * 70)
    print("  EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"  Logs:  {LOGS_DIR}/")
    print(f"  Plots: {PLOTS_DIR}/")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Run MARL communication experiment")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes per condition")
    parser.add_argument("--modes", nargs="+", default=["no_comm", "limited_comm"],
                        help="Communication modes to compare")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_experiment(modes=args.modes, num_episodes=args.episodes, seed=args.seed)


if __name__ == "__main__":
    main()

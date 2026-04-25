import argparse
import yaml
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Reinforcement Learning with Emergent Communication"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("MARL Emergent Communication Framework")
    print("=" * 60)
    print(f"Config:             {args.config}")
    print(f"Communication mode: {config.get('communication_mode', 'no_comm')}")
    print(f"Shared reward:      {config.get('shared_reward', True)}")
    print(f"Vision radius:      {config.get('vision_radius', 3)}")
    print(f"Episodes:           {config.get('num_episodes', 1000)}")
    print("=" * 60)

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()

# Multi-Agent Reinforcement Learning: Emergent Communication under Resource Scarcity

A framework for studying emergent communication in multi-agent reinforcement learning using PPO in a custom gridworld environment.

## Project Structure

```
project/
├── env/
│   └── gridworld.py          # Custom 10x10 gridworld environment
├── agents/
│   └── agent.py               # Shared neural network agent architecture
├── communication/
│   └── channel.py             # Communication channel with configurable modes
├── training/
│   ├── ppo.py                 # PPO algorithm implementation
│   └── trainer.py             # Training loop orchestrator
├── configs/
│   ├── no_comm.yaml           # No communication baseline
│   ├── limited_comm.yaml      # 1-token communication
│   └── bandwidth_comm.yaml    # 3-token communication
├── analysis/
│   ├── metrics.py             # Communication entropy and reward metrics
│   └── visualize.py           # Gridworld and agent path visualization
├── utils/
│   └── logger.py              # TensorBoard logging
├── main.py                    # Entry point
└── README.md
```

## Requirements

- Python 3.10
- PyTorch
- PyYAML
- TensorBoard
- NumPy
- Matplotlib

Install dependencies:

```bash
pip install torch numpy pyyaml tensorboard matplotlib
```

## Usage

### Training

Run training with a configuration file:

```bash
python main.py --config configs/no_comm.yaml
python main.py --config configs/limited_comm.yaml
python main.py --config configs/bandwidth_comm.yaml
```

### Monitoring

View training metrics with TensorBoard:

```bash
tensorboard --logdir runs/
```

### Communication Modes

| Mode | Message Length | Description |
|------|--------------|-------------|
| `no_comm` | 0 | Baseline without communication |
| `limited_comm` | 1 token | Minimal communication bandwidth |
| `bandwidth_comm` | 3 tokens | Higher communication bandwidth |

### Environment

- 10x10 gridworld with 2 agents
- Resources spawn every 10 steps (max 5 present)
- Episode length: 100 steps
- Reward: +1 per resource collected

### Agent Architecture

- Observation encoder: Linear(obs_dim -> 64) + ReLU
- Communication encoder: Embedding(8 -> 16) + Flatten + Linear(-> 32) + ReLU
- Policy head: Linear(-> 64) + ReLU + Linear(-> 5 actions)
- Value head: Linear(-> 64) + ReLU + Linear(-> 1)
- Parameter sharing across agents

### PPO Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate | 3e-4 |
| Gamma | 0.99 |
| GAE Lambda | 0.95 |
| Clip ratio | 0.2 |
| Update epochs | 4 |
| Batch size | 256 |

## Logged Metrics

- `episode_reward` — Total reward per episode
- `resources_collected` — Number of resources collected per episode
- `communication_entropy` — Shannon entropy of message token distribution
- `episode_length` — Steps per episode

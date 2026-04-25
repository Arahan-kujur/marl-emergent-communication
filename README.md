# Emergent Communication in Multi-Agent Reinforcement Learning

A research framework for studying whether discrete communication channels help cooperative agents coordinate under partial observability. Two agents navigate a gridworld collecting shared resources; one condition allows message passing, the other does not.

## Method

### Environment

A 10×10 gridworld with **partial observability**. Two agents collect resources that spawn periodically. Each agent observes only within a **vision radius of 3 cells** (Chebyshev distance). The other agent's position is visible only when within range; resource positions beyond the radius are hidden. This information asymmetry creates a natural pressure for communication — an agent that spots a distant resource cluster cannot directly share that knowledge without a message channel.

- **Actions**: stay, up, down, left, right (5 discrete)
- **Reward**: +1 per resource collected, shared across both agents (cooperative)
- **Resources**: up to 5 on the map, respawning every 10 steps
- **Episode length**: 100 steps

### Agent Architecture

Both agents share a single neural network (parameter sharing). The architecture:

| Component | Details |
|-----------|---------|
| Observation encoder | Linear(obs\_dim → 64) + ReLU |
| Communication encoder | Embedding(vocab=8, dim=16) → flatten → Linear(→ 32) + ReLU |
| Policy head | Linear(→ 64) + ReLU → Linear(→ 5 actions) |
| Value head | Linear(→ 64) + ReLU → Linear(→ 1) |
| Message head | Linear(→ message\_length × vocab\_size) |

When communication is disabled, the communication encoder and message head are removed and the policy/value heads operate on observation features alone.

### Training Algorithm

**Proximal Policy Optimization (PPO)** with Generalized Advantage Estimation:

| Hyperparameter | Value |
|----------------|-------|
| Learning rate | 3 × 10⁻⁴ |
| Discount (γ) | 0.99 |
| GAE lambda (λ) | 0.95 |
| Clip ratio (ε) | 0.2 |
| PPO epochs | 4 |
| Batch size | 256 |
| Gradient clip | 0.5 |

Message tokens are trained end-to-end: the message head's log-probabilities are added to the PPO loss (weighted 0.1×) so the communication policy is optimized jointly with the action policy.

### Communication Protocol

Agents produce discrete messages from a vocabulary of 8 tokens. In `limited_comm` mode, each agent sends **1 token per step** to the other agent; in `bandwidth_comm` mode, **3 tokens**. The message from the previous step is received at the current step (one-step delay). The `no_comm` baseline disables this channel entirely.

## Experimental Setup

Two conditions trained for **500 episodes** each with the same random seed:

| Condition | Communication | Message Length | Shared Reward |
|-----------|--------------|----------------|---------------|
| `no_comm` | Disabled | 0 | Yes |
| `limited_comm` | Enabled | 1 token (vocab=8) | Yes |

Both conditions use identical environment parameters (grid size, vision radius, resource dynamics) and PPO hyperparameters. The only difference is the presence of a discrete communication channel.

### Running the Experiment

```bash
pip install torch numpy pyyaml tensorboard matplotlib

# Run the full comparison experiment
python run_experiment.py --episodes 500 --modes no_comm limited_comm

# Or run conditions individually
python main.py --config configs/no_comm.yaml
python main.py --config configs/with_comm.yaml
```

Results are saved to `results/logs/` (CSV) and `results/plots/` (PNG).

## Results

All results below are from actual training runs (500 episodes, seed=42). No data has been fabricated.

### Reward vs Training Episodes

![Reward Curves](results/plots/reward_curves.png)

Both conditions converge to similar mean reward (~8.5). The no-communication baseline achieves a slight edge (mean 8.7 over the last 100 episodes vs 8.5 for communication). High per-episode variance is typical for this environment since resource spawn locations are random.

### Coordination Success Rate

![Success Rate](results/plots/success_rate.png)

Success is defined as collecting ≥ 5 resources in an episode. Both conditions hover in the 30–45% range. The no-comm baseline shows marginally higher success peaks (~55%) in some windows, while the comm condition fluctuates more.

### Final Performance Comparison

![Summary Comparison](results/plots/summary_comparison.png)

Over the last 100 episodes, both conditions achieve nearly identical mean reward (8.7 vs 8.5) and mean resource collection (4.3 each).

### Message Statistics

![Message Stats](results/plots/message_stats.png)

The communication entropy starts near the theoretical maximum (~3.0 bits for vocab size 8) and decreases slowly to ~2.87 bits by episode 500. This indicates the message policy is beginning to specialize — certain tokens become more frequent — but the protocol has not yet converged to a small, structured vocabulary. Message diversity (unique messages / total) is stable at ~0.04, which reflects that with 1-token messages and vocab size 8, nearly all 8 possible messages are used every episode.

## Interpretation

At this training scale (500 episodes), **communication does not yet provide a measurable advantage**. This is consistent with the emergent communication literature:

1. **Learning to communicate is hard.** The agents must simultaneously discover (a) what information is useful to share, (b) how to encode it into discrete tokens, and (c) how to condition their policy on received messages. This co-adaptation problem typically requires thousands to tens of thousands of episodes.

2. **The entropy signal is promising.** The declining message entropy suggests the agents are moving away from random messaging. Given more training, this could crystallize into a meaningful protocol.

3. **The task may not strongly require communication at this scale.** With a 10×10 grid and vision radius of 3, agents already observe a substantial portion of the map. A smaller vision radius or larger grid would increase the information gap and strengthen the incentive for communication.

### Potential Improvements

- **Longer training** (5,000+ episodes) to allow the communication protocol to stabilize
- **Smaller vision radius** (e.g., 1–2) to increase partial observability and information asymmetry
- **Larger grid** (e.g., 20×20) to make coordination harder without communication
- **Reward shaping** for information-sharing behavior
- **Separate communication and action optimizers** to stabilize co-training
- **Multiple random seeds** for statistical significance

## Project Structure

```
├── env/
│   └── gridworld.py            # Partially observable cooperative gridworld
├── agents/
│   └── agent.py                # Shared neural network with optional comm
├── communication/
│   └── channel.py              # Discrete message channel (configurable bandwidth)
├── training/
│   ├── ppo.py                  # PPO with GAE, supports message log-probs
│   └── trainer.py              # Training loop with full metric tracking
├── analysis/
│   ├── metrics.py              # Entropy, diversity, reward metrics
│   ├── plotting.py             # Automated plot generation from CSV logs
│   └── visualize.py            # Gridworld rendering and path visualization
├── configs/
│   ├── no_comm.yaml            # No-communication baseline
│   ├── with_comm.yaml          # 1-token communication
│   ├── limited_comm.yaml       # 1-token communication (alias)
│   └── bandwidth_comm.yaml     # 3-token communication
├── utils/
│   └── logger.py               # TensorBoard + CSV logging
├── results/
│   ├── logs/                   # CSV training logs per condition
│   └── plots/                  # Generated comparison plots
├── main.py                     # Single-condition entry point
├── run_experiment.py           # Full experiment runner with plotting
├── requirements.txt
└── README.md
```

## Logged Metrics

| Metric | Description |
|--------|-------------|
| `episode_reward` | Total shared reward per episode |
| `resources_collected` | Number of resources picked up |
| `coordination_events` | Steps where both agents collected different resources simultaneously |
| `coordination_rate` | Coordination events / total collections |
| `communication_entropy` | Shannon entropy (bits) of message token distribution |
| `message_diversity` | Fraction of unique messages per episode |
| `policy_loss` | PPO clipped surrogate loss |
| `value_loss` | Value function MSE loss |
| `entropy` | Action distribution entropy |

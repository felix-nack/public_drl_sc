# Deep Reinforcement Learning for Chemical Production Scheduling

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository implements a **Deep Reinforcement Learning (DRL)** framework for dynamic chemical production scheduling under uncertainty. The work is based on the research published in [Computers & Chemical Engineering](https://www.sciencedirect.com/science/article/pii/S0098135420301599) and was utilized as part of my Bachelor's thesis investigating the application of AI methodologies to production scheduling optimization.

The framework benchmarks DRL approaches against traditional **Mixed-Integer Linear Programming (MILP)** methods, demonstrating competitive performance in terms of profitability, inventory management, and customer service levels, while offering superior computational efficiency for real-time scheduling applications.

## Key Features

- **Multiple Agent Architectures**: Reinforcement Learning (A2C algorithm), Mixed-Integer Programming (MIP), and heuristic-based approaches
- **Production Environment Simulation**: Realistic chemical production facility model with inventory dynamics, order management, and production constraints
- **Flexible Configuration System**: YAML/CSV-based configuration for easy experimentation
- **Comprehensive Evaluation Metrics**: Customer service levels, inventory costs, late penalties, and profitability tracking
- **Demand Modeling**: Multiple demand generation models including seasonal patterns and historical data
- **Forecasting Capabilities**: Integration of demand forecasting for improved decision-making

## Architecture

### Core Components

```
ada/
├── agents/               # Agent implementations
│   ├── rl_agent.py      # Reinforcement learning agents (A2C)
│   ├── opt_agent.py     # Optimization-based agents (MIP)
│   └── heuristic_agent.py # Heuristic agents (Random, rule-based)
├── environments/         # Production environment simulators
│   ├── ppm.py           # Core production planning model
│   ├── tartan.py        # LDPE production facility simulation
│   └── demand_models/   # Customer demand generation
├── scheduler/           # Scheduling logic
│   ├── network_scheduler.py  # Neural network-based scheduler
│   └── mip_scheduler.py      # MILP-based scheduler
└── config.py            # Configuration management system
```

## Methodology

### Reinforcement Learning Approach
- **Algorithm**: Advantage Actor-Critic (A2C)
- **State Space**: Inventory levels, order backlogs, production capacity
- **Action Space**: Product selection for production scheduling
- **Reward Function**: Optimizes trade-off between customer service, inventory costs, and production efficiency

### MILP Benchmarks
- **Deterministic MIP**: Receding horizon optimization
- **Stochastic MIP**: Accounts for demand uncertainty
- **God MIP**: Perfect information benchmark (upper bound)

### Production Environment
The simulation models a chemical production facility with:
- Multi-product production lines
- Inventory management and storage costs
- Order book with delivery deadlines
- Production changeover costs and off-grade material
- Capacity constraints and maintenance schedules

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.4.0
- NumPy, Pandas, Matplotlib

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/public_drl_sc.git
cd public_drl_sc

# Create virtual environment
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training an RL Agent

```bash
python ada/train.py --config path/to/config.csv
```

### Configuration Example

Create a configuration file specifying:
- `AGENT_CLASS`: 'RL', 'MIP', or 'HEURISTIC'
- `ENVIRONMENT`: 'TARTAN' (production facility type)
- `N_PRODUCTS`: Number of products (e.g., 6)
- `START_TIME` / `END_TIME`: Simulation period
- `REWARD_FUNCTION`: Objective function definition
- `RL_ALGO`: 'A2C', 'DQN', 'PPO' (A2C fully implemented)

### Running Benchmarks

The framework supports comparison of:
- RL agents vs. MILP approaches
- Different planning horizons
- Various demand scenarios
- Custom reward functions

## Results & Performance

The RL-based approach demonstrates:
- **Competitive profitability** with optimization methods
- **Faster computation** enabling real-time scheduling
- **Robust performance** under demand uncertainty
- **Scalability** to complex production environments

Detailed performance metrics include:
- On-time delivery rate
- Inventory holding costs
- Late delivery penalties
- Total profitability

## Research Context

This implementation is part of my Bachelor's thesis exploring:
- **AI in Operations Research**: Bridging data-driven and model-based optimization
- **Online Learning**: Adapting to dynamic production environments
- **Hybrid Approaches**: Potential for integrating DRL and mathematical programming

### Original Publication

> Hubbs, C. D., Perez, H. D., Sarwar, O., Sahinidis, N. V., Grossmann, I. E., & Wassick, J. M. (2020). *Or forum—deep reinforcement learning for industrial applications: A review and a case study in the process industry.* Computers & Chemical Engineering, 140, 106964.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



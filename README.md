# FrozenLake Q-Learning Project

## Project Description
This project implements a Q-Learning algorithm to solve the FrozenLake environment from Gymnasium (formerly OpenAI Gym). FrozenLake is a 4x4 and 8x8 grid environment where an agent must navigate from the starting position to the goal while avoiding holes.

## Features
- Q-Learning implementation with epsilon-greedy exploration
- Reward shaping to enhance learning (-1 for falling into holes)
- Visualization of learning results
- Demonstration mode to showcase learned policy
- Q-table model saving for future use

## Requirements
```
numpy
gymnasium
matplotlib
pygame
```

## Installation
```bash
pip install numpy gymnasium matplotlib
```

## How to Use
1. Run the script to train the agent:
```bash
python model.py
```

2. After training is complete, the Q-table model will be saved as `q_table_frozenlake.npy` and a reward graph will be displayed.

3. To load a pre-trained model:
```python
import numpy as np
q_table = np.load("q_table_frozenlake.npy")
```

## Algorithm Parameters
- **alpha (learning rate)**: 0.8
- **gamma (discount factor)**: 0.95
- **epsilon (exploration)**: 1.0 (initial)
- **epsilon_min**: 0.01
- **epsilon_decay**: 0.995
- **episodes**: 10000

## Code Explanation
The script is divided into several main sections:

1. **Environment and Parameter Initialization**:
   - Creates a FrozenLake environment without slipping (deterministic)
   - Initializes the Q-table with zero values
   - Sets learning parameters

2. **Training Loop**:
   - Uses epsilon-greedy exploration for action selection
   - Updates the Q-table using the Bellman equation
   - Reduces epsilon over time (epsilon decay)
   - Records rewards per episode

3. **Testing and Visualization**:
   - Displays the environment in render mode
   - Executes the optimal policy learned
   - Visualizes the reward graph during the learning process

## Reward Modification
To enhance learning, rewards are modified:
```python
if terminated and reward == 0:
    modified_reward = -1.0  # Penalty for falling into holes
else:
    modified_reward = reward  # Original reward for reaching the goal
```

## Learning Outcomes
After training, the agent should be able to:
- Consistently avoid holes
- Find the shortest path to the goal
- Optimize total reward

## Future Development
Some ideas for further development:
- Implementation with slippery environment (non-deterministic)
- Comparison with other RL algorithms (SARSA, Deep Q-Learning)
- Experimentation with different parameters (alpha, gamma, epsilon decay)
- Implementation in more complex environments

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



<div align="center">
  <h3> Contact </h3>
  <a href="https://www.instagram.com/finsa080200/" target="_blank">
    <img alt="Instagram" src="https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white" />
  </a>
  <a href="https://www.facebook.com/profile.php?id=61575076614708" target="_blank">
    <img alt="Facebook" src="https://img.shields.io/badge/Facebook-1877F2?style=for-the-badge&logo=facebook&logoColor=white" />
  </a>
  <a href="https://x.com/Finsa_Kusuma" target="_blank">
    <img alt="X" src="https://img.shields.io/badge/X-000000?style=for-the-badge&logo=x&logoColor=white" />
  </a>
  <a href="https://t.me/@finsakusuma" target="_blank">
    <img alt="Telegram" src="https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white" />
  </a>
  <a href="https://www.linkedin.com/" target="_blank">
    <img alt="LinkedIn" src="https://img.shields.io/static/v1?message=LinkedIn&logo=linkedin&label=&color=0077B5&logoColor=white&labelColor=&style=for-the-badge" />
  </a>
  <a href="mailto:finsakusumaputra@gmail.com">
    <img src="https://img.shields.io/badge/Email-finsakusumaputra@gmail.com-D14836?style=flat&logo=gmail&logoColor=white" alt="Email">
  </a>
</div>

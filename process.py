import numpy as np
import gymnasium as gym
import time

env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=False)
n_states = env.observation_space.n
n_actions = env.action_space.n
q_table = np.zeros((n_states, n_actions))

alpha = 0.8
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
episodes = 5000

for episode in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        modified_reward = reward
        if terminated and reward == 0:
            modified_reward = -1.0

        q_table[state, action] = q_table[state, action] + alpha * (
                modified_reward + gamma * np.max(q_table[new_state]) - q_table[state, action]
        )

        state = new_state
        total_reward += modified_reward

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    if episode % 1 == 0:
        print(f"Episode {episode}: epsilon = {epsilon:.4f}, reward = {total_reward}")

print("Training finished!")

print("Tes run started!")
state, _ = env.reset()
done = False
total_reward = 0

while not done:
    action = np.argmax(q_table[state])
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    total_reward += reward
    time.sleep(0.3)

print("done! Total Reward:", total_reward)
env.close()

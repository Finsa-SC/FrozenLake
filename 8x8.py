import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import time
import pygame


pygame.init()


map_size = 8


if map_size == 4:
    custom_map = [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ]
elif map_size == 8:
    custom_map = [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    ]
else:
    custom_map = generate_random_map(size=map_size, p=0.8)


env = gym.make(
    "FrozenLake-v1",
    desc=custom_map,
    is_slippery=False,
    render_mode="human"
)

env_raw = env.unwrapped
if hasattr(env_raw, 'cell_size'):
    env_raw = env.unwrapped
    if hasattr(env_raw, 'cell_size'):
        if hasattr(env_raw, 'window') and env_raw.window is not None:
            pygame.display.quit()
            env_raw.window = None
            env_raw.clock = None
            env.reset()

n_states = env.observation_space.n
n_actions = env.action_space.n
q_table = np.zeros((n_states, n_actions))


alpha = 0.8
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
episodes = 3000


for episode in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0
    max_steps = map_size * map_size * 3

    while not done and steps < max_steps:

        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])


        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        steps += 1


        modified_reward = reward
        if terminated and reward == 0:
            modified_reward = -1.0


        if not done:
            modified_reward -= 0.01


        q_table[state, action] = q_table[state, action] + alpha * (
                modified_reward + gamma * np.max(q_table[new_state]) - q_table[state, action]
        )


        state = new_state
        total_reward += reward


    if epsilon > epsilon_min:
        epsilon *= epsilon_decay


    if episode % 100 == 0:
        print(f"Episode {episode}: epsilon = {epsilon:.4f}, reward = {total_reward:.2f}, steps = {steps}")

print("Training finished!")


print("Tes run started!")
state, _ = env.reset()
done = False
total_reward = 0
steps = 0


delay = 0.3

while not done:
    action = np.argmax(q_table[state])
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    total_reward += reward
    steps += 1
    time.sleep(delay)

print(f"Done! Total Reward: {total_reward}, Total Steps: {steps}")
env.close()
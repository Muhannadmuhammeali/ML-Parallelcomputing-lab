import gym
from gym import spaces
import numpy as np
import random
import matplotlib.pyplot as plt

# Custom Maze Environment
class MazeEnv(gym.Env):
    def __init__(self):
        super(MazeEnv, self).__init__()
        
        # Maze layout
        # 0 = empty
        # 1 = wall
        # 2 = goal
        self.maze = np.array([
            [0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 2, 0]
        ])
        
        self.start_pos = (0, 0)
        self.goal_pos = (4, 3)
        self.agent_pos = self.start_pos
        
        self.n_rows, self.n_cols = self.maze.shape
        
        # Action space: 0=up,1=down,2=left,3=right
        self.action_space = spaces.Discrete(4)
        
        # Observation space: position (row, col)
        self.observation_space = spaces.Discrete(self.n_rows * self.n_cols)

    def reset(self):
        self.agent_pos = self.start_pos
        return self._get_state()

    def _get_state(self):
        return self.agent_pos[0] * self.n_cols + self.agent_pos[1]

    def step(self, action):
        row, col = self.agent_pos
        
        if action == 0:   # up
            row -= 1
        elif action == 1: # down
            row += 1
        elif action == 2: # left
            col -= 1
        elif action == 3: # right
            col += 1
        
        # Check boundaries and walls
        if (0 <= row < self.n_rows and 
            0 <= col < self.n_cols and 
            self.maze[row, col] != 1):
            self.agent_pos = (row, col)
        
        reward = -1
        done = False
        
        if self.agent_pos == self.goal_pos:
            reward = 100
            done = True
            
        return self._get_state(), reward, done, {}

    def render(self):
        maze_copy = self.maze.copy()
        r, c = self.agent_pos
        maze_copy[r][c] = 9
        print(maze_copy)

# Initialize environment
env = MazeEnv()

# Q-table
state_size = env.observation_space.n
action_size = env.action_space.n
Q = np.zeros((state_size, action_size))

# Hyperparameters
alpha = 0.1        # learning rate
gamma = 0.9        # discount factor
epsilon = 1.0      # exploration rate
epsilon_decay = 0.995
epsilon_min = 0.01

episodes = 500
rewards = []

for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        # Epsilon-greedy
        if random.uniform(0,1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])
        
        next_state, reward, done, _ = env.step(action)
        
        # Q-learning update
        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )
        
        state = next_state
        total_reward += reward
    
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    rewards.append(total_reward)

print("Training Complete!")
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Learning Curve")
plt.show()

state = env.reset()
done = False

print("Agent Path:")

while not done:
    env.render()
    action = np.argmax(Q[state])
    state, reward, done, _ = env.step(action)

env.render()
print("Goal Reached!")

success = 0
test_episodes = 100

for _ in range(test_episodes):
    state = env.reset()
    done = False
    
    while not done:
        action = np.argmax(Q[state])
        state, reward, done, _ = env.step(action)
    
    if reward == 100:
        success += 1

print("Success Rate:", success / test_episodes)

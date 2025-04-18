import gymnasium as gym
import numpy as np
import torch

from model import RewardModel, WorldModel

class PendulumAgent:
    def __init__(self, env: gym.Env, num_samples: int = 200, horizon: int = 15):
        self.env = env
        self.num_samples = num_samples
        self.horizon = horizon
        self.action_dim = env.action_space.shape[0]
        self.action_low = env.action_space.low
        self.action_high = env.action_space.high
        self.reward_model = RewardModel()
        self.world_model = WorldModel()
        self.reward_model.load_state_dict(torch.load("reward_model.pth"))
        self.world_model.load_state_dict(torch.load("world_model.pth"))
    
    def get_action(self, state):
        # random shooting actions sequences
        action_sequences = np.random.uniform(
            low = self.action_low, 
            high = self.action_high, 
            size= (self.num_samples, self.horizon, self.action_dim)
        )
        # getting the cost of by simulating
        costs = []
        for seq in action_sequences: 
            sim_state = np.copy(state)
            total_cost = 0
            for a in seq:
                # total_cost += self.true_cost_fn(sim_state, a)
                total_cost += self.sim_cost_fn(sim_state, a).item()
                sim_state = self.sim_dyn(sim_state, a)
                # sim_state = self.true_dyn(sim_state, a)
            costs.append(total_cost)
        # getting best action by min cost
        best_action = action_sequences[np.argmin(costs)][0]
        return best_action

    def update(self, state, reward):
        pass
    
    # The correct cost fn based on the formula
    def true_cost_fn(self, state, action, model=None):
        theta = np.arctan2(state[1], state[0])
        theta_dot = state[2]
        torque = action[0]
        reward =  theta**2 + 0.1 * theta_dot**2 + 0.001 * (torque**2)
        return reward
    
    # The simulated cost fn using NN to appx reward fn
    def sim_cost_fn(self, state, action):
        theta = np.arctan2(state[1], state[0])
        theta_dt = state[2]
        torque = action[0]
        x = torch.tensor([theta, theta_dt, torque], dtype=torch.float32)
        return self.reward_model(x)


    # The correct simulation of state + action = next_state
    def true_dyn(self, state, action, dt=0.05):
        g = 10.0     # gravity
        m = 1.0      # mass
        l = 1.0      # length of pendulum
        max_speed = 8.0
        max_torque = 2.0
        th = np.arctan2(state[1], state[0])  # angle θ
        thdot = state[2]                     # angular velocity
        u = np.clip(action, -max_torque, max_torque)[0]  # limit torque
        # Apply the physics: θ̈ = dynamics equation
        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3.0 / (m * l ** 2) * u) * dt
        newthdot = np.clip(newthdot, -max_speed, max_speed)
        newth = th + newthdot * dt
        return np.array([np.cos(newth), np.sin(newth), newthdot])

    def sim_dyn(self, state, action):
        state, action = torch.from_numpy(state).float(), torch.from_numpy(action).float()
        if(next(self.world_model.parameters()).device == 'cpu'):
            return self.world_model(state, action).detach().numpy()
        else: 
            return self.world_model(state, action).cpu().detach().numpy()


env = gym.make("Pendulum-v1", render_mode="human")
observation, info = env.reset()

agent = PendulumAgent(env)

episode_over = False
while not episode_over:
    # action = env.action_space.sample()  # agent policy that uses the observation and info
    action = agent.get_action(observation)

    observation, reward, terminated, truncated, info = env.step(action)
    episode_over = terminated or truncated

env.close()
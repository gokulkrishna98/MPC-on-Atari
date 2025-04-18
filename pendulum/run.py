import gymnasium as gym
import numpy as np


class PendulumAgent:
    def __init__(self, env: gym.Env, num_samples: int = 200, horizon: int = 15):
        self.env = env
        self.num_samples = num_samples
        self.horizon = horizon
        self.action_dim = env.action_space.shape[0]
        self.action_low = env.action_space.low
        self.action_high = env.action_space.high
    
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
                total_cost += self.cost_fn(sim_state, a)
                sim_state = self.true_dyn(sim_state, a)
            costs.append(total_cost)
        # getting best action by min cost
        best_action = action_sequences[np.argmin(costs)][0]
        return best_action

    def update(self, state, reward):
        pass
    
    # these function are reward and world model
    def cost_fn(self, state, action, model=None):
        theta = np.arctan2(state[1], state[0])
        theta_dot = state[2]
        torque = action[0]
        reward =  theta**2 + 0.1 * theta_dot**2 + 0.001 * (torque**2)
        return reward
    
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
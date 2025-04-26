import os
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import deque
import random
from tqdm import tqdm

# Define constants
GAME_NAME = "Breakout-v0"  # Choose your Atari game
FRAME_STACK = 4  # Number of frames to stack
IMAGE_SIZE = 84  # Resized frame dimension
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
BUFFER_SIZE = 100000
N_EPOCHS = 1 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM = 256  # Latent dimension for the world model
HORIZON = 40  # Planning horizon for MPC
NUM_TRAJECTORIES = 10  # Number of random trajectories to sample
DISCOUNT = 0.99  # Discount factor for reward calculation

# Preprocessing utilities
def preprocess_frame(frame):
    """Preprocess frame: resize, convert to grayscale, normalize"""
    import cv2
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    normalized = resized / 255.0
    return normalized

class ReplayBuffer:
    """Experience replay buffer to store transitions"""
    def __init__(self, capacity=BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        
        return (
            torch.FloatTensor(np.array(states)).to(DEVICE),
            torch.LongTensor(np.array(actions)).to(DEVICE),
            torch.FloatTensor(np.array(rewards)).to(DEVICE),
            torch.FloatTensor(np.array(next_states)).to(DEVICE),
            torch.FloatTensor(np.array(dones)).to(DEVICE)
        )
    
    def __len__(self):
        return len(self.buffer)

class AtariDataset(Dataset):
    """Custom dataset for Atari frames"""
    def __init__(self, replay_buffer):
        self.buffer = replay_buffer
    
    def __len__(self):
        return len(self.buffer)
    
    def __getitem__(self, idx):
        state, action, reward, next_state, done = self.buffer.buffer[idx]
        return (
            torch.FloatTensor(state),
            torch.LongTensor([action]),
            torch.FloatTensor([reward]),
            torch.FloatTensor(next_state),
            torch.FloatTensor([done])
        )

# CNN Encoder for state representation
class Encoder(nn.Module):
    def __init__(self, input_channels=FRAME_STACK):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate output size
        conv_output_size = self._get_conv_output_size((input_channels, IMAGE_SIZE, IMAGE_SIZE))
        
        self.fc = nn.Linear(conv_output_size, LATENT_DIM)
        
    def _get_conv_output_size(self, shape):
        bs = 1
        x = torch.rand(bs, *shape)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return int(np.prod(x.size()))
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# World Model: Predicts next state and reward
class WorldModel(nn.Module):
    def __init__(self, input_channels=FRAME_STACK, num_actions=4):
        super(WorldModel, self).__init__()
        
        self.encoder = Encoder(input_channels)
        
        # Transition model (state + action -> next state)
        self.transition_model = nn.Sequential(
            nn.Linear(LATENT_DIM + num_actions, 512),
            nn.ReLU(),
            nn.Linear(512, LATENT_DIM)
        )
        
        # Reward prediction model
        self.reward_model = nn.Sequential(
            nn.Linear(LATENT_DIM + num_actions, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self.num_actions = num_actions
        
    def forward(self, state, action):
        # Encode state
        state_encoding = self.encoder(state)
        
        # One-hot encode action
        action_one_hot = F.one_hot(action.squeeze(-1), self.num_actions).float()
        
        # Combine state and action
        x = torch.cat([state_encoding, action_one_hot], dim=-1)
        
        # Predict next state and reward
        next_state_pred = self.transition_model(x)
        reward_pred = self.reward_model(x)
        
        return next_state_pred, reward_pred
    
    def predict_next_state(self, state_encoding, action):
        # One-hot encode action
        action_one_hot = F.one_hot(action, self.num_actions).float()
        
        # Combine state and action
        x = torch.cat([state_encoding, action_one_hot], dim=-1)
        
        # Predict next state
        next_state_pred = self.transition_model(x)
        
        return next_state_pred

# Dedicated Reward Model for better reward prediction
class RewardModel(nn.Module):
    def __init__(self, num_actions=4):
        super(RewardModel, self).__init__()
        
        self.encoder = Encoder()
        
        # Reward prediction layers
        self.reward_predictor = nn.Sequential(
            nn.Linear(LATENT_DIM + num_actions, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.num_actions = num_actions
        
    def forward(self, state, action):
        # Encode state
        state_encoding = self.encoder(state)
        
        # One-hot encode action
        action_one_hot = F.one_hot(action.squeeze(-1), self.num_actions).float()
        
        # Combine state and action
        x = torch.cat([state_encoding, action_one_hot], dim=-1)
        
        # Predict reward
        reward_pred = self.reward_predictor(x)
        
        return reward_pred
    
    def predict_reward(self, state_encoding, action):
        # One-hot encode action
        action_one_hot = F.one_hot(action, self.num_actions).float()
        
        # Combine state and action
        x = torch.cat([state_encoding, action_one_hot], dim=-1)
        
        # Predict reward
        reward_pred = self.reward_predictor(x)
        
        return reward_pred

def collect_data(env, buffer, num_episodes=100):
    """Collect data from the environment using random actions"""
    for episode in tqdm(range(num_episodes), desc="Collecting data"):
        state = env.reset()
        # Process the initial state
        processed_state = np.array([preprocess_frame(state) for _ in range(FRAME_STACK)])
        done = False
        
        while not done:
            action = env.action_space.sample()  # Random action
            next_state, reward, done, _ = env.step(action)
            
            # Process the next state
            next_processed = np.copy(processed_state)
            next_processed[:-1] = next_processed[1:]
            next_processed[-1] = preprocess_frame(next_state)
            
            # Add to buffer
            buffer.add(processed_state, action, reward, next_processed, done)
            
            processed_state = next_processed
    
    return buffer

def train_models():
    """Train world model and reward model"""
    # Create environment
    env = gym.make(GAME_NAME)
    num_actions = env.action_space.n
    
    # Create replay buffer and collect data
    buffer = ReplayBuffer()
    buffer = collect_data(env, buffer)
    
    # Create dataset and dataloader
    dataset = AtariDataset(buffer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize models
    world_model = WorldModel(num_actions=num_actions).to(DEVICE)
    reward_model = RewardModel(num_actions=num_actions).to(DEVICE)
    
    # Initialize optimizers
    world_optimizer = optim.Adam(world_model.parameters(), lr=LEARNING_RATE)
    reward_optimizer = optim.Adam(reward_model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    for epoch in range(N_EPOCHS):
        world_losses = []
        reward_losses = []
        
        for states, actions, rewards, next_states, dones in tqdm(dataloader, desc=f"Epoch {epoch+1}/{N_EPOCHS}"):
            states = states.to(DEVICE)
            actions = actions.to(DEVICE)
            rewards = rewards.to(DEVICE)
            next_states = next_states.to(DEVICE)
            
            # Encode the next states for computing loss
            next_state_encodings = world_model.encoder(next_states)
            
            # Train world model
            world_optimizer.zero_grad()
            next_state_preds, reward_preds = world_model(states, actions)
            
            # Compute losses
            next_state_loss = F.mse_loss(next_state_preds, next_state_encodings)
            world_reward_loss = F.mse_loss(reward_preds, rewards)
            world_loss = next_state_loss + world_reward_loss
            
            world_loss.backward()
            world_optimizer.step()
            
            # Train reward model
            reward_optimizer.zero_grad()
            reward_preds = reward_model(states, actions)
            reward_loss = F.mse_loss(reward_preds, rewards)
            
            reward_loss.backward()
            reward_optimizer.step()
            
            world_losses.append(world_loss.item())
            reward_losses.append(reward_loss.item())
        
        print(f"Epoch {epoch+1}/{N_EPOCHS}, "
              f"World Model Loss: {np.mean(world_losses):.4f}, "
              f"Reward Model Loss: {np.mean(reward_losses):.4f}")
    
    return world_model, reward_model, env, num_actions

def mpc_planning(world_model, reward_model, state, num_actions):
    """Model Predictive Control with random shooting"""
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
    state_encoding = world_model.encoder(state_tensor)
    
    best_reward = float('-inf')
    best_action = 0
    
    # Generate random trajectories
    for _ in range(NUM_TRAJECTORIES):
        total_reward = 0
        current_state_encoding = state_encoding.clone()
        
        # Rollout for HORIZON steps
        for t in range(HORIZON):
            # Sample random action
            action = torch.randint(0, num_actions, (1,)).to(DEVICE)
            
            # Predict next state and reward
            next_state_encoding = world_model.predict_next_state(current_state_encoding, action)
            reward = reward_model.predict_reward(current_state_encoding, action)
            
            # Update total reward (with discount)
            total_reward += (DISCOUNT ** t) * reward.item()
            
            # Update current state
            current_state_encoding = next_state_encoding
            
            # For the first step, track the best initial action
            if t == 0 and total_reward > best_reward:
                best_reward = total_reward
                best_action = action.item()
    
    return best_action

def evaluate_mpc(world_model, reward_model, env, num_actions, num_episodes=10):
    """Evaluate MPC planning with the trained models"""
    total_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        processed_state = np.array([preprocess_frame(state) for _ in range(FRAME_STACK)])
        done = False
        episode_reward = 0
        
        while not done:
            # Select action using MPC
            action = mpc_planning(world_model, reward_model, processed_state, num_actions)
            
            # Execute action
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            
            # Process the next state
            next_processed = np.copy(processed_state)
            next_processed[:-1] = next_processed[1:]
            next_processed[-1] = preprocess_frame(next_state)
            
            processed_state = next_processed
            
            # Optional: render the environment
            # env.render()
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode+1}/{num_episodes}, Total Reward: {episode_reward}")
    
    print(f"Average Reward over {num_episodes} episodes: {np.mean(total_rewards):.2f}")
    return total_rewards

if __name__ == "__main__":
    # # Train models
    # world_model, reward_model, env, num_actions = train_models()
    
    # # Save models
    # torch.save(world_model.state_dict(), "world_model.pth")
    # torch.save(reward_model.state_dict(), "reward_model.pth")
    world_model_path = "world_model.pth"
    reward_model_path = "reward_model.pth"


    env = gym.make(GAME_NAME, render_mode = 'human')
    env.reset()
    num_actions = env.action_space.n
    world_model = WorldModel(num_actions=num_actions).to(DEVICE)
    reward_model = RewardModel(num_actions=num_actions).to(DEVICE)
    
    # Evaluate with MPC
    evaluate_mpc(world_model, reward_model, env, num_actions)
import gym
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# ==== Hyperparameters ====
ENV_NAME = 'PongNoFrameskip-v4'
DATASET_SIZE = 5000  # How many transitions to collect
BATCH_SIZE = 32
NUM_TRAIN_STEPS = 500

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==== Preprocessing ====
def preprocess(obs):
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = cv2.resize(obs, (84, 84))
    return obs

# ==== Frame Stack ====
class FrameStack:
    def __init__(self, k):
        self.k = k
        self.frames = deque([], maxlen=k)

    def reset(self, obs):
        processed = preprocess(obs)
        for _ in range(self.k):
            self.frames.append(processed)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        processed = preprocess(obs)
        self.frames.append(processed)
        return np.stack(self.frames, axis=0)

# ==== World Model ====
class WorldModel(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU()
        )
        self.fc = nn.Linear(512 + action_size, 3136)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 4, 8, stride=4),
            nn.Sigmoid()
        )

    def forward(self, obs, action):
        x = self.encoder(obs)
        x = torch.cat([x, action], dim=-1)
        x = self.fc(x)
        x = x.view(-1, 64, 7, 7)
        x = self.decoder(x)
        return x

# ==== Reward Model ====
class RewardModel(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU()
        )
        self.fc = nn.Linear(512 + action_size, 1)

    def forward(self, obs, action):
        x = self.encoder(obs)
        x = torch.cat([x, action], dim=-1)
        reward = self.fc(x)
        return reward.squeeze(-1)

# ==== Collect Dataset ====
env = gym.make(ENV_NAME)
frame_stack = FrameStack(k=4)

dataset = []

obs = env.reset()
obs = frame_stack.reset(obs)

for _ in range(DATASET_SIZE):
    action = env.action_space.sample()
    next_obs, reward, done, info = env.step(action)
    next_obs_processed = frame_stack.step(next_obs)
    
    dataset.append((
        obs,  # stacked frames
        action,
        reward,
        next_obs_processed
    ))
    
    obs = next_obs_processed
    if done:
        obs = env.reset()
        obs = frame_stack.reset(obs)

print(f"Collected {len(dataset)} transitions!")

# ==== Prepare Torch Dataset ====
action_size = env.action_space.n

obses = torch.tensor(np.stack([d[0] for d in dataset]), dtype=torch.float32) / 255.0
actions = torch.tensor([d[1] for d in dataset], dtype=torch.long)
rewards = torch.tensor([d[2] for d in dataset], dtype=torch.float32)
next_obses = torch.tensor(np.stack([d[3] for d in dataset]), dtype=torch.float32) / 255.0

# One-hot encode actions
actions_onehot = torch.nn.functional.one_hot(actions, num_classes=action_size).float()

# ==== Create Models ====
world_model = WorldModel(action_size).to(DEVICE)
reward_model = RewardModel(action_size).to(DEVICE)

world_optimizer = optim.Adam(world_model.parameters(), lr=1e-3)
reward_optimizer = optim.Adam(reward_model.parameters(), lr=1e-3)

# ==== Training Loop ====
for step in range(NUM_TRAIN_STEPS):
    idx = np.random.choice(len(dataset), BATCH_SIZE, replace=False)
    
    obs_batch = obses[idx].to(DEVICE)
    action_batch = actions_onehot[idx].to(DEVICE)
    reward_batch = rewards[idx].to(DEVICE)
    next_obs_batch = next_obses[idx].to(DEVICE)

    # --- World Model Training ---
    pred_next_obs = world_model(obs_batch, action_batch)
    world_loss = ((pred_next_obs - next_obs_batch.unsqueeze(1)) ** 2).mean()

    world_optimizer.zero_grad()
    world_loss.backward()
    world_optimizer.step()

    # --- Reward Model Training ---
    pred_reward = reward_model(obs_batch, action_batch)
    reward_loss = ((pred_reward - reward_batch) ** 2).mean()

    reward_optimizer.zero_grad()
    reward_loss.backward()
    reward_optimizer.step()

    if step % 50 == 0:
        print(f"Step {step}: World Loss = {world_loss.item():.4f}, Reward Loss = {reward_loss.item():.4f}")

print("Training completed!")
torch.save(world_model.state_dict(), "world_model.pth")
torch.save(reward_model.state_dict(), "reward_model.pth")
print("Models saved!")

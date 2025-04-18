import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from model import RewardModel
from tqdm import tqdm


def true_reward(theta, theta_dt, torque):
    return theta**2 + 0.1 * theta_dt**2 + 0.001 * torque**2


class PendulumRewardDataset(Dataset):
    def __init__(self, size=10000):
        self.size = size
        self.data = []

        for _ in range(size):
            theta = np.random.uniform(-np.pi, np.pi)
            theta_dt = np.random.uniform(-8.0, 8.0)
            torque = np.random.uniform(-2.0, 2.0)
            reward = true_reward(theta, theta_dt, torque)
            self.data.append((theta, theta_dt, torque, reward))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        theta, theta_dt, torque, reward = self.data[idx]
        x = torch.tensor([theta, theta_dt, torque], dtype=torch.float32)
        y = torch.tensor([reward], dtype=torch.float32)
        return x, y


dataset = PendulumRewardDataset(size=100000)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


def train_reward_model(model, dataloader, epochs=10, lr=1e-3, device='cpu'):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(epochs), desc="Training epochs"):
        model.train()
        running_loss = 0.0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

model = RewardModel()
train_reward_model(model, dataloader, epochs=20, lr=1e-3)


torch.save(model.state_dict(), "reward_model.pth")
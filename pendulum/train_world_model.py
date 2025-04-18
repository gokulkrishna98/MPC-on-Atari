import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import WorldModel 
from tqdm import tqdm


def true_dyn(state, action, dt=0.05):
    g = 10.0
    m = 1.0
    l = 1.0
    max_speed = 8.0
    max_torque = 2.0
    th = np.arctan2(state[1], state[0])
    thdot = state[2]
    u = np.clip(action, -max_torque, max_torque)[0]
    newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3.0 / (m * l ** 2) * u) * dt
    newthdot = np.clip(newthdot, -max_speed, max_speed)
    newth = th + newthdot * dt
    return np.array([np.cos(newth), np.sin(newth), newthdot])


class PendulumDataset(Dataset):
    def __init__(self, num_samples=10000):
        self.states = []
        self.actions = []
        self.next_states = []

        for _ in range(num_samples):
            th = np.random.uniform(-np.pi, np.pi)
            thdot = np.random.uniform(-8, 8)
            action = np.random.uniform(-2.0, 2.0, size=(1,))
            state = np.array([np.cos(th), np.sin(th), thdot])
            next_state = true_dyn(state, action)

            self.states.append(state)
            self.actions.append(action)
            self.next_states.append(next_state)

        self.states = torch.tensor(self.states, dtype=torch.float32)
        self.actions = torch.tensor(self.actions, dtype=torch.float32)
        self.next_states = torch.tensor(self.next_states, dtype=torch.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.next_states[idx]


# Create dataset and dataloader
dataset = PendulumDataset(num_samples=100000)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# Model, loss, optimizer
model = WorldModel()

def train_world_model(model, dataloader, num_epochs=100, lr=1e-3, device='cpu'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in tqdm(range(num_epochs), desc="Training epochs"):
        model.train()
        epoch_loss = 0.0

        for state, action, next_state in dataloader:
            state, action, next_state = state.to(device), action.to(device), next_state.to(device)

            pred_next = model(state, action)
            loss = loss_fn(pred_next, next_state)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * state.size(0)

        avg_loss = epoch_loss / len(dataloader.dataset)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")

train_world_model(model, dataloader, num_epochs=20, lr=1e-3)
torch.save(model.state_dict(), "world_model.pth")
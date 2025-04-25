import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# Config
batch_size = 64
epochs = 10
device = torch.device("cpu")

# Data
transform = transforms.ToTensor()
train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

# Model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# AGILE Optimizer
class AGILE:
    def __init__(self, params, eta_min=0.00985, eta_max=0.0951, mu=1.09, alpha=0.8, P0=477):
        self.params = list(params)
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.mu = mu
        self.alpha = alpha
        self.Pk = P0
        self.g_best = [p.data.clone() for p in self.params]
        self.L_best = None
        self.steps_since_improve = 0

    def sample_eta(self):
        u = np.random.uniform()
        return ((self.eta_max**(1 - self.mu) - self.eta_min**(1 - self.mu)) * u + self.eta_min**(1 - self.mu))**(1 / (1 - self.mu))

    def step(self, loss_fn, grads):
        flat_grad = torch.cat([g.view(-1) for g in grads])
        norm_g = torch.norm(flat_grad) + 1e-8
        d = [-g / norm_g for g in grads] if norm_g > 1e-6 else [torch.randn_like(g) for g in grads]
        eta = self.sample_eta()
        for p, gdir in zip(self.params, d):
            p.data.add_(eta * gdir)
        new_loss = loss_fn().item()
        if self.L_best is None or new_loss < self.L_best:
            self.L_best = new_loss
            self.g_best = [p.data.clone() for p in self.params]
            self.Pk *= self.alpha
            self.steps_since_improve = 0
        else:
            self.steps_since_improve += 1
        if self.steps_since_improve > self.Pk:
            for p, g in zip(self.params, grads):
                p.data.sub_(self.eta_min * g)

# Manual SGD
class ManualSGD:
    def __init__(self, params, lr=0.01):
        self.params = list(params)
        self.lr = lr

    def step(self):
        for p in self.params:
            if p.grad is not None:
                p.data -= self.lr * p.grad

# Manual Adam
class ManualAdam:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.state = {id(p): {'m': torch.zeros_like(p), 'v': torch.zeros_like(p), 't': 0} for p in self.params}

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            state = self.state[id(p)]
            g = p.grad
            state['t'] += 1
            state['m'] = self.beta1 * state['m'] + (1 - self.beta1) * g
            state['v'] = self.beta2 * state['v'] + (1 - self.beta2) * g.pow(2)
            m_hat = state['m'] / (1 - self.beta1 ** state['t'])
            v_hat = state['v'] / (1 - self.beta2 ** state['t'])
            p.data -= self.lr * m_hat / (v_hat.sqrt() + self.eps)

# Train loop
def train(model, optimizer, use_agile=False):
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if use_agile:
                def compute_loss():
                    return loss_fn(model(x), y)
                loss = compute_loss()
                grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)
                optimizer.step(compute_loss, grads)
            else:
                loss = loss_fn(model(x), y)
                model.zero_grad()  # this clears grads for all params
                loss.backward()
                optimizer.step()

# Timing wrapper
def time_training(label, opt_func, use_agile=False):
    model = MLP().to(device)
    optimizer = opt_func(model)
    start = time.time()
    train(model, optimizer, use_agile)
    duration = time.time() - start
    print(f"{label} training time: {duration:.4f} seconds")

def main():
    print("MNIST Benchmark")
    time_training("AGILE", lambda model: AGILE(model.parameters()), use_agile=True)
    time_training("Manual Adam", lambda model: ManualAdam(model.parameters()))
    time_training("Manual SGD",  lambda model: ManualSGD(model.parameters()))

if __name__ == "__main__":
    main()

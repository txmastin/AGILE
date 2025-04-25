import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import optuna
import os

# Minimalist config
batch_size = 64
epochs = 2  # use 2–3 epochs for speed during search
device = torch.device("cpu")

# MNIST data
transform = transforms.ToTensor()
train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)

# Simple 2-layer MLP
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def evaluate(model):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

# AGILE optimizer with dynamic parameters
class AGILE:
    def __init__(self, params, eta_min, eta_max, mu, alpha, P0):
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

# Objective for Optuna
def objective(trial):
    eta_min = trial.suggest_float("eta_min", 1e-5, 1e-2, log=True)
    eta_max = trial.suggest_float("eta_max", 1e-2, 0.1, log=True)
    mu = trial.suggest_float("mu", 1.0, 2.5)
    alpha = trial.suggest_float("alpha", 0.7, 0.99)
    P0 = trial.suggest_int("P0", 10, 500)

    model = MLP().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AGILE(model.parameters(), eta_min, eta_max, mu, alpha, P0)
    model.train()

    for epoch in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            def compute_loss():
                return loss_fn(model(x), y)
            loss = compute_loss()
            grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)
            optimizer.step(compute_loss, grads)

    acc = evaluate(model)
    return acc

# Run Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=250)

# Report best result
print("Best trial:")
trial = study.best_trial
print(f"  Accuracy: {trial.value:.4f}")
for k, v in trial.params.items():
    print(f"  {k}: {v}")

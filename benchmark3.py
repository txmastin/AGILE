import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os

# Config
batch_size = 64
epochs = 20
device = torch.device("cpu")
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Data
transform = transforms.ToTensor()
train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)

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

# Evaluation
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

# Training with logging
def train(model, optimizer, name, use_agile=False):
    loss_fn = nn.CrossEntropyLoss()
    train_losses = []
    test_accuracies = []
    model.train()

    loss_path = os.path.join(log_dir, f"{name}_loss.dat")
    acc_path = os.path.join(log_dir, f"{name}_accuracy.dat")
    with open(loss_path, "w") as flog, open(acc_path, "w") as facc:
        flog.write("# batch_index loss\n")
        facc.write("# epoch accuracy\n")
        batch_idx = 0
        for epoch in range(epochs):
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                if use_agile:
                    def compute_loss():
                        return loss_fn(model(x), y)
                    loss = compute_loss()
                    grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)
                    optimizer.step(compute_loss, grads)
                else:
                    optimizer.zero_grad()
                    loss = loss_fn(model(x), y)
                    loss.backward()
                    optimizer.step()
                flog.write(f"{batch_idx} {loss.item():.6f}\n")
                batch_idx += 1
            acc = evaluate(model)
            facc.write(f"{epoch} {acc:.6f}\n")

# Run and log all
def run_all():
    # AGILE
    model = MLP().to(device)
    agile_opt = AGILE(model.parameters())
    train(model, agile_opt, name="agile", use_agile=True)

    # Adam
    model = MLP().to(device)
    adam_opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    train(model, adam_opt, name="adam")

    # SGD
    model = MLP().to(device)
    sgd_opt = torch.optim.SGD(model.parameters(), lr=0.01)
    train(model, sgd_opt, name="sgd")

if __name__ == "__main__":
    run_all()


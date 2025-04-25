import numpy as np
import matplotlib.pyplot as plt

def rastrigin(x):
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def rastrigin_grad(x):
    A = 10
    return 2 * x + 2 * np.pi * A * np.sin(2 * np.pi * x)

class AGILE:
    def __init__(self, eta_min, eta_max, mu, alpha, P0, epsilon=1e-8):
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.mu = mu
        self.alpha = alpha
        self.Pk = P0
        self.P0 = P0
        self.epsilon = epsilon
        self.reset()

    def reset(self):
        self.k = 0
        self.Pk = self.P0
        self.steps_since_improvement = 0
        self.phase = "exploration"
        self.theta_best = None
        self.L_best = None

    def sample_eta(self):
        u = np.random.uniform()
        return ((self.eta_max**(1 - self.mu) - self.eta_min**(1 - self.mu)) * u + self.eta_min**(1 - self.mu))**(1 / (1 - self.mu))

    def update(self, theta, grad, loss_fn):
        if self.phase == "exploration":
            eta = self.sample_eta()
            norm_g = np.linalg.norm(grad)
            if norm_g > self.epsilon:
                d = -grad / (norm_g + self.epsilon)
            else:
                d = np.random.randn(*grad.shape)
                d /= np.linalg.norm(d) + self.epsilon
            theta_new = theta + eta * d
            L_new = loss_fn(theta_new)

            if self.L_best is None or L_new < self.L_best:
                self.theta_best = np.copy(theta_new)
                self.L_best = L_new
                self.k += 1
                self.Pk *= self.alpha
                self.steps_since_improvement = 0
                return theta_new
            else:
                self.steps_since_improvement += 1
                if self.steps_since_improvement >= self.Pk:
                    self.phase = "refinement"
                    return np.copy(self.theta_best)
                return theta  # keep exploring
        else:
            return theta - self.eta_min * grad  # refinement step

def run_agile(dim, steps, trials, eta_min, eta_max, mu, alpha, P0):
    losses = []
    for _ in range(trials):
        theta = np.random.uniform(-5.12, 5.12, size=dim)
        optimizer = AGILE(eta_min, eta_max, mu, alpha, P0)
        trial_losses = []
        for _ in range(steps):
            loss = rastrigin(theta)
            grad = rastrigin_grad(theta)
            theta = optimizer.update(theta, grad, rastrigin)
            trial_losses.append(loss)
        losses.append(trial_losses)
    return np.array(losses)

def run_gd(dim, steps, trials, lr):
    losses = []
    for _ in range(trials):
        theta = np.random.uniform(-5.12, 5.12, size=dim)
        trial_losses = []
        for _ in range(steps):
            loss = rastrigin(theta)
            grad = rastrigin_grad(theta)
            theta -= lr * grad
            trial_losses.append(loss)
        losses.append(trial_losses)
    return np.array(losses)

def run_adam(dim, steps, trials, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
    losses = []
    for _ in range(trials):
        theta = np.random.uniform(-5.12, 5.12, size=dim)
        m = np.zeros_like(theta)
        v = np.zeros_like(theta)
        trial_losses = []
        for t in range(1, steps + 1):
            grad = rastrigin_grad(theta)
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            theta -= lr * m_hat / (np.sqrt(v_hat) + eps)
            loss = rastrigin(theta)
            trial_losses.append(loss)
        losses.append(trial_losses)
    return np.array(losses)

def main():
    dim = 10
    steps = 300
    trials = 10000

    agile_losses = run_agile(
        dim, steps, trials,
        eta_min=0.001,
        eta_max=0.5,
        mu=1.3,
        alpha=0.9,
        P0=30
    )
    adam_losses = run_adam(dim, steps, trials)
    sgd_losses = run_gd(dim, steps, trials, lr=0.01)

    x = np.arange(steps)
    plt.figure(figsize=(10, 6))
    for losses, label in zip([agile_losses, adam_losses, sgd_losses], ["AGILE", "Adam", "SGD"]):
        mean = losses.mean(axis=0)
        std = losses.std(axis=0)
        plt.plot(x, mean, label=label)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)
    plt.xlabel("Iteration")
    plt.ylabel("Rastrigin Loss")
    plt.title("Rastrigin 10D: AGILE vs Adam vs SGD")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("rastrigin.png")

if __name__ == "__main__":
    main()

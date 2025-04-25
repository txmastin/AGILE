import numpy as np
import matplotlib.pyplot as plt

def load_loss_file(path):
    data = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"): continue
            _, loss = line.strip().split()
            data.append(float(loss))
    return np.array(data)

def smooth_and_plot(loss_arrays, labels, window=10):
    plt.figure(figsize=(10, 5))

    for loss, label in zip(loss_arrays, labels):
        N = len(loss)
        num_windows = N // window
        means = []
        stds = []
        steps = []

        for i in range(num_windows):
            chunk = loss[i*window:(i+1)*window]
            means.append(np.mean(chunk))
            stds.append(np.std(chunk))
            steps.append(i * window)

        means = np.array(means)
        stds = np.array(stds)
        steps = np.array(steps)

        plt.plot(steps, means, label=label)
        plt.fill_between(steps, means - stds, means + stds, alpha=0.2)

    plt.xlabel("Batch")
    plt.ylabel("Loss (smoothed)")
    plt.title("Training Loss (Mean ± Std)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_smooth_plot.png")
    print("Saved: loss_smooth_plot.png")

# Load all three
agile_loss = load_loss_file("logs/agile_loss.dat")
adam_loss = load_loss_file("logs/adam_loss.dat")
sgd_loss = load_loss_file("logs/sgd_loss.dat")

# Plot
smooth_and_plot([agile_loss, adam_loss, sgd_loss], labels=["AGILE", "Adam", "SGD"], window=10)


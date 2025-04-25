import numpy as np
import matplotlib.pyplot as plt
import os

def load_loss_file(path):
    data = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"): continue
            _, loss = line.strip().split()
            data.append(float(loss))
    return np.array(data)

def load_acc_file(path):
    epochs = []
    accs = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"): continue
            epoch, acc = line.strip().split()
            epochs.append(int(epoch))
            accs.append(float(acc))
    return np.array(epochs), np.array(accs)

def smooth_and_plot_loss(loss_arrays, labels, window=10):
    plt.figure(figsize=(10, 5))
    for loss, label in zip(loss_arrays, labels):
        N = len(loss)
        num_windows = N // window
        means, stds, steps = [], [], []
        for i in range(num_windows):
            chunk = loss[i*window:(i+1)*window]
            means.append(np.mean(chunk))
            stds.append(np.std(chunk))
            steps.append(i * window)
        means, stds, steps = np.array(means), np.array(stds), np.array(steps)
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

def plot_accuracy(accuracy_arrays, labels):
    plt.figure(figsize=(10, 5))
    for (epochs, accs), label in zip(accuracy_arrays, labels):
        plt.plot(epochs, accs, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy over Epochs")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("accuracy_plot.png")
    print("Saved: accuracy_plot.png")

# Load files
agile_loss = load_loss_file("logs/agile_loss.dat")
adam_loss = load_loss_file("logs/adam_loss.dat")
sgd_loss = load_loss_file("logs/sgd_loss.dat")

agile_acc = load_acc_file("logs/agile_accuracy.dat")
adam_acc = load_acc_file("logs/adam_accuracy.dat")
sgd_acc = load_acc_file("logs/sgd_accuracy.dat")

# Plot both
smooth_and_plot_loss([agile_loss, adam_loss, sgd_loss], labels=["AGILE", "Adam", "SGD"], window=10)
plot_accuracy([agile_acc, adam_acc, sgd_acc], labels=["AGILE", "Adam", "SGD"])


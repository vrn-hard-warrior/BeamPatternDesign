# Plotting learning results and curves
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize


def equal_gain_combining(channel_file: str = "data/channels.npy") -> float:
    """
    Return array gain for certain channel using weights, computed by
    Equal Gain Combining technique.
    """
    H = np.load(channel_file)
    w = (H / np.abs(H)).conj()
    
    gain = np.abs(np.dot(w.T, H)) ** 2
    
    return gain


def plot_results(log_folder: str = "logs/A2C", \
                 filename: str = "progress.csv", \
                 yaxis: str = "train/entropy_loss", \
                 title: str = "Learning Curve",
                 saving: bool = True) -> None:
    """
    Plot learning results per timesteps.
    """
    # progress_data = pd.read_csv(log_folder + '/' + filename)["rollout/ep_rew_mean"].to_numpy()
    progress_data = pd.read_csv(log_folder + '/' + filename).sort_index()
    
    xy_list = progress_data[["time/total_timesteps", yaxis]].to_numpy()
    
    # xticks processing
    x_sticks = np.arange(0, xy_list[-1, 0] + 1, step = 10000)
    x_labels = [str(i) + r"$\cdot{10}^{4}$" for i in range(1, len(x_sticks))]
    x_labels.insert(0, r"$0$")
    
    figsize(10, 4)
    plt.plot(xy_list[:, 0], xy_list[:, 1], color = 'k', alpha = 0.8, lw = 2.0)
    plt.xlabel('Number of Timesteps', fontsize = 10)
    plt.ylabel(yaxis, fontsize = 10)
    plt.title(title, fontsize = 12)
    plt.xticks(ticks = x_sticks, labels = x_labels, fontsize = 10)
    plt.yticks(fontsize = 10)
    plt.grid()
    
    if saving:
        plt.savefig("graphs/entropy_loss.pdf", format = "pdf", bbox_inches = "tight")
    else:
        plt.show()


def plot_beamforming_gains(log_folder: str = "logs/A2C", \
                           channel_file: str = "data/channels.npy", \
                           saving: bool = True) -> None:
    """
    Plot beamforming gains for every timestep.
    """
    xy = np.loadtxt(log_folder + "/beamforming_gains.dat", dtype = np.float64)
    
    gains = xy[: -1, 0]
    timesteps = xy[: -1, 1]
    
    # Compute EGC estimation for comparing
    EGC_gain = equal_gain_combining(channel_file)
    EGC_gains = np.full_like(timesteps, EGC_gain, dtype = float)
    
    # xticks processing
    x_sticks = np.arange(0, timesteps[-1] + 1, step = 10000)
    x_labels = [str(i) + r"$\cdot{10}^{4}$" for i in range(1, len(x_sticks))]
    x_labels.insert(0, r"$0$")
    
    figsize(10, 4)
    plt.plot(timesteps, gains, color = 'k', alpha = 0.8, lw = 1.5)
    plt.plot(timesteps, EGC_gains, color = 'b', alpha = 0.8, lw = 1.5)
    plt.xlim((timesteps[0], timesteps[-1] - 3e3))
    plt.xlabel('Number of Timesteps', fontsize = 10)
    plt.ylabel('Beamforming gain', fontsize = 10)
    plt.title("Learning beamforming gains", fontsize = 12)
    plt.xticks(ticks = x_sticks, labels = x_labels, fontsize = 10)
    plt.yticks(fontsize = 10)
    plt.legend(["A2C", "EGC"])
    plt.grid()
    
    if saving:
        plt.savefig("graphs/beamforming_gains.pdf", format = "pdf", bbox_inches = "tight")
    else:
        plt.show()
    

if __name__ == "__main__":
    plot_beamforming_gains(log_folder = "logs/A2C", channel_file = "data/channels.npy", saving = True)
    plot_results(saving = True)
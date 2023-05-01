# Plotting learning results and curves
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize


def beam_pattern_plot(log_folder: str = "logs/A2C", n_timestep: int = 0, \
                      channel_file: str = "data/channels.npy", \
                      d_phi: float = 0.5, saving: bool = True) -> None:
    """
    Plotting beam pattern on certain timestep with learned weights.
    """
    weights = np.loadtxt(log_folder + "/weights.dat", dtype = np.complex128)
    weights = weights[n_timestep].conj()
    len_weigths = len(weights)
    
    # ideal weights computing
    H = np.load(channel_file)
    weights_ideal = np.squeeze((H / np.abs(H)).conj().T)
    
    phi = np.arange(-np.pi / 2, np.pi / 2, np.deg2rad(d_phi), dtype = np.float64)
    
    BP_RL = np.zeros_like(phi, dtype = np.complex128)
    BP_ideal = np.zeros_like(phi, dtype = np.complex128)
    for i in range(len(BP_RL)):
        exps = np.array([np.exp(1j * np.pi * n * np.sin(phi[i])) \
                         for n in range(len_weigths)], dtype = np.complex128)
        
        BP_RL[i] = np.matmul(weights, exps)
        BP_ideal[i] = np.matmul(weights_ideal, exps)
        
    figsize(10, 4)
    plt.plot(np.rad2deg(phi), 10 * np.log10(np.abs(BP_ideal)), color = 'k', alpha = 0.8, lw = 1.5)
    plt.plot(np.rad2deg(phi), 10 * np.log10(np.abs(BP_RL)), color = 'r', linestyle = "dashed", alpha = 0.8, lw = 1.5)
    plt.xlim((-90, 90))
    plt.xlabel('Азимут, град.', fontsize = 10)
    plt.ylabel('Усиление антенной решетки, дБ', fontsize = 10)
    plt.title("Диаграмма направленности антенны", fontsize = 12)
    plt.xticks(fontsize = 10)
    plt.yticks(fontsize = 10)
    plt.legend(["EGT", "RL"])
    plt.grid()
    
    if saving:
        plt.savefig("graphs/beam_patterns" + str(n_timestep) + "ts.pdf", format = "pdf", bbox_inches = "tight")
        plt.close('all')
    else:
        plt.show()


def equal_gain_combining(channel_file: str = "data/channels.npy") -> float:
    """
    Return array gain for certain channel using weights, computed by
    Equal Gain Combining technique.
    """
    H = np.load(channel_file)
    w = (H / np.abs(H)).conj()
    
    gain = np.abs(np.dot(w.T, H)) ** 2
    
    return gain


def beam_steering_codebook(channel_file: str = "data/channels.npy", \
                           N_p: np.int8 = 5) -> float:
    """
    Return array gain for certain channel using weights from beam-steering
    codebook design. Weight vector is got by pure Exhaustive Search method.
    """
    H = np.load(channel_file)
    
    # create codebook
    N_phases = np.power(2, N_p)
    N_elements = len(H)
    
    W = np.array([[np.exp(1j * np.pi * m * np.sin(2 * np.pi * k / N_phases)) \
                   for k in range(N_phases)] for m in range(N_elements)], dtype = np.complex128)
        
    # exhaustive search
    gain = np.max(np.power(np.absolute(np.matmul(W.conj().T, H)), 2))
    
    return gain


def plot_results(log_folder: str = "logs/A2C_v0", \
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
        plt.close('all')
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
    
    # Compute beam-steering codebook estimation for comparing
    BSC_gain = beam_steering_codebook(channel_file, N_p = 6)
    BSC_gains = np.full_like(timesteps, BSC_gain, dtype = float)
    
    # xticks processing
    x_sticks = np.arange(0, timesteps[-1] + 1, step = 10000)
    x_labels = [str(i) + r"$\cdot{10}^{4}$" for i in range(1, len(x_sticks))]
    x_labels.insert(0, r"$0$")
    
    figsize(10, 4)
    plt.plot(timesteps, gains, color = 'k', alpha = 0.8, lw = 1.5)
    plt.plot(timesteps, EGC_gains, color = 'b', linestyle = "dashed", alpha = 0.8, lw = 1.5)
    plt.plot(timesteps, BSC_gains, color = 'g', linestyle = "dashdot", alpha = 0.8, lw = 1.5)
    plt.xlim((timesteps[0], timesteps[-1] - 3e3))
    plt.xlabel('Номер прецедента', fontsize = 10)
    plt.ylabel('Усиление антенной решетки', fontsize = 10)
    plt.title("Процесс обучения агента", fontsize = 12)
    plt.xticks(ticks = x_sticks, labels = x_labels, fontsize = 10)
    plt.yticks(fontsize = 10)
    plt.legend(["A2C", "EGT", "Beam-steering codebook"])
    plt.grid()
    
    if saving:
        plt.savefig("graphs/beamforming_gains.pdf", format = "pdf", bbox_inches = "tight")
        plt.close('all')
    else:
        plt.show()


if __name__ == "__main__":
    plot_beamforming_gains(log_folder = "logs/A2C_v2", \
                           channel_file = "models/A2C_v2/channels.npy", saving = False)
    plot_results(log_folder = "logs/A2C_v2", saving = False)
    beam_pattern_plot(log_folder = "logs/A2C_v2", n_timestep = 40000, \
                      channel_file = "models/A2C_v2/channels.npy", \
                      d_phi = 0.5, saving = False)
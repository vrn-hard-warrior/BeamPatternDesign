# Creating a reinforcement learning model for Beam Pattern Learning task.
# Model is an environment, which is inherited from gym.Env-class.
# Creating a Callback function-class for saving beamforming gains with
# certain timestep's frequency.
import numpy as np
import gym
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback


class RL_model(gym.Env):
    """
    Creation of wireless communication system with one BS and one UE.
    
    In this version states and actions are the vectors of weights and
    reward is a RSSI-parameter of digital antenna array.
    """
    # main UPA parameters
    M: np.int16                        # number of UPA elements
    N_p: np.int8                       # number of possible phases for UPA elements
        
    # environment parameters
    H_ch: np.ndarray                   # channels samples for learning
    max_interactions: np.int32         # max length of one trajectory
    
    def __init__(self, M: np.int16 = 32,
                     N_p: np.int8 = 3,
                     H_ch: np.ndarray = np.ones((32, 100)),
                     max_interactions: np.int32 = 10000):
        super(RL_model, self).__init__()
        self.M = M
        self.N_p = N_p
        self.H_ch = H_ch
        
        # self.max_interactions = self.M * (2 ** self.N_p)
        self.max_interactions = max_interactions   # trying to imitate infinite horizon
        
        self.phase_shift = 2 * np.pi / (2 ** self.N_p - 1)
        self.phases = np.linspace(-np.pi, np.pi, 2 ** self.N_p)
        self.actions = [len(self.phases)] * self.M           # all possible actions for antenna elements adjustment
        
        # Actions = States for this task
        self.observation_space = gym.spaces.MultiDiscrete(self.actions, dtype = np.int8)
        self.action_space = gym.spaces.MultiDiscrete(self.actions, dtype = np.int8)
        
        
    def reset(self) -> np.array:
        self.interactions_n = 0
        self.done = False
        
        self.h = self.H_ch[:, np.random.choice(self.H_ch.shape[1])]
        
        w_mask = np.random.choice(len(self.phases), size = (self.M,))
        w_best = np.exp(1j * w_mask * self.phase_shift)
        
        self.g_old = np.abs(np.dot(w_best.conj().T, self.h)) ** 2
        
        return w_mask.astype(np.int8)


    def step(self, action: np.array) -> (np.array, np.int8, bool, dict):
        w = np.exp(1j * action * self.phase_shift)
        observation = action
        
        self.g = np.abs(np.dot(w.conj().T, self.h)) ** 2
        
        self.interactions_n += 1
        
        reward = self.g - self.g_old
        
        self.g_old = self.g
        self.w = w
        
        if self.interactions_n == self.max_interactions:
            self.done = True
        
        # Only for debugging and collection statistical data
        info = {"W": w}
        
        return observation, reward, self.done, info


# Environment checking
env = RL_model()
check_env(env)


class SaveBeamformingGainsCallback(BaseCallback):
    """
    Save best beamforming gains on certain intervals through learning.
    """
    def __init__(self, check_freq: np.uint16, n_steps: np.uint16, log_dir: str, verbose = 1):
        super(SaveBeamformingGainsCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.beamforming_gains = []
        self.mean_rewards = []
        self.weight_vec = []
        self.total_timesteps = []
        self.n_steps = n_steps
        self.i = 0
    
    
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            gain_i = self.training_env.get_attr("g", [0])
            weight_vec_i = self.training_env.get_attr("w", [0])
            
            self.i += self.check_freq
            self.beamforming_gains.append(gain_i)
            self.weight_vec.append(weight_vec_i[0])
            self.total_timesteps.append(self.i)
        
        if self.n_calls % self.n_steps == 0:
            mean_reward = getattr(self.model.rollout_buffer, "R_mean")
            
            self.mean_rewards.append(mean_reward)
        
        return True
    
    
    def _on_training_end(self) -> None:
        self.beamforming_gains = np.array(self.beamforming_gains, dtype = np.float64)
        self.mean_rewards = np.array(self.mean_rewards, dtype = np.float64)
        self.total_timesteps = np.array(self.total_timesteps, dtype = np.int32)[:, np.newaxis]
        
        self.weight_vec = list(map(np.squeeze, self.weight_vec))
        self.weight_vec = np.array(self.weight_vec, dtype = np.complex128)
        
        np.savetxt(self.log_dir + "/beamforming_gains.dat", \
                   np.concatenate((self.beamforming_gains, self.total_timesteps), axis = 1), \
                   fmt = '%.18e')
            
        np.savetxt(self.log_dir + "/mean_rewards.dat", \
                   np.concatenate((self.mean_rewards[:, np.newaxis], \
                   self.total_timesteps[::self.n_steps]), axis = 1), \
                   fmt = '%.18e')
        
        np.savetxt(self.log_dir + "/weights.dat", self.weight_vec, fmt = '%.18e')


# Example
if __name__ == "__main__":
    state_0 = env.reset()
    print(f"Random weight vector W: " "\n" \
          f"{np.array2string(np.exp(1j * state_0 * env.phase_shift), precision = 2, floatmode = 'fixed')}")
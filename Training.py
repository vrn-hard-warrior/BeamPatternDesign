# Training RL_model for Beam Pattern Design.
# Dataset is formed by DeepMIMO framework:
# [1]: A. Alkhateeb, “DeepMIMO: A Generic Deep Learning Dataset for Millimeter Wave and Massive MIMO Applications,” 
# in Proc. of The Information Theory and Applications Workshop (ITA), San Diego, CA, Feb. 2019;
# [2]: The Remcom Wireless InSite website: RemCom, Wireless InSite, “https://www.remcom.com/wireless-insite”.
import numpy as np
import DeepMIMO
import os
import json
import RL_model
import Customs
from stable_baselines3 import A2C
from stable_baselines3.common.logger import configure


# Read parameters for channel generation
with open(r'data/parameters.json') as json_file:
    channel_parameters = json.load(json_file)
    
# Generate channels for certain parameters
channels_dataset = DeepMIMO.generate_data(channel_parameters)

# Extraction only channels with No Line of Sight condition
nLoS_mask = np.where(channels_dataset[0]['user']['LoS'] == 0)[0]
channels = channels_dataset[0]['user']['channel'][nLoS_mask, :, :, :]

# Preprocessing tensor of channel matrices
channels = np.squeeze(channels.mean(axis = 3))
channels = np.divide(channels, np.linalg.norm(channels, ord = 2, axis = 1)[:, np.newaxis])
channels = np.transpose(channels)

# Create RL_model object
M = 32
N_p = 3
H_ch = channels[:, 9][:, np.newaxis]
logdir = "logs/A2C_v2"
models_dir = "models/A2C_v2"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    
if not os.path.exists(logdir):
    os.makedirs(logdir)
    
# Set up logger
time_steps = 50000
new_logger = configure(logdir, ["stdout", "csv", "tensorboard"])

env = RL_model.RL_model(M = M, N_p = N_p, H_ch = H_ch, max_interactions = time_steps)
env.reset()

# Create the custom callback: check every 1 timestep
check_freq = 1
callback = RL_model.SaveBeamformingGainsCallback(check_freq, logdir)

# Learning parameters and processing
params = {
    'learning_rate': 7e-4,
    'gamma': .99,
    'gae_lambda': 0.0,
    'ent_coef': 0.001,
    'vf_coef': 0.5,
    'n_steps': 5,
    'max_grad_norm': 0.5,
    'rms_prop_eps': 1e-5,
    'use_rms_prop': True,
    'normalize_advantage': False,
    'verbose': 1
}

model = A2C(Customs.ActorCriticPolicy_custom, env, **params)
# model = A2C("MlpPolicy", env, **params)

# Set new logger
model.set_logger(new_logger)

model.learn(total_timesteps = time_steps, callback = callback)

model.save(f"{models_dir}/{time_steps}")
np.save(f"{models_dir}/channels", H_ch)
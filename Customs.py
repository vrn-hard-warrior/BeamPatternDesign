# Creation of custom learning algorithms, policies, buffers 
# based on Stable-Baselines3 package. In this version:
# [1]: Custom networks for Actor-Critic algorithm.
import torch as th
from gym import spaces
from typing import Callable, Tuple
from stable_baselines3.common.policies import ActorCriticPolicy


class Network_custom(th.nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.
    """
    
    def __init__(self,
                 feature_dim: int = 64,
                 hidden_layer_width: int = 64,
                 last_layer_dim_pi: int = 64,
                 last_layer_dim_vf: int = 64):
        super(Network_custom, self).__init__()
        
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        
        # Policy network
        self.policy_net = th.nn.Sequential(
            th.nn.Linear(feature_dim, hidden_layer_width), th.nn.Tanh(),
            # th.nn.BatchNorm1d(hidden_layer_width),
            th.nn.Linear(hidden_layer_width, hidden_layer_width), th.nn.Tanh(),
            # th.nn.BatchNorm1d(hidden_layer_width),
            th.nn.Linear(hidden_layer_width, last_layer_dim_pi), th.nn.Tanh()
        )
        
        # Value network
        self.value_net = th.nn.Sequential(
            th.nn.Linear(feature_dim, hidden_layer_width), th.nn.Tanh(),
            # th.nn.BatchNorm1d(hidden_layer_width),
            th.nn.Linear(hidden_layer_width, hidden_layer_width), th.nn.Tanh(),
            # th.nn.BatchNorm1d(hidden_layer_width),
            th.nn.Linear(hidden_layer_width, last_layer_dim_pi), th.nn.Tanh()
        )
    
    
    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)
    
    
    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)
    
    
    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class ActorCriticPolicy_custom(ActorCriticPolicy):
    """
    Create custom Actor-Critic class.
    """
    
    def __init__(self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs):
        super(ActorCriticPolicy_custom, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs)
        
        # Disable orthogonal initialization
        self.ortho_init = False
    
    
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = Network_custom(self.features_dim)


if __name__ == "__main__":
    print("That's all!")
"""
This is an example submission that can give participants a reference 
"""

# from dac4automlcomp.run_experiments import run_experiment

from dac4automlcomp.policy import DACPolicy
from rlenv.generators import DefaultRLGenerator, RLInstance

from carl.envs import *

import gym
import pdb

import time
import torch
import stable_baselines3


class ZooHyperparams(DACPolicy):
    """
    A policy which checks the instance and applies fixed parameters for PPO
    to the model. The parameters are based on the ones specified in stable_baselines
    zoo (https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml)

    """

    def __init__(self):
        """
        Initialize all the aspects needed for the policy.
        """

        # Set the algorithm that will be used
        self.algorithm = "PPO"

    def _get_zoo_params(self, state):
        """
        Return a set of hyperparameters for the given environment.

        Args:
            env: str
                The name of the environment.

        Returns:
            params: Dict
                The hyperparameters for the environment.
        """
        # import pdb; pdb.set_trace()
        env = self.env
        if env == "CARLPendulumEnv":
            params = {
                "algorithm": "DDPG",
                "learning_rate": 1e-3,
                "gamma": 0.98,
                "buffer_size": 200000,
                "learning_starts": 10000,
                # "noise_type": 'normal',
                # "noise_std": 0.1,
                "action_noise": stable_baselines3.common.noise.NormalActionNoise(0, 0.1),
                "gradient_steps": -1,
                # 'train_freq': (1, "episode"),
                "policy_kwargs": dict(net_arch=[400, 300]),
                # "gae_lambda": 0.95,
                # "ent_coef": 0.0,
                # "batch_size": 64,
                # "n_steps": 1024,
                # "n_epochs": 10,
            }
        elif env == "CARLCartPoleEnv":
            params = {
                "algorithm": "PPO",
                "learning_rate": 0.001,  # should be a schedule, but for this example we use a constant
                "gamma": 0.98,
                "gae_lambda": 0.8,
                "ent_coef": 0.0,
                "n_steps": 32,
                "n_epochs": 10,
                "batch_size": 256,
            }
        elif env == "CARLAcrobotEnv":
            params = {
                "algorithm": "PPO",
                "gamma": 0.99,
                "gae_lambda": 0.94,
                "ent_coef": 0.0,
                "n_epochs": 4,
            }
        elif env == "CARLMountainCarContinuousEnv":
            params = {
                "algorithm": "SAC",
                "learning_rate": 0.000969826,
                "gamma": 0.95,
                'learning_starts' : 10,
                # 'log_std_init': -1.4627, 
                # 'train_freq': 64, 
                # 'gradient_steps': 64,
                # 'ent_coef': 'auto',
                'tau': 0.02,
                'sde_sample_freq': 32,
                # 'target_entropy': 'auto',
                "policy_kwargs": dict(log_std_init=-1.4627, net_arch=[256, 256]),
                "buffer_size": 10000,
                # "gae_lambda": 0.9,
                # "vf_coef": 0.19,
                # "ent_coef": 0.00429,
                # "policy_kwargs": dict(log_std_init=-3.29, ortho_init=False),
                # "max_grad_norm": 5,
                # "n_epochs": 10,
                "batch_size": 256,
                "use_sde": True,
                # "verbose": 1,
            }
        elif env == "CARLLunarLanderEnv":
            params = {
                "algorithm": "PPO",
                'batch_size': 8,
                # 'clip_range': 0.4,
                'ent_coef': 1.11811e-07,
                'gae_lambda': 0.9,
                'gamma': 0.9999,
                "learning_rate": 0.000522198,
                'max_grad_norm': 0.6,
                'sde_sample_freq': 64,
                "n_steps": 2048,
                "n_epochs": 10,
                'vf_coef': 0.887769,
    #             "policy_kwargs": dict(
    #     log_std_init=-0.647632,
    #     net_arch=[dict(pi=[128, 128], vf=[128, 128])],
    #     activation_fn= torch.nn.LeakyReLU,
    #     ortho_init=True,

    # ),
            }

        return params

    def act(self, env):
        """
        Generate an action in the form of the hyperparameters based on
        the given instance

        Args:
            state: Dict
                The state of the environment.

        Returns:
            action: Dict
        """
        # Get the environment from the state
        
        # Get the zoo parameters for hte environment
        zoo_params = self._get_zoo_params(env)

        # Create the action dictionary
        action = {"algorithm": self.algorithm}
        action = {**action, **zoo_params}

        return action

    def reset(self, instance):
        """Reset a policy's internal state.

        The reset method is there to support 'stateful' policies (e.g., LSTM),
        i.e., whose actions are a function not only of the current
        observations, but of the entire observation history from the
        current episode/execution. It is called at the beginning of the
        target algorithm execution (before the first call to act()) and also provides the policy
        with information about the target problem instance being solved.

        Args:
            instance: The problem instance the target algorithm to be configured is currently solving
        """
        self.env = instance.env_type

    def seed(self, seed):
        """Sets random state of the policy.
        Subclasses should implement this method if their policy is stochastic
        """
        pass

    def save(self, path):
        """Saves the policy to given folder path."""
        pass

    @classmethod
    def load(cls, path):
        """Loads the policy from given folder path."""
        pass


if __name__ == "__main__":

    start_time = time.time()

    policy = ZooHyperparams()
    env = gym.make("dac4carl-v0")
    done = False

    state = env.reset()
    env_type = state["env"]

    reward_history = []
    while not done:
        # get the default stat at reset

        init_time = time.time()

        # generate an action
        action = policy.act(env_type)

        # Apply the action t get hte reward, next state and done
        state, reward, done, _ = env.step(action)

        # save the reward
        reward_history.append(reward)
        print("--- %s seconds per instance---" % (time.time() - init_time))

    print(reward_history)

    print("--- %s seconds ---" % (time.time() - start_time))

import gym

from stable_baselines3 import PPO, SAC, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import json

# Parallel environments
env = gym.make('MountainCarContinuous-v0')

# hp = {'batch_size': 256, 'buffer_size': 10000, 
#       'gamma': 0.95, 'learning_starts': 10, 'log_std_init': -1.4627, 
#       'lr': 0.000969826, 'net_arch': [256, 256], 
#       'sde_sample_freq': 32, 'tau': 0.02, 'train_freq': 64, 
#       'ent_coef': 'auto', 'target_entropy': 'auto', 'gradient_steps': 64, 'use_sde': True}

hp = {
                "algorithm": "SAC",
                "lr": 0.000969826,
                "gamma": 0.95,
                'learning_starts' : 10,
                # 'log_std_init': -1.4627, 
                'train_freq': 64, 
                'gradient_steps': 64,
                'ent_coef': 'auto',
                'tau': 0.02,
                'sde_sample_freq': 32,
                'target_entropy': 'auto',
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
            }
# model = SAC("MlpPolicy", env, verbose=1, learning_rate = 9.999999747378752e-06, ent_coef = 0.001, gae_lambda = 0.956157386302948, clip_range = 0.10000000149011612, n_steps = 45264, tensorboard_log="/home/niranth/Desktop/Work/UniFreiburg/tb_logs")
model = SAC(
    "MlpPolicy",
    env,
    gamma=hp["gamma"],
    learning_rate=hp["lr"],
    batch_size=hp["batch_size"],
    buffer_size=hp["buffer_size"],
    learning_starts=hp["learning_starts"],
    train_freq=hp["train_freq"],
    gradient_steps=hp["gradient_steps"],
    ent_coef=hp["ent_coef"],
    tau=hp["tau"],
    use_sde=hp["use_sde"],
    sde_sample_freq=hp["sde_sample_freq"],
    target_entropy=hp["target_entropy"],
    policy_kwargs=hp["policy_kwargs"],
    tensorboard_log="/home/niranth/Desktop/Work/automl_comp/dac4automlcomp",
    verbose=1
)
# for i in range(100):
model.load("ppo_mcc")
model.learn(total_timesteps = 1000, n_eval_episodes = 10)
# mean, sd = evaluate_policy(model, env)
    # avgs.append(mean)
    # sds.append(sd)
    # with open("avg_rewards_bw_hc.txt", 'w') as f:
    #     json.dump(avgs, f)
    
    # with open("sd_rewards_bw_hc.txt", 'w') as f:
    #     json.dump(sds, f)
    # if  i%5 == 0:
    #     model.save("ppo_bipedal_walker_hc") 

model.save("ppo_mcc") 
# del model # remove to demonstrate saving and loading

# model = PPO.load("ppo_cartpole")

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()
import gym
import rlenv

# Create an (outer) env to perform DAC with 3,000 inner env steps and 2 reconfiguration points:
n_epochs = 2
env = gym.make("dac4carl-v0", total_timesteps = 1e3, n_epochs = n_epochs)
env.seed(123)
obs = env.reset()
done = False

print("Environment of sampled instance set to: ", env.current_instance.env_type)

for i in range(n_epochs):

    # Create an action dict containing the algorithm to apply along with its hyperparameter configuration:
    action = {'algorithm': 'PPO', 'learning_rate': 0.001, 'gamma': 0.98, 'gae_lambda': 0.8, 'ent_coef': 0.0, 'n_steps': 32, 'n_epochs': 10, 'batch_size': 256}

    # Apply the desired hyperparameter configs:
    obs, reward, done, info = env.step(action)
    import pdb; pdb.set_trace()

    # obs, reward, done, info = env.step(env.action_space.sample())
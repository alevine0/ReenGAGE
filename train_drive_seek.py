from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from drive_seek import DriveSeek
from multi_reengage_ddpg import GradDDPG
from pooling_td3_policies_with_sharing import PoolingMultiInputPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import random
import numpy as np
import argparse
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.logger import configure
import time
import gym

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

time_sig = str(round(time.time() * 1000)) + str(random.random())[2:]



parser = argparse.ArgumentParser(description='PyTorch example')
parser.add_argument('--gradient_reg', default=0.0, type=float, help='Gradient regularization')
parser.add_argument('--seed', default=0, type=int, help='Seed')

args = parser.parse_args()

env_extra_args = {}
model_extra_args =  {}

policy_class = PoolingMultiInputPolicy
policy_kwargs = {
    "net_arch_head": [400],
    "net_arch_goal_extractor": [400, 400],
    "embedding_dim": 20,
    "max_desired_goals": 200,
    "dim_per_desired_goal": 2,
    "achieved_dim": 2,
    "obs_dim": 4,
    'share_features_extractor' : False
}
env_class = DriveSeek
env_args = {"max_position":10,"max_goals":200,"timesteps":40}

name_mid = str(args.gradient_reg)
model_class = GradDDPG  
buff_class = None
buff_args =None

env = env_class(**{**env_args, **env_extra_args})

max_episode_length = env.timesteps



name =   name_mid+"/"




# Initialize the model
new_logger = configure("./drive_seek/"+name+time_sig+"/training_logs/", ["csv"])


model = model_class(
    policy_class,
    env,
    buffer_size = 1000000,
    replay_buffer_class= buff_class,
    replay_buffer_kwargs= buff_args,
    train_freq= 1,
    action_noise = NormalActionNoise([0.], [0.05]),
    policy_kwargs=policy_kwargs,
    batch_size = 256,
    gamma =0.95,
    learning_rate=0.001,
    learning_starts=1000,
    gradient_reg = args.gradient_reg,
    bypass_zero = False,
    max_desired_goals =  200,
    dim_per_desired_goal = 2,
    seed=args.seed,
    tensorboard_log= "./drive_seek/"+name+time_sig+"/tensorboard_logs/", **model_extra_args
)

model.set_logger(new_logger)

eval_envs =  [(env_class(**{**env_args, **{"render_skip":100,"render_path":"./drive_seek/"+name+time_sig+"/eval_gifs"}}), "")]
eval_envs[0][0].seed(args.seed+1)
eval_callbacks  = list([EvalCallback(x[0], best_model_save_path="./drive_seek/"+name+x[1]+time_sig+"/",
    log_path="./drive_seek/"+name+x[1]+time_sig+"/eval_logs/", eval_freq=max_episode_length*100, n_eval_episodes=100, render=True) for x in eval_envs])
for e in eval_callbacks:
    e._on_training_start=e._on_step


model.learn(4000*max_episode_length, callback=eval_callbacks)


from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import random
import numpy as np
import argparse
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.logger import configure
import time
import gym
from continuous_seek import ContinuousSeek
from reengage_ddpg import GradDDPG

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



parser = argparse.ArgumentParser(description='PyTorch example')
parser.add_argument('--dim', type=int, default=20)
parser.add_argument('--lr', type=float, default=0.00125)
parser.add_argument('--gradient_reg', default=0.0, type=float, help='Gradient regularization')
parser.add_argument('--steps', type=int, default=200000)
parser.add_argument('--batch', type=int, default=256)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--bypass_zero', action = 'store_true') # For testing: ccmputes gradient loss, but sets loss coefficient to zero.

args = parser.parse_args()
time_sig = str(round(time.time() * 1000)) + str(random.random())[2:] + str(args.seed)


env_args = {"n_dimensions" : args.dim}

name_mid = "baseline"
if (args.gradient_reg != 0.0 or args.bypass_zero):
    name_mid = "gradient_reg_" + str(args.gradient_reg)
policy_class = "MultiInputPolicy"

policy_kwargs = {"net_arch" :[256,256], 'share_features_extractor' : False}

model_class = GradDDPG  
max_episode_length = 10

buff_class = HerReplayBuffer
buff_args =dict(
    n_sampled_goal=4,
    goal_selection_strategy="future",
    online_sampling=True,
    max_episode_length=max_episode_length,
)
env = ContinuousSeek(**env_args)



name =   name_mid+"_dim_"+str(args.dim)+"_lr_"+str(args.lr)+"_steps_"+str(args.steps)+"_batch_"+str(args.batch)+"/"
eval_env_extra_args = {}
eval_env_extra_args["render_path"] = "./continuous_seek/"+name+time_sig+"/eval_gifs"
eval_env_extra_args["render_skip"] = 50



# Initialize the model
new_logger = configure("./continuous_seek/"+name+time_sig+"/training_logs/", ["csv"])



model = model_class(
    policy_class,
    env,
    buffer_size = 1000000,
    replay_buffer_class= buff_class,
    replay_buffer_kwargs= buff_args,
    train_freq= 1,
    action_noise = NormalActionNoise([0.]*args.dim, [0.03]*args.dim),
    policy_kwargs=policy_kwargs,
    batch_size = args.batch,
    gamma =0.95,
    learning_rate=args.lr,
    learning_starts=1000,
    gradient_reg = args.gradient_reg,
    bypass_zero = args.bypass_zero,
    seed=args.seed,
    tensorboard_log= "./continuous_seek/"+name+time_sig+"/tensorboard_logs/", 
)

model.set_logger(new_logger)

eval_envs =  [(ContinuousSeek(**{**env_args, **eval_env_extra_args}), "")]
eval_envs[0][0].seed(args.seed+1)
eval_callbacks  = list([EvalCallback(x[0], best_model_save_path="./continuous_seek/"+name+x[1]+time_sig+"/",
    log_path="./continuous_seek/"+name+x[1]+time_sig+"/eval_logs/", eval_freq= 2000, n_eval_episodes=50, render=True) for x in eval_envs])
for e in eval_callbacks:
    e._on_training_start=e._on_step

model.learn(args.steps, callback=eval_callbacks)


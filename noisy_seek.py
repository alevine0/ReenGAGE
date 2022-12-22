from collections import OrderedDict
from typing import Any, Dict, Optional, Union

import numpy as np
import math
from gym import GoalEnv, spaces
from gym.envs.registration import EnvSpec
import torch
from scipy.special import softmax
from stable_baselines3.common.type_aliases import GymStepReturn
from torch.distributions.exponential import Exponential
from torch.nn.functional import leaky_relu
import torch.nn as nn
from PIL import Image
import os

class NoisySeek(GoalEnv):

	spec = EnvSpec("NoisySeek-v0")

	def __init__(
		self,
		max_goals = 200,
		timesteps = 40,
		render_skip = 100,
		cluster_sigma = 2.,
		center_sigma = 10.,
		noise_sigma = 1.,
		speed = 1.,
		clusters_param = 0.15,
		render_path = ""
	):
		super(NoisySeek, self).__init__()
		self.max_goals = max_goals
		self.timesteps = timesteps
		self.speed =speed
		self.render_dones = 0
		self.render_skip = render_skip
		self.render_path = render_path
		self.clusters_param = clusters_param
		self.cluster_sigma =cluster_sigma
		self.center_sigma = center_sigma
		self.noise_sigma = noise_sigma
		self.observation_space = spaces.Dict(
			{
				"observation":  spaces.Box(low = np.array([-np.inf]*2),high = np.array([np.inf]*2) , dtype=np.float32), 
				"achieved_goal":  spaces.Box(low = np.array([-np.inf]*2),high = np.array([np.inf]*2) , dtype=np.int32), 
				"desired_goal":  spaces.Box(low = np.array([[0] +[-np.inf] * 2]* self.max_goals ).reshape(-1), high = np.array([[1] +[np.inf] *  2]* self.max_goals ).reshape(-1),  dtype=np.int32), 
			}
			)
		self.random_gen = np.random.default_rng()
		self.action_space =  spaces.Box(low = np.array([-1.]* 2),high = np.array([1. ]* 2) , dtype=np.float32)

		self.queued_render = []
		self.has_success = False
		self.cur_position = np.zeros(2,dtype=np.float32)
		self.current_step = 0

		self.desired_goal = np.zeros((self.max_goals,2+1), dtype=int)
		sampled_goals = self._sample_clustered_goals()
		num_desired_goals = sampled_goals.shape[0]
		self.desired_goal[:num_desired_goals,0] = 1
		sampled_goals = sampled_goals[np.lexsort(sampled_goals.transpose())]
		self.desired_goal[:num_desired_goals,1:] = sampled_goals



	def _sample_clustered_goals(self):
		num_desired_goals = self.random_gen.integers(low= 1, high = self.max_goals+1)
		num_clusters = self.random_gen.geometric(self.clusters_param)
		cluster_sizes = self.random_gen.multinomial(num_desired_goals,self.random_gen.dirichlet([1.]*num_clusters), size=1)[0]
		goals = []
		for i in range(num_clusters):
			center = self.random_gen.normal(loc=0.,scale=self.center_sigma,size=(2,))
			new_goals = self.random_gen.normal(loc= 0., scale=self.cluster_sigma,size = (cluster_sizes[i],2) )
			new_goals += center[None,:]
			new_goals = np.rint(new_goals)
			goals.append(new_goals)
		goals = np.concatenate(goals,axis = 0)
		goals = np.unique(goals, axis =0)
		return goals
	def seed(self, seed: int) -> None:
		self.random_gen = np.random.default_rng(seed)

	def _get_obs(self) -> Dict[str, Union[int, np.ndarray]]:
		"""
		Helper to create the observation.
		:return: The current observation.
		"""
		return OrderedDict(
			[
				("observation",  self.cur_position.copy().astype(np.float32)),
				("achieved_goal",  np.rint(self.cur_position).astype(np.float32) ),
				("desired_goal", self.desired_goal.reshape(-1).astype(np.float32) ),
			]
		)
	def reset(self) -> Dict[str, Union[int, np.ndarray]]:
		self.queued_render = []
		self.has_success = False
		self.cur_position = np.zeros(2,dtype=np.float32)
		self.current_step = 0

		self.desired_goal = np.zeros((self.max_goals,2+1), dtype=int)
		sampled_goals = self._sample_clustered_goals()
		num_desired_goals = sampled_goals.shape[0]
		self.desired_goal[:num_desired_goals,0] = 1
		sampled_goals = sampled_goals[np.lexsort(sampled_goals.transpose())]
		self.desired_goal[:num_desired_goals,1:] = sampled_goals
		obs = self._get_obs()
		#print(obs)
		return obs

	def step(self, action) -> GymStepReturn:

		norm_action = np.linalg.norm(action)
		if (norm_action > 1.):
			self.cur_position += action / norm_action
		else:
			self.cur_position += action 
		self.cur_position += self.random_gen.normal(loc=0.,scale=self.noise_sigma,size=(2,))
		self.current_step += 1

		obs = self._get_obs()

		reward = float(self.compute_reward(obs["achieved_goal"], obs["desired_goal"], None))
		done = self.current_step >= self.timesteps
		if (done):
			self.render_dones += 1
		self.has_success = self.has_success  or reward == 1
		info = {"is_success": self.has_success, "TimeLimit.truncated": done}
		return obs, reward, done, info

	def compute_reward(
		self, achieved_goal: np.ndarray, desired_goal:  np.ndarray, _info: Optional[Dict[str, Any]]
	) -> np.float32:


		desired_goal = desired_goal.reshape( desired_goal.shape[:-1] + (self.max_goals,2+1,))
		achieved_goal = achieved_goal.reshape( achieved_goal.shape[:-1] + (2,))
		achieved_goal = achieved_goal[...,None,:]
		return ((desired_goal[...,1:] == achieved_goal).all(axis=-1) &  (desired_goal[...,0] == 1)).any(axis=-1).astype(np.float32) 

	def compute_reward_explanation(
		self, achieved_goal: np.ndarray, desired_goal:  np.ndarray, _info: Optional[Dict[str, Any]]
	) -> np.float32:


		desired_goal = desired_goal.reshape( desired_goal.shape[:-1] + (self.max_goals,2+1,))
		achieved_goal = achieved_goal.reshape( achieved_goal.shape[:-1] + (2,))
		achieved_goal = achieved_goal[...,None,:]
		return ((desired_goal[...,1:] == achieved_goal).all(axis=-1) &  (desired_goal[...,0] == 1)).astype(np.float32)

	def compute_reward_explanation_torch(
		self, achieved_goal, desired_goal
	) -> np.float32:
		desired_goal = desired_goal.reshape( desired_goal.shape[:-1] + (self.max_goals,2+1,))
		achieved_goal = achieved_goal.reshape( achieved_goal.shape[:-1] + (2,))
		achieved_goal = achieved_goal[...,None,:]
		return ((desired_goal[...,1:] == achieved_goal).all(-1) &  (desired_goal[...,0] == 1)).float()

	def render(self, mode: str = "human") -> Optional[np.ndarray]:
		if (self.render_dones % self.render_skip  == 0 ):
			icon_size = 4
			offset = np.abs(self.desired_goal).max()
			resolution = icon_size*(offset*2+1)

			frame = np.ones((resolution,resolution, 3), dtype = np.uint8) * 255
			render_pos = np.rint(self.cur_position).astype(int)
			frame[(render_pos[0] + offset+ 1)* icon_size:(render_pos[0] + offset + 2)* icon_size,(render_pos[1] + offset+ 1)* icon_size:(render_pos[1] + offset + 2)* icon_size, 1] = 0

			frame[0:icon_size,0:icon_size,2] = 255-5*len(self.queued_render)



			for i in range(self.max_goals):
				if (self.desired_goal[i,0] == 1):
					frame[(self.desired_goal[i,1] + offset+ 1)* icon_size:(self.desired_goal[i,1] + offset+ 2)* icon_size,(self.desired_goal[i,2] + offset+ 1)* icon_size:(self.desired_goal[i,2] + offset + 2)* icon_size, 0] =  0

			#frame[0:icon_size*2+2, 0:icon_size*2+2,2] = 5*len(self.queued_render)
			self.queued_render.append(Image.fromarray(frame))
			if (self.current_step == self.timesteps -1):
				os.makedirs(os.path.dirname(self.render_path + '/' + str(self.render_dones) + '.gif'), exist_ok=True)
				self.queued_render[0].save(self.render_path + '/' + str(self.render_dones) + '.gif',save_all=True,  append_images=self.queued_render[1:],optimize=False)
				self.queued_render = []

	def close(self) -> None:
		pass
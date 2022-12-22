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

class DriveSeek(GoalEnv):

	spec = EnvSpec("DriveSeek-v0")

	def __init__(
		self,
		max_position = 10,
		max_goals = 200,
		timesteps = 40,
		render_skip = 100,
		angular_speed = 0.5,
		speed = 1.,
		render_path = ""
	):
		super(DriveSeek, self).__init__()

		self.max_position = max_position
		self.max_goals = max_goals
		self.timesteps = timesteps
		self.angular_speed = angular_speed
		self.speed =speed
		self.slack = 0.5
		self.render_dones = 0
		self.render_skip = render_skip
		self.render_path = render_path
		self.observation_space = spaces.Dict(
			{
				"observation":  spaces.Box(low = np.array([-self.max_position-self.slack, -self.max_position-self.slack, -1., -1. ]),high = np.array([self.max_position+self.slack, self.max_position+self.slack, 1., 1. ]) , dtype=np.float32), 
				"achieved_goal":  spaces.Box(-self.max_position,self.max_position, shape=(2,), dtype=np.int8), 
				"desired_goal":  spaces.Box(low = np.array([[0] +[-self.max_position] * 2]* self.max_goals ).reshape(-1), high = np.array([[1] +[self.max_position] * 2]* self.max_goals ).reshape(-1),  dtype=np.int8), 
			}
			)
		self.random_gen = np.random.default_rng()
		self.action_space =  spaces.Box(low = np.array([-1.]),high = np.array([1. ]) , dtype=np.float32)

		self.queued_render = []
		self.has_success = False
		self.cur_position = np.zeros(2,dtype=np.float32)
		self.current_step = 0
		self.cur_angle = 0
		self.desired_goal = np.zeros((self.max_goals,2+1), dtype=int)
		num_desired_goals = self.random_gen.integers(low= 1, high = self.max_goals+1)
		sampled_goals = self.random_gen.choice( np.array(np.ones((1+2*self.max_position,)*2).nonzero()).transpose()-self.max_position,size=num_desired_goals,replace=False,shuffle=False)
		self.desired_goal[:num_desired_goals,0] = 1
		sampled_goals = sampled_goals[np.lexsort(sampled_goals.transpose())]
		self.desired_goal[:num_desired_goals,1:] = sampled_goals


	def seed(self, seed: int) -> None:
		self.random_gen = np.random.default_rng(seed)

	def _get_obs(self) -> Dict[str, Union[int, np.ndarray]]:
		"""
		Helper to create the observation.
		:return: The current observation.
		"""
		return OrderedDict(
			[
				("observation",  np.concatenate([self.cur_position, np.array([math.cos(self.cur_angle), math.sin(self.cur_angle)])]).astype(np.float32)),
				("achieved_goal",  np.rint(self.cur_position).astype(np.float32) ),
				("desired_goal", self.desired_goal.reshape(-1).astype(np.float32) ),
			]
		)
	def reset(self) -> Dict[str, Union[int, np.ndarray]]:
		self.queued_render = []
		self.has_success = False
		self.cur_position = np.zeros(2,dtype=np.float32)
		self.current_step = 0
		self.cur_angle = 0
		self.desired_goal = np.zeros((self.max_goals,2+1), dtype=int)
		num_desired_goals = self.random_gen.integers(low= 1, high = self.max_goals+1)
		sampled_goals = self.random_gen.choice( np.array(np.ones((1+2*self.max_position,)*2).nonzero()).transpose()-self.max_position,size=num_desired_goals,replace=False,shuffle=False)
		self.desired_goal[:num_desired_goals,0] = 1
		sampled_goals = sampled_goals[np.lexsort(sampled_goals.transpose())]
		self.desired_goal[:num_desired_goals,1:] = sampled_goals
		obs = self._get_obs()
		#print(obs)
		return obs

	def step(self, action) -> GymStepReturn:

		self.cur_angle += action*self.angular_speed


		self.cur_position += np.array([math.cos(self.cur_angle), math.sin(self.cur_angle)])*self.speed
		self.cur_position  = (self.cur_position + self.max_position + self.slack)%(2*self.max_position + 1) - (self.max_position + self.slack)
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

		desired_goal = desired_goal.reshape( desired_goal.shape[:-1] + (self.max_goals,3,))
		achieved_goal = achieved_goal.reshape( achieved_goal.shape[:-1] + (2,))
		achieved_goal = achieved_goal[...,None,:]
		return ((desired_goal[...,1:] == achieved_goal).all(axis=-1) &  (desired_goal[...,0] == 1)).any(axis=-1).astype(np.float32) 

	def compute_reward_explanation(
		self, achieved_goal: np.ndarray, desired_goal:  np.ndarray, _info: Optional[Dict[str, Any]]
	) -> np.float32:


		desired_goal = desired_goal.reshape( desired_goal.shape[:-1] + (self.max_goals,3,))
		achieved_goal = achieved_goal.reshape( achieved_goal.shape[:-1] + (2,))
		achieved_goal = achieved_goal[...,None,:]
		return ((desired_goal[...,1:] == achieved_goal).all(axis=-1) &  (desired_goal[...,0] == 1)).astype(np.float32)

	def compute_reward_explanation_torch(
		self, achieved_goal, desired_goal
	) -> np.float32:
		desired_goal = desired_goal.reshape( desired_goal.shape[:-1] + (self.max_goals,3,))
		achieved_goal = achieved_goal.reshape( achieved_goal.shape[:-1] + (2,))
		achieved_goal = achieved_goal[...,None,:]
		return ((desired_goal[...,1:] == achieved_goal).all(-1) &  (desired_goal[...,0] == 1)).float()


	def render(self, mode: str = "human") -> Optional[np.ndarray]:
		if (self.render_dones % self.render_skip  == 0 ):
			icon_size = 4
			render_slack = 0

			resolution=icon_size*(2*(self.max_position+render_slack) + 3)
			frame = np.ones((resolution,resolution, 3), dtype = np.uint8) * 255
			render_pos = np.rint(self.cur_position).astype(int)
			frame[(render_pos[0] + self.max_position +render_slack+ 1)* icon_size:(render_pos[0] + self.max_position +render_slack+ 2)* icon_size,(render_pos[1] + self.max_position +render_slack+ 1)* icon_size:(render_pos[1] + self.max_position+render_slack + 2)* icon_size, 1] = 0


			frame[(render_slack)* icon_size:(render_slack+1)* icon_size,(render_slack)* icon_size:(render_slack + 2*self.max_position + 3)* icon_size, 2] = 0
			frame[(render_slack)* icon_size:(render_slack + 2*self.max_position + 3)* icon_size,(render_slack)* icon_size:(render_slack+1)* icon_size, 2] = 0
			frame[(render_slack + 2*self.max_position + 2)* icon_size:(render_slack + 2*self.max_position + 3)* icon_size,(render_slack)* icon_size:(render_slack + 2*self.max_position + 3)* icon_size, 2] = 0
			frame[(render_slack)* icon_size:(render_slack + 2*self.max_position + 3)* icon_size,(render_slack + 2*self.max_position + 2)* icon_size:(render_slack + 2*self.max_position + 3)* icon_size, 2] = 0



			for i in range(self.max_goals):
				if (self.desired_goal[i,0] == 1):
					frame[(self.desired_goal[i,1] + self.max_position +render_slack+ 1)* icon_size:(self.desired_goal[i,1] + self.max_position +render_slack+ 2)* icon_size,(self.desired_goal[i,2] + self.max_position +render_slack+ 1)* icon_size:(self.desired_goal[i,2] + self.max_position+render_slack + 2)* icon_size, 0] =  0

			#frame[0:icon_size*2+2, 0:icon_size*2+2,2] = 5*len(self.queued_render)
			self.queued_render.append(Image.fromarray(frame))
			if (self.current_step == self.timesteps -1):
				os.makedirs(os.path.dirname(self.render_path + '/' + str(self.render_dones) + '.gif'), exist_ok=True)
				self.queued_render[0].save(self.render_path + '/' + str(self.render_dones) + '.gif',save_all=True,  append_images=self.queued_render[1:],optimize=False)
				self.queued_render = []

	def close(self) -> None:
		pass
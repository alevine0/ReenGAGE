from collections import OrderedDict
from typing import Any, Dict, Optional, Union

import numpy as np
import math
from gym import GoalEnv, spaces
from gym.envs.registration import EnvSpec
import torch
from stable_baselines3.common.type_aliases import GymStepReturn
import torch.nn as nn
from PIL import Image
import os


class ContinuousSeek(GoalEnv):

	spec = EnvSpec("ContinuousSeek-v0")

	def __init__(
		self,
		n_dimensions = 2,
		max_size = 5,
		max_steps = 10,
		epsilon = 0.1,
		render_skip = 20,
		render_path = ""
	):
		super(ContinuousSeek, self).__init__()

		self.n_dimensions = n_dimensions
		self.observation_space = spaces.Dict(
			{
				"observation":   spaces.Box(low=-max_size, high = max_size,  shape=(self.n_dimensions,), dtype=np.float32),
				"achieved_goal": spaces.Box(low=-max_size, high = max_size,  shape=(self.n_dimensions,), dtype=np.float32),
				"desired_goal": spaces.Box(low=-max_size, high = max_size,  shape=(self.n_dimensions,), dtype=np.float32),
			}
				)
		self.max_steps = max_steps
		self.max_size = max_size
		self.epsilon = epsilon
		self.random_gen = np.random.default_rng()
		self.action_space =  spaces.Box(low=-1., high = 1.,  shape=(self.n_dimensions,), dtype=np.float32)
		self.desired_goal = (2*self.random_gen.random( size=(self.n_dimensions,)) -1)*self.max_size
		self.state = np.zeros((self.n_dimensions,), dtype=np.float32)
		self.current_step = 0
		self.render_dones = 0
		self.render_skip = render_skip
		self.render_path = render_path
		self.has_success = False
		self.queued_render = []
	def seed(self, seed: int) -> None:
		self.random_gen = np.random.default_rng(seed)

	def _get_obs(self) -> Dict[str, Union[int, np.ndarray]]:
		"""
		Helper to create the observation.
		:return: The current observation.
		"""
		return OrderedDict(
			[
				("observation",  self.state.astype(np.float32) ),
				("achieved_goal", self.state.astype(np.float32) ),
				("desired_goal", self.desired_goal.astype(np.float32) ),
			]
		)
	def reset(self) -> Dict[str, Union[int, np.ndarray]]:
		self.state = np.zeros((self.n_dimensions,))
		self.current_step = 0


		self.has_success = False
		self.desired_goal = (2*self.random_gen.random( size=(self.n_dimensions,)) -1)*self.max_size

		self.queued_render = []
		obs = self._get_obs()
		return obs
	def step(self, action) -> GymStepReturn:
		self.touched=True
		action = np.clip(action, -1., 1.)
		self.state = np.clip(self.state + action, -self.max_size, self.max_size)

		self.current_step += 1

		obs = self._get_obs()
		reward = float(self.compute_reward(obs["achieved_goal"], obs["desired_goal"], None))
		done = self.current_step >= self.max_steps
		if (done):
			self.render_dones += 1
		self.has_success = self.has_success  or reward == 0
		info = {"is_success": self.has_success, "TimeLimit.truncated": done}
		return obs, reward, done, info
	def render(self, mode: str = "human") :
		#print(self._get_obs())

		if (self.render_dones % self.render_skip  == 0):
			resolution=250
			frame = np.ones((resolution,resolution, 3), dtype = np.uint8) * 255
			frame[max(0,int((self.state[0]+self.max_size - self.epsilon/2.)*resolution/(2.*self.max_size))):min(resolution,int((self.state[0]+self.max_size + self.epsilon/2.)*resolution/(2.*self.max_size)) + 1),  max(0,int((self.state[1]+self.max_size - self.epsilon/2.)*resolution/(2.*self.max_size))):min(resolution,int((self.state[1]+self.max_size + self.epsilon/2.)*resolution/(2.*self.max_size)) + 1)  , 1] = 0
			frame[max(0,int((self.desired_goal[0]+self.max_size - self.epsilon/2.)*resolution/(2.*self.max_size))):min(resolution,int((self.desired_goal[0]+self.max_size + self.epsilon/2.)*resolution/(2.*self.max_size)) + 1),  max(0,int((self.desired_goal[1]+self.max_size - self.epsilon/2.)*resolution/(2.*self.max_size))):min(resolution,int((self.desired_goal[1]+self.max_size + self.epsilon/2.)*resolution/(2.*self.max_size)) + 1) ,  0] = 0
			frame[0:2, 0:2,2] = 5*len(self.queued_render)
			self.queued_render.append(Image.fromarray(frame))
			if (self.current_step == self.max_steps -1):
				os.makedirs(os.path.dirname(self.render_path + '/' + str(self.render_dones) + '.gif'), exist_ok=True)
				self.queued_render[0].save(self.render_path + '/' + str(self.render_dones) + '.gif',save_all=True,  append_images=self.queued_render[1:],optimize=False)
				self.queued_render = []

	def compute_reward(
		self, achieved_goal: np.ndarray, desired_goal:  np.ndarray, _info: Optional[Dict[str, Any]]
	) -> np.float32:

		return -((np.linalg.norm(desired_goal-achieved_goal, ord = np.inf,axis=-1) >  self.epsilon )).astype(np.float32)

	def close(self) -> None:
		pass
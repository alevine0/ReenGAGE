from typing import Any, Dict, Optional, Tuple, Type, Union

import torch as th
import numpy as np

from torch.nn import functional as F
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3.td3.td3 import TD3
from stable_baselines3.common.utils import polyak_update


class GradDDPG(TD3):
    """
    Deep Deterministic Policy Gradient (DDPG).

    Deterministic Policy Gradient: http://proceedings.mlr.press/v32/silver14.pdf
    DDPG Paper: https://arxiv.org/abs/1509.02971
    Introduction to DDPG: https://spinningup.openai.com/en/latest/algorithms/ddpg.html

    Note: we treat DDPG as a special case of its successor TD3.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[TD3Policy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 100,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "episode"),
        gradient_steps: int = -1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        gradient_reg: float = 0.0,
        bypass_zero = False,
        max_desired_goals = 50,
        dim_per_desired_goal = 2,
        _init_setup_model: bool = True,
    ):

        super(GradDDPG, self).__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            optimize_memory_usage=optimize_memory_usage,
            # Remove all tricks from TD3 to obtain DDPG:
            # we still need to specify target_policy_noise > 0 to avoid errors
            policy_delay=1,
            target_noise_clip=0.0,
            target_policy_noise=0.1,
            _init_setup_model=False,
        )
        self.gradient_reg = gradient_reg
        self.bypass_zero= bypass_zero
        self.max_desired_goals= max_desired_goals
        self.dim_per_desired_goal = dim_per_desired_goal
        # Use only one critic
        if "n_critics" not in self.policy_kwargs:
            self.policy_kwargs["n_critics"] = 1

        if _init_setup_model:
            self._setup_model()

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []

        for _ in range(gradient_steps):

            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            if self.gradient_reg == 0.0 and not self.bypass_zero:

                with th.no_grad():
                    # Select action according to policy and add clipped noise
                    noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                    noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                    next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                    # Compute the next Q-values: min over all critics targets
                    next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                    next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                    target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

                # Get current Q-values estimates for each critic network
                current_q_values = self.critic(replay_data.observations, replay_data.actions)

                # Compute critic loss
                critic_loss = sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values])
                critic_losses.append(critic_loss.item())
            else:
                batch_dim = replay_data.next_observations['desired_goal'].shape[0]
                next_obs_full_desired = replay_data.next_observations['desired_goal'].reshape(batch_dim,self.max_desired_goals,self.dim_per_desired_goal+1)
                diff_next_obs_var = next_obs_full_desired[:,:,:1].float().detach().requires_grad_(True)
                diff_next_obs = {'achieved_goal': replay_data.next_observations['achieved_goal'],
                'desired_goal':  th.cat([diff_next_obs_var, next_obs_full_desired[:,:,1:]], dim = -1).reshape(batch_dim, -1) ,
                'observation': replay_data.next_observations['observation']}

                # Select action according to policy and add clipped noise
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(diff_next_obs) + noise).clamp(-1, 1)

                # Compute the next Q-values: min over all critics targets
                next_q_values = th.cat(self.critic_target(diff_next_obs, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
                reward_dummy_gradient = 2.* th.tensor(self.env.env_method(
                    "compute_reward_explanation_torch",
                    replay_data.next_observations['achieved_goal'],
                    replay_data.next_observations['desired_goal']
                )[0]).unsqueeze(-1)
                next_gradient = th.autograd.grad(target_q_values.sum(), diff_next_obs_var, create_graph=True)[0] 
                next_gradient = next_gradient + reward_dummy_gradient
                current_obs_full_desired = replay_data.observations['desired_goal'].reshape(batch_dim,self.max_desired_goals,self.dim_per_desired_goal+1)
                diff_current_obs_var =  current_obs_full_desired[:,:,:1].float().detach().requires_grad_(True)
                diff_current_obs = {'achieved_goal': replay_data.observations['achieved_goal'],
                'desired_goal':  th.cat([diff_current_obs_var, current_obs_full_desired[:,:,1:]], dim = -1).reshape(batch_dim, -1),
                'observation': replay_data.observations['observation']}

                # Get current Q-values estimates for each critic network
                current_q_values = self.critic(diff_current_obs, replay_data.actions)

                # Compute critic loss
                crit_losses = []
                for current_q in current_q_values:
                    mse_ls = F.mse_loss(current_q, target_q_values)
                    current_grad = th.autograd.grad(current_q.sum(), diff_current_obs_var, create_graph=True)[0]
                    sal_ls =  self.gradient_reg * F.mse_loss(current_grad, next_gradient, reduction = 'none').mean()
                    # if (gradient_step == 0):
                    #     print(mse_ls)
                    #     print(sal_ls)
                    ls = mse_ls + sal_ls
                    crit_losses.append(ls)

                critic_loss = sum(crit_losses)
                critic_losses.append(critic_loss.item())


            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss
                actor_loss = -self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations)).mean()
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))



    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "DDPG",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:

        return super(GradDDPG, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )

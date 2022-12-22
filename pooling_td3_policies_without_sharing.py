from typing import Any, Dict, List, Optional, Type, Union, Tuple

import gym
import torch as th
from torch import nn
from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3.common.policies import BaseModel, BasePolicy, ContinuousCritic, register_policy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)
from stable_baselines3.common.type_aliases import Schedule

class PoolingContinuousCritic(BaseModel):
    """
    Critic network(s) for DDPG/SAC/TD3.
    It represents the action-state value function (Q-value function).
    Compared to A2C/PPO critics, this one represents the Q-value
    and takes the continuous action as input. It is concatenated with the state
    and then fed to the network which outputs a single value: Q(s, a).
    For more recent algorithms like SAC/TD3, multiple networks
    are created to give different estimates.

    By default, it creates two critic networks used to reduce overestimation
    thanks to clipped Q-learning (cf TD3 paper).

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether the features extractor is shared or not
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch_head: List[int],
        net_arch_goal_extractor: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        state_to_head = True,
        state_to_goal_extractor = True,
        embedding_dim = 20,
        max_desired_goals = 50,
        dim_per_desired_goal = 2,
        achieved_dim = 2,
        obs_dim = 4,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )
        self.state_to_head = state_to_head
        self.state_to_goal_extractor = state_to_goal_extractor
        self.embedding_dim = embedding_dim
        self.net_arch_goal_extractor = net_arch_goal_extractor
        self.max_desired_goals = max_desired_goals
        self.dim_per_desired_goal = dim_per_desired_goal
        self.achieved_dim = achieved_dim
        self.obs_dim= obs_dim 
        self.action_dim = get_action_dim(self.action_space)

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.embedding_q_networks = []
        self.head_q_networks = []
        state_action_size = self.achieved_dim + self.obs_dim + self.action_dim
        state_size = self.achieved_dim + self.obs_dim

        for idx in range(n_critics):
            embedding_q_net = create_mlp( self.dim_per_desired_goal + (state_size if self.state_to_goal_extractor else 0), self.embedding_dim, net_arch_goal_extractor, activation_fn)
            embedding_q_net = nn.Sequential(*embedding_q_net)
            self.add_module(f"qef{idx}", embedding_q_net)
            self.embedding_q_networks.append(embedding_q_net)
            head_q_net = create_mlp( self.embedding_dim + (state_action_size if self.state_to_head else 0), 1, net_arch_head, activation_fn)
            head_q_net = nn.Sequential(*head_q_net)
            self.add_module(f"qhf{idx}", head_q_net)
            self.head_q_networks.append(head_q_net)
    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs)
        non_goal = th.cat([features[:,:self.achieved_dim],features[:,-self.obs_dim:]], dim = 1) 
        goals = features[:,self.achieved_dim:-self.obs_dim].reshape((features.shape[0], self.max_desired_goals,self.dim_per_desired_goal + 1 ))
        pure_goals = goals[:,:,1:]
        goal_indicators = goals[:,:,0]
        extractor_input = pure_goals
        if (self.state_to_goal_extractor):
            extractor_input = th.cat([extractor_input, non_goal.unsqueeze(1).expand(-1,self.max_desired_goals, -1)], dim = -1)
        outputs = []
        for idx in range(self.n_critics):
            intermediate = self.embedding_q_networks[idx](extractor_input.reshape((features.shape[0]*self.max_desired_goals, -1,))).reshape((features.shape[0],self.max_desired_goals, -1))
            intermediate = intermediate*(goal_indicators.unsqueeze(-1)**2)
            intermediate = intermediate.sum(dim = 1)
            intermediate = intermediate/self.max_desired_goals
            if (self.state_to_head):
                intermediate = th.cat([intermediate,non_goal,actions], dim= -1)
            outputs.append(self.head_q_networks[idx](intermediate))
        return tuple(outputs)


    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs)
        non_goal = th.cat([features[:,:self.achieved_dim],features[:,-self.obs_dim:]], dim = 1) 
        goals = features[:,self.achieved_dim:-self.obs_dim].reshape((features.shape[0], self.max_desired_goals,self.dim_per_desired_goal + 1 ))
        pure_goals = goals[:,:,1:]
        goal_indicators = goals[:,:,0]
        extractor_input = pure_goals
        if (self.state_to_goal_extractor):
            extractor_input = th.cat([extractor_input, non_goal.unsqueeze(1).expand(-1,self.max_desired_goals, -1)], dim = -1)
        intermediate = self.embedding_q_networks[0](extractor_input.reshape((features.shape[0]*self.max_desired_goals, -1,))).reshape((features.shape[0],self.max_desired_goals, -1))
        intermediate = intermediate*goal_indicators.unsqueeze(-1)
        intermediate = intermediate.sum(dim = 1)
        intermediate = intermediate/self.max_desired_goals
        if (self.state_to_head):
            intermediate = th.cat([intermediate,non_goal,actions], dim= -1)
        return self.head_q_networks[0](intermediate)




class PoolingActor(BasePolicy):
    """
    Actor network (policy) for TD3.

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch_head: List[int],
        net_arch_goal_extractor: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        state_to_head = True,
        state_to_goal_extractor = True,
        embedding_dim = 20,
        max_desired_goals = 50,
        dim_per_desired_goal = 2,
        achieved_dim = 2,
        obs_dim = 4,
        normalize_images: bool = True,
    ):
        super(PoolingActor, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        self.net_arch_head = net_arch_head
        self.features_dim = features_dim
        self.activation_fn = activation_fn

        self.state_to_head = state_to_head
        self.state_to_goal_extractor = state_to_goal_extractor
        self.embedding_dim = embedding_dim
        self.net_arch_goal_extractor = net_arch_goal_extractor
        self.max_desired_goals = max_desired_goals
        self.dim_per_desired_goal = dim_per_desired_goal
        self.achieved_dim = achieved_dim
        self.obs_dim= obs_dim 
        self.action_dim = get_action_dim(self.action_space)
        state_size = self.achieved_dim + self.obs_dim
        embedding_actor_net = create_mlp( self.dim_per_desired_goal + (state_size if self.state_to_goal_extractor else 0), self.embedding_dim, net_arch_goal_extractor, activation_fn)
        self.embedding_mu = nn.Sequential(*embedding_actor_net)
        head_actor_net = create_mlp( self.embedding_dim + (state_size if self.state_to_head else 0), self.action_dim, net_arch_head, activation_fn, squash_output=True)
        self.head_mu = nn.Sequential(*head_actor_net)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch_head = self.net_arch_head,
                net_arch_goal_extractor = self.net_arch_goal_extractor,
                state_to_head=self.state_to_head,
                state_to_goal_extractor = self.state_to_goal_extractor,
                embedding_dim = self.embedding_dim,
                max_desired_goals =self.max_desired_goals,
                dim_per_desired_goal =self.dim_per_desired_goal,
                achieved_dim =self.achieved_dim,
                obs_dim = self.obs_dim,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data

    def forward(self, obs: th.Tensor) -> th.Tensor:
        # assert deterministic, 'The TD3 actor only outputs deterministic actions'
        features = self.extract_features(obs)
        non_goal = th.cat([features[:,:self.achieved_dim],features[:,-self.obs_dim:]], dim = 1) 
        goals = features[:,self.achieved_dim:-self.obs_dim].reshape((features.shape[0], self.max_desired_goals,self.dim_per_desired_goal + 1 ))
        pure_goals = goals[:,:,1:]
        goal_indicators = goals[:,:,0]
        extractor_input = pure_goals
        if (self.state_to_goal_extractor):
            extractor_input = th.cat([extractor_input, non_goal.unsqueeze(1).expand(-1,self.max_desired_goals, -1)], dim = -1)
        intermediate = self.embedding_mu(extractor_input.reshape((features.shape[0]*self.max_desired_goals, -1,))).reshape((features.shape[0],self.max_desired_goals, -1))
        intermediate = intermediate*goal_indicators.unsqueeze(-1)
        intermediate = intermediate.sum(dim = 1)
        intermediate = intermediate/self.max_desired_goals
        if (self.state_to_head):
            intermediate = th.cat([intermediate,non_goal], dim= -1)
        return self.head_mu(intermediate)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Note: the deterministic deterministic parameter is ignored in the case of TD3.
        #   Predictions are always deterministic.
        return self.forward(observation)


class PoolingTD3Policy(TD3Policy):
    """
    Policy class (with both actor and critic) for TD3.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch_head: List[int],
        net_arch_goal_extractor: List[int],
        state_to_head = True,
        state_to_goal_extractor = True,
        embedding_dim = 20,
        max_desired_goals = 50,
        dim_per_desired_goal = 2,
        achieved_dim = 2,
        obs_dim = 4,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        BasePolicy.__init__(self,
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
        )
        if net_arch_head is None:
            net_arch_head = [128]
        if net_arch_goal_extractor is None:
            net_arch_goal_extractor = [128, 128]
        # Default network architecture, from the original paper
        # if net_arch is None:
        #     if features_extractor_class == NatureCNN:
        #         net_arch = []
        #     else:
        #         net_arch = [400, 300]

        actor_arch_head, critic_arch_head = get_actor_critic_arch(net_arch_head)
        actor_arch_goal_extractor, critic_arch_goal_extractor = get_actor_critic_arch(net_arch_goal_extractor)

        self.net_arch_head = net_arch_head
        self.net_arch_goal_extractor = net_arch_goal_extractor
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
            "net_arch_head" : actor_arch_head,
            "net_arch_goal_extractor" : actor_arch_goal_extractor,
            "state_to_head" : state_to_head,
            "state_to_goal_extractor" : state_to_goal_extractor,
            "embedding_dim" : embedding_dim,
            "max_desired_goals" : max_desired_goals,
            "dim_per_desired_goal" : dim_per_desired_goal,
            "achieved_dim" :achieved_dim,
            "obs_dim" :obs_dim,
        }
        self.actor_kwargs = self.net_args.copy()
        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "net_arch_head" : critic_arch_head,
                "net_arch_goal_extractor" : critic_arch_goal_extractor,
                "n_critics": n_critics,
                "share_features_extractor": share_features_extractor,
            }
        )

        self.actor, self.actor_target = None, None
        self.critic, self.critic_target = None, None
        self.share_features_extractor = share_features_extractor

        self._build(lr_schedule)
    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = BasePolicy._get_constructor_parameters(self)

        data.update(
            dict(
                net_arch_head=self.net_args["net_arch_head"],
                net_arch_goal_extractor=self.net_args["net_arch_goal_extractor"],
                state_to_head =self.net_args["state_to_head"],
                state_to_goal_extractor =self.net_args["state_to_goal_extractor"],
                embedding_dim =self.net_args["embedding_dim"],
                max_desired_goals =self.net_args["max_desired_goals"],
                dim_per_desired_goal=self.net_args["dim_per_desired_goal"],
                achieved_dim =self.net_args["achieved_dim"],
                obs_dim =self.net_args["obs_dim"],
                activation_fn=self.net_args["activation_fn"],
                n_critics=self.critic_kwargs["n_critics"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
                share_features_extractor=self.share_features_extractor,
            )
        )
        return data

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> PoolingActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return PoolingActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> PoolingContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return PoolingContinuousCritic(**critic_kwargs).to(self.device)





PoolingMlpPolicy = PoolingTD3Policy


class PoolingCnnPolicy(PoolingTD3Policy):
    """
    Policy class (with both actor and critic) for TD3.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch_head: List[int],
        net_arch_goal_extractor: List[int],
        state_to_head = True,
        state_to_goal_extractor = True,
        embedding_dim = 20,
        max_desired_goals = 50,
        dim_per_desired_goal = 2,
        achieved_dim = 2,
        obs_dim = 4,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super(PoolingCnnPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch_head,
            net_arch_goal_extractor,
            state_to_head,
            state_to_goal_extractor,
            embedding_dim,
            max_desired_goals,
            dim_per_desired_goal,
            achieved_dim,
            obs_dim,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )


class PoolingMultiInputPolicy(PoolingTD3Policy):
    """
    Policy class (with both actor and critic) for TD3 to be used with Dict observation spaces.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch_head: List[int],
        net_arch_goal_extractor: List[int],
        state_to_head = True,
        state_to_goal_extractor = True,
        embedding_dim = 20,
        max_desired_goals = 50,
        dim_per_desired_goal = 2,
        achieved_dim = 2,
        obs_dim = 4,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super(PoolingMultiInputPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch_head,
            net_arch_goal_extractor,
            state_to_head,
            state_to_goal_extractor,
            embedding_dim,
            max_desired_goals,
            dim_per_desired_goal,
            achieved_dim,
            obs_dim,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )


register_policy("PoolingMlpPolicy", PoolingMlpPolicy)
register_policy("PoolingCnnPolicy", PoolingCnnPolicy)
register_policy("PoolingMultiInputPolicy", PoolingMultiInputPolicy)

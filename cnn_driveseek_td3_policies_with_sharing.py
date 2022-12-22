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

class DriveSeekContinuousCritic(BaseModel):
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
        features_extractor: nn.Module,
        features_dim: int,
        state_to_head = True,
        state_to_goal_extractor = True,
        embedding_dim = 20,
        max_desired_goals = 50,
        dim_per_desired_goal = 2,
        achieved_dim = 2,
        obs_dim = 4,
        resolution = 8,
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
        self.max_desired_goals = max_desired_goals
        self.dim_per_desired_goal = dim_per_desired_goal
        self.achieved_dim = achieved_dim
        self.obs_dim= obs_dim 
        self.action_dim = get_action_dim(self.action_space)
        self.resolution = resolution
        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.embedding_q_networks = []
        self.head_q_networks = []
        state_action_size = self.achieved_dim + self.obs_dim + self.action_dim
        state_size = self.achieved_dim + self.obs_dim

        for idx in range(n_critics):

            embedding_q_net = nn.Sequential(
                nn.Conv2d(3 if state_to_goal_extractor else 1, 32, kernel_size=8, stride=4, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),
                nn.LazyLinear(embedding_dim)
            )
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

        canvas = th.zeros(features.shape[0], 21,21,device='cuda')

        canvas[th.arange(features.shape[0]).repeat_interleave(self.max_desired_goals),pure_goals[:,:,0].reshape(-1).long() + 10, pure_goals[:,:,1].reshape(-1).long() + 10] = goal_indicators.reshape(-1)**2

        canvas = canvas.repeat_interleave(4,dim=1).repeat_interleave(4,dim=2)

        if (self.state_to_goal_extractor):
            canvas[:,::4,:] = 0
            canvas[:,:,::4] = 0
            canvas[:,3::4,:] = 0
            canvas[:,:,3::4] = 0
        canvas = canvas.repeat_interleave(self.resolution//4,dim=1).repeat_interleave(self.resolution//4,dim=2)
        canvas = canvas.unsqueeze(1)
        if (self.state_to_goal_extractor):
            frame = th.zeros((features.shape[0],int(21.5*self.resolution),int(21.5*self.resolution), ) ,device='cuda')
            prop_frame = th.zeros((features.shape[0],int(21.5*self.resolution),int(21.5*self.resolution), ) ,device='cuda')
            starting_x = ((non_goal[:,-4] + 10.5 ) *self.resolution ).floor().long()
            starting_y = ((non_goal[:,-3] + 10.5 ) *self.resolution ).floor().long()
            propagated_postion_x = (((non_goal[:,-4] + non_goal[:,-2] + 10.5)%21 )  *self.resolution ).floor().long()
            propagated_postion_y = (((non_goal[:,-3] + non_goal[:,-1] + 10.5)%21 ) *self.resolution ).floor().long()
            resolution = self.resolution
            for i in range(self.resolution//2):
                for j in range(self.resolution//2):
                    frame[th.arange(features.shape[0]),starting_x+i,starting_y+j] = 1
                    prop_frame[th.arange(features.shape[0]),propagated_postion_x+i,propagated_postion_y+j] = 1
            frame[:,resolution//4:resolution//2,:] = frame[:,resolution//4:resolution//2,:].int() | frame[:,-resolution//4:,:].int()
            frame[:,-resolution//2:-resolution//4,:] = frame[:,-resolution//2:-resolution//4,:].int()  | frame[:,:resolution//4,:].int() 
            frame[:,:,resolution//4:resolution//2] = frame[:,:,resolution//4:resolution//2].int()  | frame[:,:,-resolution//4:].int() 
            frame[:,:,-resolution//2:-resolution//4] = frame[:,:,-resolution//2:-resolution//4].int()  | frame[:,:,:resolution//4].int() 
            prop_frame[:,resolution//4:resolution//2,:] = prop_frame[:,resolution//4:resolution//2,:].int()  | prop_frame[:,-resolution//4:,:].int() 
            prop_frame[:,-resolution//2:-resolution//4,:] = prop_frame[:,-resolution//2:-resolution//4,:].int()  | prop_frame[:,:resolution//4,:].int() 
            prop_frame[:,:,resolution//4:resolution//2] = prop_frame[:,:,resolution//4:resolution//2].int()  | prop_frame[:,:,-resolution//4:].int() 
            prop_frame[:,:,-resolution//2:-resolution//4] = prop_frame[:,:,-resolution//2:-resolution//4].int()  | prop_frame[:,:,:resolution//4].int() 


            canvas = th.cat([canvas,frame[:,None,resolution//4:-resolution//4,resolution//4:-resolution//4],prop_frame[:,None,resolution//4:-resolution//4,resolution//4:-resolution//4]],dim = 1)
        #print(goals)
        #print(canvas[0,:,:16,:16])
        outputs = []
        for idx in range(self.n_critics):
            intermediate = self.embedding_q_networks[idx](canvas).reshape((features.shape[0], -1))
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
        canvas = th.zeros(features.shape[0], 21,21,device='cuda')

        canvas[th.arange(features.shape[0]).repeat_interleave(self.max_desired_goals),pure_goals[:,:,0].reshape(-1).long() + 10, pure_goals[:,:,1].reshape(-1).long() + 10] = goal_indicators.reshape(-1)**2

        canvas = canvas.repeat_interleave(4,dim=1).repeat_interleave(4,dim=2)

        if (self.state_to_goal_extractor):
            canvas[:,::4,:] = 0
            canvas[:,:,::4] = 0
            canvas[:,3::4,:] = 0
            canvas[:,:,3::4] = 0
        canvas = canvas.repeat_interleave(self.resolution//4,dim=1).repeat_interleave(self.resolution//4,dim=2)
        canvas = canvas.unsqueeze(1)
        if (self.state_to_goal_extractor):
            frame = th.zeros((features.shape[0],int(21.5*self.resolution),int(21.5*self.resolution), ) ,device='cuda')
            prop_frame = th.zeros((features.shape[0],int(21.5*self.resolution),int(21.5*self.resolution), ) ,device='cuda')
            starting_x = ((non_goal[:,-4] + 10.5 ) *self.resolution ).floor().long()
            starting_y = ((non_goal[:,-3] + 10.5 ) *self.resolution ).floor().long()
            propagated_postion_x = (((non_goal[:,-4] + non_goal[:,-2] + 10.5)%21 )  *self.resolution ).floor().long()
            propagated_postion_y = (((non_goal[:,-3] + non_goal[:,-1] + 10.5)%21 ) *self.resolution ).floor().long()
            resolution = self.resolution
            for i in range(self.resolution//2):
                for j in range(self.resolution//2):
                    frame[th.arange(features.shape[0]),starting_x+i,starting_y+j] = 1
                    prop_frame[th.arange(features.shape[0]),propagated_postion_x+i,propagated_postion_y+j] = 1
            frame[:,resolution//4:resolution//2,:] = frame[:,resolution//4:resolution//2,:].int() | frame[:,-resolution//4:,:].int()
            frame[:,-resolution//2:-resolution//4,:] = frame[:,-resolution//2:-resolution//4,:].int()  | frame[:,:resolution//4,:].int() 
            frame[:,:,resolution//4:resolution//2] = frame[:,:,resolution//4:resolution//2].int()  | frame[:,:,-resolution//4:].int() 
            frame[:,:,-resolution//2:-resolution//4] = frame[:,:,-resolution//2:-resolution//4].int()  | frame[:,:,:resolution//4].int() 
            prop_frame[:,resolution//4:resolution//2,:] = prop_frame[:,resolution//4:resolution//2,:].int()  | prop_frame[:,-resolution//4:,:].int() 
            prop_frame[:,-resolution//2:-resolution//4,:] = prop_frame[:,-resolution//2:-resolution//4,:].int()  | prop_frame[:,:resolution//4,:].int() 
            prop_frame[:,:,resolution//4:resolution//2] = prop_frame[:,:,resolution//4:resolution//2].int()  | prop_frame[:,:,-resolution//4:].int() 
            prop_frame[:,:,-resolution//2:-resolution//4] = prop_frame[:,:,-resolution//2:-resolution//4].int()  | prop_frame[:,:,:resolution//4].int() 

            canvas = th.cat([canvas,frame[:,None,resolution//4:-resolution//4,resolution//4:-resolution//4],prop_frame[:,None,resolution//4:-resolution//4,resolution//4:-resolution//4]],dim = 1)
 
        intermediate = self.embedding_q_networks[0](canvas).reshape((features.shape[0], -1))
        if (self.state_to_head):
            intermediate = th.cat([intermediate,non_goal,actions], dim= -1)
        return self.head_q_networks[0](intermediate)



class DriveSeekActor(BasePolicy):
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
        resolution = 8,
        normalize_images: bool = True,
    ):
        super(DriveSeekActor, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        self.net_arch_head = net_arch_head
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.resolution = resolution

        self.state_to_head = state_to_head
        self.state_to_goal_extractor = state_to_goal_extractor
        self.embedding_dim = embedding_dim
        self.max_desired_goals = max_desired_goals
        self.dim_per_desired_goal = dim_per_desired_goal
        self.achieved_dim = achieved_dim
        self.obs_dim= obs_dim 
        self.action_dim = get_action_dim(self.action_space)
        state_size = self.achieved_dim + self.obs_dim

        self.embedding_mu = []
        head_actor_net = create_mlp( self.embedding_dim + (state_size if self.state_to_head else 0), self.action_dim, net_arch_head, activation_fn, squash_output=True)
        self.head_mu = nn.Sequential(*head_actor_net)
    def set_embedder(self,embedder):
        self.embedding_mu.append(embedder)
    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch_head = self.net_arch_head,
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
        with th.set_grad_enabled(False):
            non_goal = th.cat([features[:,:self.achieved_dim],features[:,-self.obs_dim:]], dim = 1) 
            goals = features[:,self.achieved_dim:-self.obs_dim].reshape((features.shape[0], self.max_desired_goals,self.dim_per_desired_goal + 1 ))
            pure_goals = goals[:,:,1:]
            goal_indicators = goals[:,:,0]
            canvas = th.zeros(features.shape[0], 21,21,device='cuda')

            canvas[th.arange(features.shape[0]).repeat_interleave(self.max_desired_goals),pure_goals[:,:,0].reshape(-1).long() + 10, pure_goals[:,:,1].reshape(-1).long() + 10] = goal_indicators.reshape(-1)**2

            canvas = canvas.repeat_interleave(4,dim=1).repeat_interleave(4,dim=2)

            if (self.state_to_goal_extractor):
                canvas[:,::4,:] = 0
                canvas[:,:,::4] = 0
                canvas[:,3::4,:] = 0
                canvas[:,:,3::4] = 0
            canvas = canvas.repeat_interleave(self.resolution//4,dim=1).repeat_interleave(self.resolution//4,dim=2)
            canvas = canvas.unsqueeze(1)
            resolution = self.resolution
            if (self.state_to_goal_extractor):
                frame = th.zeros((features.shape[0],int(21.5*self.resolution),int(21.5*self.resolution), ) ,device='cuda')
                prop_frame = th.zeros((features.shape[0],int(21.5*self.resolution),int(21.5*self.resolution), ) ,device='cuda')
                starting_x = ((non_goal[:,-4] + 10.5 ) *self.resolution ).floor().long()
                starting_y = ((non_goal[:,-3] + 10.5 ) *self.resolution ).floor().long()
                propagated_postion_x = (((non_goal[:,-4] + non_goal[:,-2] + 10.5)%21 )  *self.resolution ).floor().long()
                propagated_postion_y = (((non_goal[:,-3] + non_goal[:,-1] + 10.5)%21 ) *self.resolution ).floor().long()
                for i in range(self.resolution//2):
                    for j in range(self.resolution//2):
                        frame[th.arange(features.shape[0]),starting_x+i,starting_y+j] = 1
                        prop_frame[th.arange(features.shape[0]),propagated_postion_x+i,propagated_postion_y+j] = 1
                frame[:,resolution//4:resolution//2,:] = frame[:,resolution//4:resolution//2,:].int() | frame[:,-resolution//4:,:].int()
                frame[:,-resolution//2:-resolution//4,:] = frame[:,-resolution//2:-resolution//4,:].int()  | frame[:,:resolution//4,:].int() 
                frame[:,:,resolution//4:resolution//2] = frame[:,:,resolution//4:resolution//2].int()  | frame[:,:,-resolution//4:].int() 
                frame[:,:,-resolution//2:-resolution//4] = frame[:,:,-resolution//2:-resolution//4].int()  | frame[:,:,:resolution//4].int() 
                prop_frame[:,resolution//4:resolution//2,:] = prop_frame[:,resolution//4:resolution//2,:].int()  | prop_frame[:,-resolution//4:,:].int() 
                prop_frame[:,-resolution//2:-resolution//4,:] = prop_frame[:,-resolution//2:-resolution//4,:].int()  | prop_frame[:,:resolution//4,:].int() 
                prop_frame[:,:,resolution//4:resolution//2] = prop_frame[:,:,resolution//4:resolution//2].int()  | prop_frame[:,:,-resolution//4:].int() 
                prop_frame[:,:,-resolution//2:-resolution//4] = prop_frame[:,:,-resolution//2:-resolution//4].int()  | prop_frame[:,:,:resolution//4].int() 

                canvas = th.cat([canvas,frame[:,None,resolution//4:-resolution//4,resolution//4:-resolution//4],prop_frame[:,None,resolution//4:-resolution//4,resolution//4:-resolution//4]],dim = 1)
            #print("here!")
            #print(canvas.shape)
            #print(canvas.min())
            #print(canvas.max())
            intermediate = self.embedding_mu[0](canvas).reshape((features.shape[0], -1))
            if (self.state_to_head):
                intermediate = th.cat([intermediate,non_goal], dim= -1)
        return self.head_mu(intermediate)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Note: the deterministic deterministic parameter is ignored in the case of TD3.
        #   Predictions are always deterministic.
        return self.forward(observation)


class DriveSeekTD3Policy(TD3Policy):
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
        state_to_head = True,
        state_to_goal_extractor = True,
        embedding_dim = 20,
        max_desired_goals = 50,
        dim_per_desired_goal = 2,
        achieved_dim = 2,
        obs_dim = 4,
        resolution=8,
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
        # Default network architecture, from the original paper
        # if net_arch is None:
        #     if features_extractor_class == NatureCNN:
        #         net_arch = []
        #     else:
        #         net_arch = [400, 300]

        actor_arch_head, critic_arch_head = get_actor_critic_arch(net_arch_head)

        self.net_arch_head = net_arch_head
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
            "net_arch_head" : actor_arch_head,
            "state_to_head" : state_to_head,
            "state_to_goal_extractor" : state_to_goal_extractor,
            "embedding_dim" : embedding_dim,
            "max_desired_goals" : max_desired_goals,
            "dim_per_desired_goal" : dim_per_desired_goal,
            "achieved_dim" :achieved_dim,
            "obs_dim" :obs_dim,
            "resolution": resolution
        }
        self.actor_kwargs = self.net_args.copy()
        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "net_arch_head" : critic_arch_head,
                "n_critics": n_critics,
                "share_features_extractor": share_features_extractor,
            }
        )

        self.actor, self.actor_target = None, None
        self.critic, self.critic_target = None, None
        self.share_features_extractor = share_features_extractor

        self._build(lr_schedule)
        self.actor.set_embedder(self.critic.embedding_q_networks[0])
        self.actor_target.set_embedder(self.critic_target.embedding_q_networks[0])
    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = BasePolicy._get_constructor_parameters(self)

        data.update(
            dict(
                net_arch_head=self.net_args["net_arch_head"],
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

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> DriveSeekActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return DriveSeekActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> DriveSeekContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return DriveSeekContinuousCritic(**critic_kwargs).to(self.device)





DriveSeekMlpPolicy = DriveSeekTD3Policy



class DriveSeekMultiInputPolicy(DriveSeekTD3Policy):
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
        state_to_head = True,
        state_to_goal_extractor = True,
        embedding_dim = 20,
        max_desired_goals = 50,
        dim_per_desired_goal = 2,
        achieved_dim = 2,
        obs_dim = 4,
        resolution = 8,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super(DriveSeekMultiInputPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch_head,
            state_to_head,
            state_to_goal_extractor,
            embedding_dim,
            max_desired_goals,
            dim_per_desired_goal,
            achieved_dim,
            obs_dim,
            resolution,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )


register_policy("DriveSeekMlpPolicy", DriveSeekMlpPolicy)
register_policy("DriveSeekMultiInputPolicy", DriveSeekMultiInputPolicy)

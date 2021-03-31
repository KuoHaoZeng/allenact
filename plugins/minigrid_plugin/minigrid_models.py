import abc
from typing import Callable, Dict, Optional, Tuple, cast

import gym
import numpy as np
import torch
from gym.spaces.dict import Dict as SpaceDict
from torch import nn

from core.algorithms.onpolicy_sync.policy import (
    ActorCriticModel,
    Memory,
    DistributionType,
    ActorCriticOutput,
    ObservationType,
)
from core.base_abstractions.distributions import CategoricalDistr
from core.models.basic_models import LinearActorCritic, RNNActorCritic, RNNStateEncoder
from utils.misc_utils import prepare_locals_for_super


class MiniGridSimpleConvBase(ActorCriticModel[CategoricalDistr], abc.ABC):
    actor_critic: ActorCriticModel

    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        num_objects: int,
        num_colors: int,
        num_states: int,
        object_embedding_dim: int = 8,
        **kwargs,
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)

        self.num_objects = num_objects
        self.object_embedding_dim = object_embedding_dim

        vis_input_shape = observation_space["minigrid_ego_image"].shape
        agent_view_x, agent_view_y, view_channels = vis_input_shape
        assert agent_view_x == agent_view_y
        self.agent_view = agent_view_x
        self.view_channels = view_channels

        assert (np.array(vis_input_shape[:2]) >= 3).all(), (
            "MiniGridSimpleConvRNN requires" "that the input size be at least 3x3."
        )

        self.num_channels = 0

        if self.num_objects > 0:
            # Object embedding
            self.object_embedding = nn.Embedding(
                num_embeddings=num_objects, embedding_dim=self.object_embedding_dim
            )
            self.object_channel = self.num_channels
            self.num_channels += 1

        self.num_colors = num_colors
        if self.num_colors > 0:
            # Same dimensionality used for colors and states
            self.color_embedding = nn.Embedding(
                num_embeddings=num_colors, embedding_dim=self.object_embedding_dim
            )
            self.color_channel = self.num_channels
            self.num_channels += 1

        self.num_states = num_states
        if self.num_states > 0:
            self.state_embedding = nn.Embedding(
                num_embeddings=num_states, embedding_dim=self.object_embedding_dim
            )
            self.state_channel = self.num_channels
            self.num_channels += 1

        assert self.num_channels == self.view_channels > 0

        self.ac_key = "enc"
        self.observations_for_ac: Dict[str, Optional[torch.Tensor]] = {
            self.ac_key: None
        }

        self.num_agents = 1

    def forward(  # type:ignore
        self,
        observations: ObservationType,
        memory: Memory,
        prev_actions: torch.Tensor,
        masks: torch.FloatTensor,
        current_actions: torch.FloatTensor = None,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        minigrid_ego_image = cast(torch.Tensor, observations["minigrid_ego_image"])
        use_agent = minigrid_ego_image.shape == 6
        nrow, ncol, nchannels = minigrid_ego_image.shape[-3:]
        nsteps, nsamplers, nagents = masks.shape[:3]

        assert nrow == ncol == self.agent_view
        assert nchannels == self.view_channels == self.num_channels

        embed_list = []
        if self.num_objects > 0:
            ego_object_embeds = self.object_embedding(
                minigrid_ego_image[..., self.object_channel].long()
            )
            embed_list.append(ego_object_embeds)
        if self.num_colors > 0:
            ego_color_embeds = self.color_embedding(
                minigrid_ego_image[..., self.color_channel].long()
            )
            embed_list.append(ego_color_embeds)
        if self.num_states > 0:
            ego_state_embeds = self.state_embedding(
                minigrid_ego_image[..., self.state_channel].long()
            )
            embed_list.append(ego_state_embeds)
        ego_embeds = torch.cat(embed_list, dim=-1)

        if use_agent:
            self.observations_for_ac[self.ac_key] = ego_embeds.view(
                nsteps, nsamplers, nagents, -1
            )
        else:
            self.observations_for_ac[self.ac_key] = ego_embeds.view(
                nsteps, nsamplers * nagents, -1
            )

        # noinspection PyCallingNonCallable
        out, mem_return = self.actor_critic(
            observations=self.observations_for_ac,
            memory=memory,
            prev_actions=prev_actions,
            masks=masks,
        )

        self.observations_for_ac[self.ac_key] = None

        return out, mem_return


class MiniGridSimpleConvRNN(MiniGridSimpleConvBase):
    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        num_objects: int,
        num_colors: int,
        num_states: int,
        object_embedding_dim: int = 8,
        hidden_size=512,
        num_layers=1,
        rnn_type="GRU",
        head_type: Callable[
            ..., ActorCriticModel[CategoricalDistr]
        ] = LinearActorCritic,
        **kwargs,
    ):
        super().__init__(**prepare_locals_for_super(locals()))

        self._hidden_size = hidden_size
        agent_view_x, agent_view_y, view_channels = observation_space[
            "minigrid_ego_image"
        ].shape
        self.actor_critic = RNNActorCritic(
            input_uuid=self.ac_key,
            action_space=action_space,
            observation_space=SpaceDict(
                {
                    self.ac_key: gym.spaces.Box(
                        low=np.float32(-1.0),
                        high=np.float32(1.0),
                        shape=(
                            self.object_embedding_dim
                            * agent_view_x
                            * agent_view_y
                            * view_channels,
                        ),
                    )
                }
            ),
            hidden_size=hidden_size,
            num_layers=num_layers,
            rnn_type=rnn_type,
            head_type=head_type,
        )
        self.memory_key = "rnn"

        self.train()

    @property
    def num_recurrent_layers(self):
        return self.actor_critic.num_recurrent_layers

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size

    def _recurrent_memory_specification(self):
        return {
            self.memory_key: (
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            )
        }


class MiniGridMASimpleConvRNN(MiniGridSimpleConvBase):
    def __init__(
            self,
            action_space: gym.spaces.Discrete,
            observation_space: SpaceDict,
            num_objects: int,
            num_colors: int,
            num_states: int,
            object_embedding_dim: int = 8,
            missing_action_embedding_dim: int = 32,
            missing_action_uuid: str = "missing_action",
            hidden_size=512,
            num_layers=1,
            rnn_type="GRU",
            head_type: Callable[
                ..., ActorCriticModel[CategoricalDistr]
            ] = LinearActorCritic,
            **kwargs,
    ):
        super().__init__(**prepare_locals_for_super(locals()))

        # Missing Action embedding
        self.missing_action_uuid = missing_action_uuid
        self.missing_action_embedding = nn.Embedding(
            num_embeddings=action_space.n, embedding_dim=missing_action_embedding_dim
        )

        self._hidden_size = hidden_size
        agent_view_x, agent_view_y, view_channels = observation_space[
            "minigrid_ego_image"
        ].shape
        self.actor_critic = RNNActorCritic(
            input_uuid=self.ac_key,
            action_space=action_space,
            observation_space=SpaceDict(
                {
                    self.ac_key: gym.spaces.Box(
                        low=np.float32(-1.0),
                        high=np.float32(1.0),
                        shape=(
                            self.object_embedding_dim
                            * agent_view_x
                            * agent_view_y
                            * view_channels + missing_action_embedding_dim,
                        ),
                    )
                }
            ),
            hidden_size=hidden_size,
            num_layers=num_layers,
            rnn_type=rnn_type,
            head_type=head_type,
        )
        self.memory_key = "rnn"

        self.train()

    def get_missing_action_embedding(self, observations):
        return self.missing_action_embedding(observations[self.missing_action_uuid].to(torch.int64))

    @property
    def num_recurrent_layers(self):
        return self.actor_critic.num_recurrent_layers

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size

    def _recurrent_memory_specification(self):
        return {
            self.memory_key: (
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            )
        }

    def forward(  # type:ignore
            self,
            observations: ObservationType,
            memory: Memory,
            prev_actions: torch.Tensor,
            masks: torch.FloatTensor,
            current_actions: torch.FloatTensor = None,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        minigrid_ego_image = cast(torch.Tensor, observations["minigrid_ego_image"])
        use_agent = minigrid_ego_image.shape == 6
        nrow, ncol, nchannels = minigrid_ego_image.shape[-3:]
        nsteps, nsamplers, nagents = masks.shape[:3]

        assert nrow == ncol == self.agent_view
        assert nchannels == self.view_channels == self.num_channels

        embed_list = []
        if self.num_objects > 0:
            ego_object_embeds = self.object_embedding(
                minigrid_ego_image[..., self.object_channel].long()
            )
            embed_list.append(ego_object_embeds)
        if self.num_colors > 0:
            ego_color_embeds = self.color_embedding(
                minigrid_ego_image[..., self.color_channel].long()
            )
            embed_list.append(ego_color_embeds)
        if self.num_states > 0:
            ego_state_embeds = self.state_embedding(
                minigrid_ego_image[..., self.state_channel].long()
            )
            embed_list.append(ego_state_embeds)
        ego_embeds = torch.cat(embed_list, dim=-1)

        if use_agent:
            self.observations_for_ac[self.ac_key] = ego_embeds.view(
                nsteps, nsamplers, nagents, -1
            )
        else:
            self.observations_for_ac[self.ac_key] = ego_embeds.view(
                nsteps, nsamplers * nagents, -1
            )

        self.observations_for_ac[self.ac_key] = torch.cat([self.observations_for_ac[self.ac_key],
                                                           self.get_missing_action_embedding(observations)], dim=-1)

        # noinspection PyCallingNonCallable
        out, mem_return = self.actor_critic(
            observations=self.observations_for_ac,
            memory=memory,
            prev_actions=prev_actions,
            masks=masks,
        )

        self.observations_for_ac[self.ac_key] = None

        return out, mem_return


class MiniGridMAInternalSimpleConvRNN(MiniGridSimpleConvBase):
    def __init__(
            self,
            action_space: gym.spaces.Discrete,
            observation_space: SpaceDict,
            num_objects: int,
            num_colors: int,
            num_states: int,
            object_embedding_dim: int = 8,
            prev_action_embedding_dim: int = 32,
            hidden_size=512,
            num_layers=1,
            rnn_type="GRU",
            head_type: Callable[
                ..., ActorCriticModel[CategoricalDistr]
            ] = LinearActorCritic,
            **kwargs,
    ):
        super().__init__(**prepare_locals_for_super(locals()))

        # Missing Action embedding
        self.prev_action_embedding_dim = prev_action_embedding_dim
        self.prev_action_embedding = nn.Embedding(
            num_embeddings=action_space.n, embedding_dim=prev_action_embedding_dim
        )
        self.intenral_model_classifier = nn.Linear(prev_action_embedding_dim, action_space.n)

        self._hidden_size = hidden_size
        agent_view_x, agent_view_y, view_channels = observation_space[
            "minigrid_ego_image"
        ].shape

        self.internal_model = RNNStateEncoder(
            self.object_embedding_dim
            * agent_view_x
            * agent_view_y
            * view_channels + prev_action_embedding_dim,
            prev_action_embedding_dim,
            num_layers=1,
            rnn_type=rnn_type,
        )

        self.actor_critic = RNNActorCritic(
            input_uuid=self.ac_key,
            action_space=action_space,
            observation_space=SpaceDict(
                {
                    self.ac_key: gym.spaces.Box(
                        low=np.float32(-1.0),
                        high=np.float32(1.0),
                        shape=(
                            self.object_embedding_dim
                            * agent_view_x
                            * agent_view_y
                            * view_channels + prev_action_embedding_dim,
                        ),
                    )
                }
            ),
            hidden_size=hidden_size,
            num_layers=num_layers,
            rnn_type=rnn_type,
            head_type=head_type,
        )
        self.memory_key = "rnn"

        self.train()

    @property
    def num_recurrent_layers(self):
        return self.actor_critic.num_recurrent_layers

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size

    def _recurrent_memory_specification(self):
        return {
            self.memory_key: (
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            ),
            "internal": (
                 (
                     ("layer", self.num_recurrent_layers),
                     ("sampler", None),
                     ("hidden", self.prev_action_embedding_dim),
                 ),
                 torch.float32,
            ),
        }

    def forward(  # type:ignore
            self,
            observations: ObservationType,
            memory: Memory,
            prev_actions: torch.Tensor,
            masks: torch.FloatTensor,
            current_actions: torch.FloatTensor = None,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        minigrid_ego_image = cast(torch.Tensor, observations["minigrid_ego_image"])
        prev_minigrid_ego_image = cast(torch.Tensor, observations["prev_minigrid_ego_image"])
        use_agent = minigrid_ego_image.shape == 6
        nrow, ncol, nchannels = minigrid_ego_image.shape[-3:]
        nsteps, nsamplers, nagents = masks.shape[:3]

        assert nrow == ncol == self.agent_view
        assert nchannels == self.view_channels == self.num_channels

        prev_embed_list = []
        if self.num_objects > 0:
            ego_object_embeds = self.object_embedding(
                prev_minigrid_ego_image[..., self.object_channel].long()
            )
            prev_embed_list.append(ego_object_embeds)
        if self.num_colors > 0:
            ego_color_embeds = self.color_embedding(
                prev_minigrid_ego_image[..., self.color_channel].long()
            )
            prev_embed_list.append(ego_color_embeds)
        if self.num_states > 0:
            ego_state_embeds = self.state_embedding(
                prev_minigrid_ego_image[..., self.state_channel].long()
            )
            prev_embed_list.append(ego_state_embeds)
        prev_ego_embeds = torch.cat(prev_embed_list, dim=-1)
        prev_action_embeds = self.prev_action_embedding(prev_actions)

        embed_list = []
        if self.num_objects > 0:
            ego_object_embeds = self.object_embedding(
                minigrid_ego_image[..., self.object_channel].long()
            )
            embed_list.append(ego_object_embeds)
        if self.num_colors > 0:
            ego_color_embeds = self.color_embedding(
                minigrid_ego_image[..., self.color_channel].long()
            )
            embed_list.append(ego_color_embeds)
        if self.num_states > 0:
            ego_state_embeds = self.state_embedding(
                minigrid_ego_image[..., self.state_channel].long()
            )
            embed_list.append(ego_state_embeds)
        ego_embeds = torch.cat(embed_list, dim=-1)

        ego_embeds_different = ego_embeds - prev_ego_embeds

        if use_agent:
            self.observations_for_ac[self.ac_key] = ego_embeds.view(
                nsteps, nsamplers, nagents, -1
            )
            ego_embeds_different = ego_embeds_different.view(nsteps, nsamplers, nagents, -1)
            prev_action_embeds = prev_action_embeds.view(nsteps, nsamplers, nagents, -1)
        else:
            self.observations_for_ac[self.ac_key] = ego_embeds.view(
                nsteps, nsamplers * nagents, -1
            )
            ego_embeds_different = ego_embeds_different.view(nsteps, nsamplers * nagents, -1)
            prev_action_embeds = prev_action_embeds.view(nsteps, nsamplers * nagents, -1)

        internal_feats = torch.cat([ego_embeds_different, prev_action_embeds], -1)
        internal_feats, internal_model_states = self.internal_model(internal_feats,
                                                                    memory.tensor("internal"),
                                                                    masks)
        internal_output = self.intenral_model_classifier(internal_feats)

        self.observations_for_ac[self.ac_key] = torch.cat([self.observations_for_ac[self.ac_key],
                                                           internal_feats], dim=-1)

        # noinspection PyCallingNonCallable
        out, mem_return = self.actor_critic(
            observations=self.observations_for_ac,
            memory=memory,
            prev_actions=prev_actions,
            masks=masks,
        )

        if not isinstance(current_actions, type(None)):
            out = {"ac_output": out, "internal_output": internal_output, "internal_prev_actions": prev_actions}

        mem_return = mem_return.set_tensor("internal", internal_model_states)

        self.observations_for_ac[self.ac_key] = None

        return out, mem_return


class MiniGridMAInternalAESimpleConvRNN(MiniGridSimpleConvBase):
    def __init__(
            self,
            action_space: gym.spaces.Discrete,
            observation_space: SpaceDict,
            num_objects: int,
            num_colors: int,
            num_states: int,
            object_embedding_dim: int = 8,
            prev_action_embedding_dim: int = 32,
            hidden_size=512,
            num_layers=1,
            rnn_type="GRU",
            head_type: Callable[
                ..., ActorCriticModel[CategoricalDistr]
            ] = LinearActorCritic,
            **kwargs,
    ):
        super().__init__(**prepare_locals_for_super(locals()))

        # Missing Action embedding
        self.prev_action_embedding_dim = prev_action_embedding_dim
        self.prev_action_embedding = nn.Embedding(
            num_embeddings=action_space.n, embedding_dim=prev_action_embedding_dim
        )
        self.intenral_model_classifier = nn.Linear(prev_action_embedding_dim, action_space.n)
        self.intenral_model_decoder = nn.Linear(action_space.n, prev_action_embedding_dim)

        self._hidden_size = hidden_size
        agent_view_x, agent_view_y, view_channels = observation_space[
            "minigrid_ego_image"
        ].shape

        self.internal_model = RNNStateEncoder(
            self.object_embedding_dim
            * agent_view_x
            * agent_view_y
            * view_channels + prev_action_embedding_dim,
            prev_action_embedding_dim,
            num_layers=1,
            rnn_type=rnn_type,
        )

        self.actor_critic = RNNActorCritic(
            input_uuid=self.ac_key,
            action_space=action_space,
            observation_space=SpaceDict(
                {
                    self.ac_key: gym.spaces.Box(
                        low=np.float32(-1.0),
                        high=np.float32(1.0),
                        shape=(
                            self.object_embedding_dim
                            * agent_view_x
                            * agent_view_y
                            * view_channels + prev_action_embedding_dim,
                        ),
                    )
                }
            ),
            hidden_size=hidden_size,
            num_layers=num_layers,
            rnn_type=rnn_type,
            head_type=head_type,
        )
        self.memory_key = "rnn"

        self.train()

    @property
    def num_recurrent_layers(self):
        return self.actor_critic.num_recurrent_layers

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size

    def _recurrent_memory_specification(self):
        return {
            self.memory_key: (
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            ),
            "internal": (
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.prev_action_embedding_dim),
                ),
                torch.float32,
            ),
        }

    def forward(  # type:ignore
            self,
            observations: ObservationType,
            memory: Memory,
            prev_actions: torch.Tensor,
            masks: torch.FloatTensor,
            current_actions: torch.FloatTensor = None,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        minigrid_ego_image = cast(torch.Tensor, observations["minigrid_ego_image"])
        prev_minigrid_ego_image = cast(torch.Tensor, observations["prev_minigrid_ego_image"])
        use_agent = minigrid_ego_image.shape == 6
        nrow, ncol, nchannels = minigrid_ego_image.shape[-3:]
        nsteps, nsamplers, nagents = masks.shape[:3]

        assert nrow == ncol == self.agent_view
        assert nchannels == self.view_channels == self.num_channels

        prev_embed_list = []
        if self.num_objects > 0:
            ego_object_embeds = self.object_embedding(
                prev_minigrid_ego_image[..., self.object_channel].long()
            )
            prev_embed_list.append(ego_object_embeds)
        if self.num_colors > 0:
            ego_color_embeds = self.color_embedding(
                prev_minigrid_ego_image[..., self.color_channel].long()
            )
            prev_embed_list.append(ego_color_embeds)
        if self.num_states > 0:
            ego_state_embeds = self.state_embedding(
                prev_minigrid_ego_image[..., self.state_channel].long()
            )
            prev_embed_list.append(ego_state_embeds)
        prev_ego_embeds = torch.cat(prev_embed_list, dim=-1)
        prev_action_embeds = self.prev_action_embedding(prev_actions)

        embed_list = []
        if self.num_objects > 0:
            ego_object_embeds = self.object_embedding(
                minigrid_ego_image[..., self.object_channel].long()
            )
            embed_list.append(ego_object_embeds)
        if self.num_colors > 0:
            ego_color_embeds = self.color_embedding(
                minigrid_ego_image[..., self.color_channel].long()
            )
            embed_list.append(ego_color_embeds)
        if self.num_states > 0:
            ego_state_embeds = self.state_embedding(
                minigrid_ego_image[..., self.state_channel].long()
            )
            embed_list.append(ego_state_embeds)
        ego_embeds = torch.cat(embed_list, dim=-1)

        ego_embeds_different = ego_embeds - prev_ego_embeds

        if use_agent:
            self.observations_for_ac[self.ac_key] = ego_embeds.view(
                nsteps, nsamplers, nagents, -1
            )
            ego_embeds_different = ego_embeds_different.view(nsteps, nsamplers, nagents, -1)
            prev_action_embeds = prev_action_embeds.view(nsteps, nsamplers, nagents, -1)
        else:
            self.observations_for_ac[self.ac_key] = ego_embeds.view(
                nsteps, nsamplers * nagents, -1
            )
            ego_embeds_different = ego_embeds_different.view(nsteps, nsamplers * nagents, -1)
            prev_action_embeds = prev_action_embeds.view(nsteps, nsamplers * nagents, -1)

        internal_feats = torch.cat([ego_embeds_different, prev_action_embeds], -1)
        internal_feats, internal_model_states = self.internal_model(internal_feats,
                                                                    memory.tensor("internal"),
                                                                    masks)
        internal_output = self.intenral_model_classifier(internal_feats)
        internal_feats = self.intenral_model_decoder(nn.functional.sigmoid(internal_output))

        self.observations_for_ac[self.ac_key] = torch.cat([self.observations_for_ac[self.ac_key],
                                                           internal_feats], dim=-1)

        # noinspection PyCallingNonCallable
        out, mem_return = self.actor_critic(
            observations=self.observations_for_ac,
            memory=memory,
            prev_actions=prev_actions,
            masks=masks,
        )

        if not isinstance(current_actions, type(None)):
            out = {"ac_output": out, "internal_output": internal_output, "internal_prev_actions": prev_actions}

        mem_return = mem_return.set_tensor("internal", internal_model_states)

        self.observations_for_ac[self.ac_key] = None

        return out, mem_return


class MiniGridSimpleConv(MiniGridSimpleConvBase):
    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        num_objects: int,
        num_colors: int,
        num_states: int,
        object_embedding_dim: int = 8,
        **kwargs,
    ):
        super().__init__(**prepare_locals_for_super(locals()))

        agent_view_x, agent_view_y, view_channels = observation_space[
            "minigrid_ego_image"
        ].shape
        self.actor_critic = LinearActorCritic(
            self.ac_key,
            action_space=action_space,
            observation_space=SpaceDict(
                {
                    self.ac_key: gym.spaces.Box(
                        low=np.float32(-1.0),
                        high=np.float32(1.0),
                        shape=(
                            self.object_embedding_dim
                            * agent_view_x
                            * agent_view_y
                            * view_channels,
                        ),
                    )
                }
            ),
        )
        self.memory_key = None

        self.train()

    @property
    def num_recurrent_layers(self):
        return 0

    @property
    def recurrent_hidden_state_size(self):
        return 0

    # noinspection PyMethodMayBeStatic
    def _recurrent_memory_specification(self):
        return None

import typing
from typing import Tuple, Dict, Optional, Union, List, cast, Sequence

import gym
import torch
import torch.nn as nn
from gym.spaces.dict import Dict as SpaceDict

from core.algorithms.onpolicy_sync.policy import (
    ActorCriticModel,
    LinearCriticHead,
    LinearActorHead,
    ObservationType,
    DistributionType,
)
from core.base_abstractions.distributions import CategoricalDistr
from core.base_abstractions.misc import ActorCriticOutput, Memory
from core.models.basic_models import SimpleCNN, RNNStateEncoder
from utils.model_utils import make_cnn, compute_cnn_output
from utils.utils_3d_torch import plot_next_keypoints, save_keypoints_results
import numpy as np

class RGBDSCNN(nn.Module):
    """A Simple N-Conv CNN followed by a fully connected layer. Takes in
    observations (of type gym.spaces.dict) and produces an embedding of the
    `rgb_uuid` and/or `depth_uuid` components.

    # Attributes

    observation_space : The observation_space of the agent, should have `rgb_uuid` or `depth_uuid` as
        a component (otherwise it is a blind model).
    output_size : The size of the embedding vector to produce.
    """

    def __init__(
            self,
            observation_space: SpaceDict,
            output_size: int,
            layer_channels: Sequence[int] = (32, 64, 32),
            kernel_sizes: Sequence[Tuple[int, int]] = ((8, 8), (4, 4), (3, 3)),
            layers_stride: Sequence[Tuple[int, int]] = ((4, 4), (2, 2), (1, 1)),
            paddings: Sequence[Tuple[int, int]] = ((0, 0), (0, 0), (0, 0)),
            dilations: Sequence[Tuple[int, int]] = ((1, 1), (1, 1), (1, 1)),
            rgb_uuid: str = "rgb",
            depth_uuid: str = "depth",
            seg_uuid: str = "seg",
            flatten: bool = True,
            output_relu: bool = True,
    ):
        """Initializer.

        # Parameters

        observation_space : See class attributes documentation.
        output_size : See class attributes documentation.
        """
        super().__init__()

        self.rgb_uuid = rgb_uuid
        if self.rgb_uuid in observation_space.spaces:
            self._n_input_rgb = observation_space.spaces[self.rgb_uuid].shape[2]
            assert self._n_input_rgb >= 0
        else:
            self._n_input_rgb = 0

        self.depth_uuid = depth_uuid
        if self.depth_uuid in observation_space.spaces:
            self._n_input_depth = observation_space.spaces[self.depth_uuid].shape[2]
            assert self._n_input_depth >= 0
        else:
            self._n_input_depth = 0

        self.seg_uuid = seg_uuid
        self._n_input_seg = 1

        if not self.is_blind:
            # hyperparameters for layers
            self._cnn_layers_channels = list(layer_channels)
            self._cnn_layers_kernel_size = list(kernel_sizes)
            self._cnn_layers_stride = list(layers_stride)
            self._cnn_layers_paddings = list(paddings)
            self._cnn_layers_dilations = list(dilations)

            if self._n_input_rgb > 0:
                input_rgb_cnn_dims = np.array(
                    observation_space.spaces[self.rgb_uuid].shape[:2], dtype=np.float32
                )
                self.rgb_cnn = self.make_cnn_from_params(
                    output_size=output_size,
                    input_dims=input_rgb_cnn_dims,
                    input_channels=self._n_input_rgb,
                    flatten=flatten,
                    output_relu=output_relu,
                )

            if self._n_input_depth > 0:
                input_depth_cnn_dims = np.array(
                    observation_space.spaces[self.depth_uuid].shape[:2],
                    dtype=np.float32,
                )
                self.depth_cnn = self.make_cnn_from_params(
                    output_size=output_size,
                    input_dims=input_depth_cnn_dims,
                    input_channels=self._n_input_depth,
                    flatten=flatten,
                    output_relu=output_relu,
                )

            if self._n_input_seg > 0:
                input_seg_cnn_dims = np.array(
                    observation_space.spaces[self.depth_uuid].shape[:2],
                    dtype=np.float32,
                )
                self.seg_cnn = self.make_cnn_from_params(
                    output_size=output_size,
                    input_dims=input_seg_cnn_dims,
                    input_channels=self._n_input_depth,
                    flatten=flatten,
                    output_relu=output_relu,
                )

    def make_cnn_from_params(
            self,
            output_size: int,
            input_dims: np.ndarray,
            input_channels: int,
            flatten: bool,
            output_relu: bool,
    ) -> nn.Module:
        output_dims = input_dims
        for kernel_size, stride, padding, dilation in zip(
                self._cnn_layers_kernel_size,
                self._cnn_layers_stride,
                self._cnn_layers_paddings,
                self._cnn_layers_dilations,
        ):
            # noinspection PyUnboundLocalVariable
            output_dims = self._conv_output_dim(
                dimension=output_dims,
                padding=np.array(padding, dtype=np.float32),
                dilation=np.array(dilation, dtype=np.float32),
                kernel_size=np.array(kernel_size, dtype=np.float32),
                stride=np.array(stride, dtype=np.float32),
            )

        # noinspection PyUnboundLocalVariable
        cnn = make_cnn(
            input_channels=input_channels,
            layer_channels=self._cnn_layers_channels,
            kernel_sizes=self._cnn_layers_kernel_size,
            strides=self._cnn_layers_stride,
            paddings=self._cnn_layers_paddings,
            dilations=self._cnn_layers_dilations,
            output_height=output_dims[0],
            output_width=output_dims[1],
            output_channels=output_size,
            flatten=flatten,
            output_relu=output_relu,
        )
        self.layer_init(cnn)

        return cnn

    @staticmethod
    def _conv_output_dim(
            dimension: Sequence[int],
            padding: Sequence[int],
            dilation: Sequence[int],
            kernel_size: Sequence[int],
            stride: Sequence[int],
    ) -> Tuple[int, ...]:
        """Calculates the output height and width based on the input height and
        width to the convolution layer. For parameter definitions see.

        [here](https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d).

        # Parameters

        dimension : See above link.
        padding : See above link.
        dilation : See above link.
        kernel_size : See above link.
        stride : See above link.
        """
        assert len(dimension) == 2
        out_dimension = []
        for i in range(len(dimension)):
            out_dimension.append(
                int(
                    np.floor(
                        (
                                (
                                        dimension[i]
                                        + 2 * padding[i]
                                        - dilation[i] * (kernel_size[i] - 1)
                                        - 1
                                )
                                / stride[i]
                        )
                        + 1
                    )
                )
            )
        return tuple(out_dimension)

    @staticmethod
    def layer_init(cnn) -> None:
        """Initialize layer parameters using Kaiming normal."""
        for layer in cnn:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(layer.weight, nn.init.calculate_gain("relu"))
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    @property
    def is_blind(self):
        """True if the observation space doesn't include `self.rgb_uuid` or
        `self.depth_uuid`."""
        return self._n_input_rgb + self._n_input_depth == 0

    def forward(self, observations: Dict[str, torch.Tensor]):  # type: ignore
        if self.is_blind:
            return None

        def check_use_agent(new_setting):
            if use_agent is not None:
                assert (
                        use_agent is new_setting
                ), "rgb and depth must both use an agent dim or none"
            return new_setting

        cnn_output_list: List[torch.Tensor] = []
        use_agent: Optional[bool] = None

        if self._n_input_rgb > 0:
            use_agent = check_use_agent(len(observations[self.rgb_uuid]) == 6)
            cnn_output_list.append(
                compute_cnn_output(self.rgb_cnn, observations[self.rgb_uuid])
            )

        if self._n_input_depth > 0:
            use_agent = check_use_agent(len(observations[self.depth_uuid]) == 6)
            cnn_output_list.append(
                compute_cnn_output(self.depth_cnn, observations[self.depth_uuid])
            )

        if self._n_input_seg > 0:
            use_agent = check_use_agent(len(observations[self.seg_uuid]) == 6)
            cnn_output_list.append(
                compute_cnn_output(self.seg_cnn, observations[self.seg_uuid])
            )

        if use_agent:
            channels_dim = 3  # [step, sampler, agent, channel (, height, width)]
        else:
            channels_dim = 2  # [step, sampler, channel (, height, width)]

        return torch.cat(cnn_output_list, dim=channels_dim)


class SA(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.magic_number = self.hidden_size ** 0.5

        self.embedding = nn.Sequential(nn.Linear(input_size, 3 * hidden_size))
        self.layer_norm_a = nn.Sequential(nn.LayerNorm(hidden_size))
        self.layer_norm_b = nn.Sequential(nn.LayerNorm(hidden_size))
        self.MLP = nn.Sequential(nn.Linear(hidden_size, hidden_size))

    def obtain_k_q_v(self, x):
        y = self.embedding(x)
        k = y[:,:,:,:,:self.hidden_size]
        q = y[:,:,:,:,self.hidden_size:2*self.hidden_size]
        v = y[:,:,:,:,2*self.hidden_size:3*self.hidden_size]
        return k, q, v

    def forward(self, x):
        x = x.transpose(2,3)
        k, q, v = self.obtain_k_q_v(x)
        k = k.transpose(3,4)
        s = nn.functional.softmax(torch.matmul(q, k) / self.magic_number, 4)
        z = torch.matmul(s, v) + x
        z_norm = self.layer_norm_a(z)
        f = self.MLP(z_norm) + z_norm
        f_norm = self.layer_norm_b(f)
        out = f_norm.transpose(2,3)
        return out


class PointNavActorCriticSimpleConvRNN(ActorCriticModel[CategoricalDistr]):
    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        goal_sensor_uuid: str,
        hidden_size=512,
        embed_coordinates=False,
        coordinate_embedding_dim=8,
        coordinate_dims=2,
        num_rnn_layers=1,
        rnn_type="GRU",
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)

        self.goal_sensor_uuid = goal_sensor_uuid
        self._hidden_size = hidden_size
        self.embed_coordinates = embed_coordinates
        if self.embed_coordinates:
            self.coorinate_embedding_size = coordinate_embedding_dim
        else:
            self.coorinate_embedding_size = coordinate_dims

        self.sensor_fusion = False
        if "rgb" in observation_space.spaces and "depth" in observation_space.spaces:
            self.sensor_fuser = nn.Linear(hidden_size * 2, hidden_size)
            self.sensor_fusion = True

        self.visual_encoder = SimpleCNN(observation_space, hidden_size)

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self.recurrent_hidden_state_size)
            + self.coorinate_embedding_size,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
        )

        self.actor = LinearActorHead(self._hidden_size, action_space.n)
        self.critic = LinearCriticHead(self._hidden_size)

        if self.embed_coordinates:
            self.coordinate_embedding = nn.Linear(
                coordinate_dims, coordinate_embedding_dim
            )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_target_coordinates_encoding(self, observations):
        if self.embed_coordinates:
            return self.coordinate_embedding(
                observations[self.goal_sensor_uuid].to(torch.float32)
            )
        else:
            return observations[self.goal_sensor_uuid].to(torch.float32)

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size

    def _recurrent_memory_specification(self):
        return dict(
            rnn=(
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            )
        )

    def forward(  # type:ignore
        self,
        observations: ObservationType,
        memory: Memory,
        prev_actions: torch.Tensor,
        masks: torch.FloatTensor,
        current_actions: torch.FloatTensor = None,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        target_encoding = self.get_target_coordinates_encoding(observations)
        x: Union[torch.Tensor, List[torch.Tensor]]
        x = [target_encoding]

        # if observations["rgb"].shape[0] != 1:
        #     print("rgb", (observations["rgb"][...,0,0,:].unsqueeze(-2).unsqueeze(-2) == observations["rgb"][...,0,0,:]).float().mean())
        #     if "depth" in observations:
        #         print("depth", (observations["depth"][...,0,0,:].unsqueeze(-2).unsqueeze(-2) == observations["depth"][...,0,0,:]).float().mean())

        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            if self.sensor_fusion:
                perception_embed = self.sensor_fuser(perception_embed)
            x = [perception_embed] + x

        x = torch.cat(x, dim=-1)
        x, rnn_hidden_states = self.state_encoder(x, memory.tensor("rnn"), masks)

        ac_output = ActorCriticOutput(
            distributions=self.actor(x), values=self.critic(x), extras={}
        )

        return ac_output, memory.set_tensor("rnn", rnn_hidden_states)


class PointNavMAActorCriticSimpleConvRNN(ActorCriticModel[CategoricalDistr]):
    def __init__(
            self,
            action_space: gym.spaces.Discrete,
            observation_space: SpaceDict,
            goal_sensor_uuid: str,
            hidden_size=512,
            embed_coordinates=False,
            coordinate_embedding_dim=8,
            coordinate_dims=2,
            missing_action_embedding_dim=32,
            missing_action_uuid="missing_action",
            num_rnn_layers=1,
            rnn_type="GRU",
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)

        self.goal_sensor_uuid = goal_sensor_uuid
        self.missing_action_uuid = missing_action_uuid
        self._hidden_size = hidden_size
        self.embed_coordinates = embed_coordinates
        if self.embed_coordinates:
            self.coorinate_embedding_size = coordinate_embedding_dim
        else:
            self.coorinate_embedding_size = coordinate_dims

        self.sensor_fusion = False
        if "rgb" in observation_space.spaces and "depth" in observation_space.spaces:
            self.sensor_fuser = nn.Linear(hidden_size * 2, hidden_size)
            self.sensor_fusion = True

        self.visual_encoder = SimpleCNN(observation_space, hidden_size)

        # Missing Action embedding
        self.missing_action_embedding = nn.Embedding(
            num_embeddings=action_space.n, embedding_dim=missing_action_embedding_dim
        )

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self.recurrent_hidden_state_size)
            + self.coorinate_embedding_size + missing_action_embedding_dim,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
            )

        self.actor = LinearActorHead(self._hidden_size, action_space.n)
        self.critic = LinearCriticHead(self._hidden_size)

        if self.embed_coordinates:
            self.coordinate_embedding = nn.Linear(
                coordinate_dims, coordinate_embedding_dim
            )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_target_coordinates_encoding(self, observations):
        if self.embed_coordinates:
            return self.coordinate_embedding(
                observations[self.goal_sensor_uuid].to(torch.float32)
            )
        else:
            return observations[self.goal_sensor_uuid].to(torch.float32)

    def get_missing_action_embedding(self, observations):
        return self.missing_action_embedding(observations[self.missing_action_uuid].to(torch.int64))

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size

    def _recurrent_memory_specification(self):
        return dict(
            rnn=(
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            )
        )

    def forward(  # type:ignore
            self,
            observations: ObservationType,
            memory: Memory,
            prev_actions: torch.Tensor,
            masks: torch.FloatTensor,
            current_actions: torch.FloatTensor = None,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        target_encoding = self.get_target_coordinates_encoding(observations)
        x: Union[torch.Tensor, List[torch.Tensor]]
        x = [target_encoding]

        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            if self.sensor_fusion:
                perception_embed = self.sensor_fuser(perception_embed)
            x = [perception_embed] + x

        missing_action_feat = self.get_missing_action_embedding(observations)
        x = [missing_action_feat] + x

        x = torch.cat(x, dim=-1)
        x, rnn_hidden_states = self.state_encoder(x, memory.tensor("rnn"), masks)

        ac_output = ActorCriticOutput(
            distributions=self.actor(x), values=self.critic(x), extras={}
        )

        return ac_output, memory.set_tensor("rnn", rnn_hidden_states)


class PointNavPBLActorCriticSimpleConvRNN(ActorCriticModel[CategoricalDistr]):
    def __init__(
            self,
            action_space: gym.spaces.Discrete,
            observation_space: SpaceDict,
            goal_sensor_uuid: str,
            hidden_size=512,
            embed_coordinates=False,
            coordinate_embedding_dim=8,
            coordinate_dims=2,
            num_rnn_layers=1,
            rnn_type="GRU",
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)

        self.goal_sensor_uuid = goal_sensor_uuid
        self._hidden_size = hidden_size
        self.embed_coordinates = embed_coordinates
        if self.embed_coordinates:
            self.coorinate_embedding_size = coordinate_embedding_dim
        else:
            self.coorinate_embedding_size = coordinate_dims

        self.sensor_fusion = False
        if "rgb" in observation_space.spaces and "depth" in observation_space.spaces:
            self.sensor_fuser = nn.Linear(hidden_size * 2, hidden_size)
            self.sensor_fusion = True
        self.sensor_fuser_2 = nn.Linear(self.coorinate_embedding_size + hidden_size, hidden_size)

        self.visual_encoder = SimpleCNN(observation_space, hidden_size)

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self.recurrent_hidden_state_size) + 1,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
            )

        self.state_encoder_predict = RNNStateEncoder(
            1,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
        )

        self.g_forward = nn.Sequential(
            nn.Linear(self._hidden_size, self._hidden_size),
            nn.ReLU(),
            nn.Linear(self._hidden_size, self._hidden_size),
        )

        self.g_reverse = nn.Sequential(
            nn.Linear(self._hidden_size, self._hidden_size),
            nn.ReLU(),
            nn.Linear(self._hidden_size, self._hidden_size),
        )

        self.actor = LinearActorHead(self._hidden_size, action_space.n)
        self.critic = LinearCriticHead(self._hidden_size)

        if self.embed_coordinates:
            self.coordinate_embedding = nn.Linear(
                coordinate_dims, coordinate_embedding_dim
            )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_target_coordinates_encoding(self, observations):
        if self.embed_coordinates:
            return self.coordinate_embedding(
                observations[self.goal_sensor_uuid].to(torch.float32)
            )
        else:
            return observations[self.goal_sensor_uuid].to(torch.float32)

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size

    def _recurrent_memory_specification(self):
        return dict(
            rnn=(
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            )
        )

    def forward(  # type:ignore
            self,
            observations: ObservationType,
            memory: Memory,
            prev_actions: torch.Tensor,
            masks: torch.FloatTensor,
            current_actions: torch.FloatTensor = None,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        target_encoding = self.get_target_coordinates_encoding(observations)
        x: Union[torch.Tensor, List[torch.Tensor]]
        x = [target_encoding]

        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            if self.sensor_fusion:
                perception_embed = self.sensor_fuser(perception_embed)
            x = torch.cat([perception_embed] + x, dim=-1)
            x_obs = self.sensor_fuser_2(x)

        x = [x_obs] + [prev_actions.squeeze(-1)]
        x = torch.cat(x, dim=-1)
        x, rnn_hidden_states = self.state_encoder(x, memory.tensor("rnn"), masks)

        out = ActorCriticOutput(
            distributions=self.actor(x), values=self.critic(x), extras={}
        )

        if not isinstance(current_actions, type(None)):
            nt, ng = current_actions.shape[0], current_actions.shape[1]
            x_action = current_actions.squeeze(-1).view(1, nt * ng, 1).float()
            b = x.view(1, nt * ng, -1)
            m = masks.view(1, nt * ng, 1, 1)
            x_pred, _ = self.state_encoder_predict(x_action, b, m)
            x_pred = x_pred.view(nt, ng, -1)
            x_forward = self.g_forward(x_pred)
            #x_forward = 0.02 * (((x_forward ** 2) - 1) ** 2)
            x_forward = x_forward / (torch.norm(x_forward, dim=-1, keepdim=True) + 10**(-8))
            #x_obs = 0.02 * (((x_obs ** 2) - 1) ** 2)
            x_obs = x_obs / (torch.norm(x_obs, dim=-1, keepdim=True) + 10**(-8))
            x_reverse = self.g_reverse(x_obs)
            b = b.view(nt, ng, -1)
            out = {"ac_output": out, "x_obs": x_obs,
                   "x_forward": x_forward, "x_reverse": x_reverse, "b": b}

        return out, memory.set_tensor("rnn", rnn_hidden_states)


class PointNavCPCActorCriticSimpleConvRNN(ActorCriticModel[CategoricalDistr]):
    def __init__(
            self,
            action_space: gym.spaces.Discrete,
            observation_space: SpaceDict,
            goal_sensor_uuid: str,
            hidden_size=512,
            embed_coordinates=False,
            coordinate_embedding_dim=8,
            coordinate_dims=2,
            num_rnn_layers=1,
            rnn_type="GRU",
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)

        self.goal_sensor_uuid = goal_sensor_uuid
        self._hidden_size = hidden_size
        self.embed_coordinates = embed_coordinates
        if self.embed_coordinates:
            self.coorinate_embedding_size = coordinate_embedding_dim
        else:
            self.coorinate_embedding_size = coordinate_dims

        self.sensor_fusion = False
        if "rgb" in observation_space.spaces and "depth" in observation_space.spaces:
            self.sensor_fuser = nn.Linear(hidden_size * 2, hidden_size)
            self.sensor_fusion = True
        self.sensor_fuser_2 = nn.Linear(self.coorinate_embedding_size + hidden_size, hidden_size)

        self.visual_encoder = SimpleCNN(observation_space, hidden_size)

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self.recurrent_hidden_state_size) + 1,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
            )

        self.state_encoder_predict = RNNStateEncoder(
            1,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
        )

        self.classifier = nn.Sequential(
            nn.Linear(self._hidden_size * 2, self._hidden_size),
            nn.ReLU(),
            nn.Linear(self._hidden_size, 1),
        )

        self.actor = LinearActorHead(self._hidden_size, action_space.n)
        self.critic = LinearCriticHead(self._hidden_size)

        if self.embed_coordinates:
            self.coordinate_embedding = nn.Linear(
                coordinate_dims, coordinate_embedding_dim
            )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_target_coordinates_encoding(self, observations):
        if self.embed_coordinates:
            return self.coordinate_embedding(
                observations[self.goal_sensor_uuid].to(torch.float32)
            )
        else:
            return observations[self.goal_sensor_uuid].to(torch.float32)

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size

    def _recurrent_memory_specification(self):
        return dict(
            rnn=(
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            )
        )

    def forward(  # type:ignore
            self,
            observations: ObservationType,
            memory: Memory,
            prev_actions: torch.Tensor,
            masks: torch.FloatTensor,
            current_actions: torch.FloatTensor = None,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        target_encoding = self.get_target_coordinates_encoding(observations)
        x: Union[torch.Tensor, List[torch.Tensor]]
        x = [target_encoding]

        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            if self.sensor_fusion:
                perception_embed = self.sensor_fuser(perception_embed)
            x = torch.cat([perception_embed] + x, dim=-1)
            x_obs = self.sensor_fuser_2(x)

        x = [x_obs] + [prev_actions.squeeze(-1)]
        x = torch.cat(x, dim=-1)
        x, rnn_hidden_states = self.state_encoder(x, memory.tensor("rnn"), masks)

        out = ActorCriticOutput(
            distributions=self.actor(x), values=self.critic(x), extras={}
        )

        if not isinstance(current_actions, type(None)):
            nt, ng = current_actions.shape[0], current_actions.shape[1]
            x_action = current_actions.squeeze(-1).view(1, nt * ng, 1).float()
            b = x.view(1, nt * ng, -1)
            m = masks.view(1, nt * ng, 1, 1)
            x_pred, _ = self.state_encoder_predict(x_action, b, m)
            x_pred = x_pred.view(nt, ng, -1)[:-1]

            x_obs_positive = x_obs[1:]
            concat_positive = torch.cat([x_pred, x_obs_positive], dim=-1)
            logit_positive = self.classifier(concat_positive)

            idx = torch.randperm(nt).to(x_obs.device)
            x_obs_negative = x_obs[idx, :, :][1:]
            concat_negative = torch.cat([x_pred, x_obs_negative], dim=-1)
            logit_negative = self.classifier(concat_negative)

            out = {"ac_output": out, "logit_positive": logit_positive, "logit_negative": logit_negative}

        return out, memory.set_tensor("rnn", rnn_hidden_states)


class PointNavRGBDSActorCriticSimpleConvRNN(ActorCriticModel[CategoricalDistr]):
    def __init__(
            self,
            action_space: gym.spaces.Discrete,
            observation_space: SpaceDict,
            goal_sensor_uuid: str,
            hidden_size=512,
            embed_coordinates=False,
            coordinate_embedding_dim=8,
            coordinate_dims=2,
            num_rnn_layers=1,
            rnn_type="GRU",
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)

        self.goal_sensor_uuid = goal_sensor_uuid
        self._hidden_size = hidden_size
        self.embed_coordinates = embed_coordinates
        if self.embed_coordinates:
            self.coorinate_embedding_size = coordinate_embedding_dim
        else:
            self.coorinate_embedding_size = coordinate_dims

        self.sensor_fusion = False
        if "rgb" in observation_space.spaces and "depth" in observation_space.spaces:
            self.sensor_fuser = nn.Linear(hidden_size * 3, hidden_size)
            self.sensor_fusion = True

        self.visual_encoder = RGBDSCNN(observation_space, hidden_size)

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self.recurrent_hidden_state_size)
            + self.coorinate_embedding_size,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
            )

        self.actor = LinearActorHead(self._hidden_size, action_space.n)
        self.critic = LinearCriticHead(self._hidden_size)

        if self.embed_coordinates:
            self.coordinate_embedding = nn.Linear(
                coordinate_dims, coordinate_embedding_dim
            )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_target_coordinates_encoding(self, observations):
        if self.embed_coordinates:
            return self.coordinate_embedding(
                observations[self.goal_sensor_uuid].to(torch.float32)
            )
        else:
            return observations[self.goal_sensor_uuid].to(torch.float32)

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size

    def _recurrent_memory_specification(self):
        return dict(
            rnn=(
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            )
        )

    def forward(  # type:ignore
            self,
            observations: ObservationType,
            memory: Memory,
            prev_actions: torch.Tensor,
            masks: torch.FloatTensor,
            current_actions: torch.FloatTensor = None,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        target_encoding = self.get_target_coordinates_encoding(observations)
        x: Union[torch.Tensor, List[torch.Tensor]]
        x = [target_encoding]

        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            if self.sensor_fusion:
                perception_embed = self.sensor_fuser(perception_embed)
            x = [perception_embed] + x

        x = torch.cat(x, dim=-1)
        x, rnn_hidden_states = self.state_encoder(x, memory.tensor("rnn"), masks)

        ac_output = ActorCriticOutput(
            distributions=self.actor(x), values=self.critic(x), extras={}
        )

        return ac_output, memory.set_tensor("rnn", rnn_hidden_states)


class PointNavKeyPointsNPMActorCriticSimpleConvRNN(ActorCriticModel[CategoricalDistr]):
    def __init__(
            self,
            action_space: gym.spaces.Discrete,
            observation_space: SpaceDict,
            goal_sensor_uuid: str,
            obstacle_keypoints_sensor_uuid: str,
            hidden_size=512,
            embed_coordinates=False,
            coordinate_embedding_dim=8,
            coordinate_dims=2,
            obstacle_type_embedding_dim=8,
            obstacle_state_hidden_dim=16,
            num_obstacle_types=20,
            num_rnn_layers=1,
            rnn_type="GRU",
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)

        self.goal_sensor_uuid = goal_sensor_uuid
        self._hidden_size = hidden_size
        self.embed_coordinates = embed_coordinates
        if self.embed_coordinates:
            self.coorinate_embedding_size = coordinate_embedding_dim
        else:
            self.coorinate_embedding_size = coordinate_dims
        self.obstacle_keypoints_sensor_uuid = obstacle_keypoints_sensor_uuid
        self.obstacle_type_embedding_size = obstacle_type_embedding_dim
        self.obstacle_state_hidden_dim = obstacle_state_hidden_dim

        self.sensor_fusion = False
        if "rgb" in observation_space.spaces and "depth" in observation_space.spaces:
            self.sensor_fuser = nn.Linear(hidden_size * 2, hidden_size)
            self.sensor_fusion = True

        self.visual_encoder = SimpleCNN(observation_space, hidden_size)

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self.recurrent_hidden_state_size)
            + self.coorinate_embedding_size + obstacle_state_hidden_dim * action_space.n,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
            )

        self.actor = LinearActorHead(self._hidden_size, action_space.n)
        self.critic = LinearCriticHead(self._hidden_size)

        if self.embed_coordinates:
            self.coordinate_embedding = nn.Linear(
                coordinate_dims, coordinate_embedding_dim
            )

        # Action embedding
        self.action_embedding = nn.Embedding(
            num_embeddings=action_space.n, embedding_dim=self.obstacle_state_hidden_dim
        )
        self.num_actions = self.action_space.n

        # Object hidden state encoding
        self.meta_embedding = nn.Embedding(
            num_embeddings=num_obstacle_types, embedding_dim=self.obstacle_state_hidden_dim
        )
        self.rotation_encoding = nn.Sequential(
            nn.Linear(24, self.obstacle_state_hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim // 2, self.obstacle_state_hidden_dim),
        )
        self.state_encoding = nn.Sequential(
            nn.Linear(3, self.obstacle_state_hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim // 2, self.obstacle_state_hidden_dim),
        )

        # NPM
        self.NPM = nn.Sequential(
            nn.Linear(self.obstacle_state_hidden_dim * 3, self.obstacle_state_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim, self.obstacle_state_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim, 12)
        )
        self.NPM[4].weight.data.zero_()
        self.NPM[4].bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float))

        # NPM attention
        self.NPM_atten = nn.Sequential(
            nn.Linear(self.obstacle_state_hidden_dim * 3, self.obstacle_state_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim, 1)
        )

        # NPM Summary
        self.NPM_summary = nn.Sequential(
            nn.Linear(self.obstacle_state_hidden_dim * 4, self.obstacle_state_hidden_dim * 3),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim * 3, self.obstacle_state_hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim * 2, self.obstacle_state_hidden_dim),
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_target_coordinates_encoding(self, observations):
        if self.embed_coordinates:
            return self.coordinate_embedding(
                observations[self.goal_sensor_uuid].to(torch.float32)
            )
        else:
            return observations[self.goal_sensor_uuid].to(torch.float32)

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size

    def _recurrent_memory_specification(self):
        return dict(
            rnn=(
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            )
        )

    def forward(  # type:ignore
            self,
            observations: ObservationType,
            memory: Memory,
            prev_actions: torch.Tensor,
            masks: torch.FloatTensor,
            current_actions: torch.FloatTensor = None,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        target_encoding = self.get_target_coordinates_encoding(observations)
        x: Union[torch.Tensor, List[torch.Tensor]]
        x = [target_encoding]

        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            if self.sensor_fusion:
                perception_embed = self.sensor_fuser(perception_embed)
            x = [perception_embed] + x

        nb, ng, no, np, nd = observations[self.obstacle_keypoints_sensor_uuid].shape
        nh = self.obstacle_state_hidden_dim

        keypoints = observations[self.obstacle_keypoints_sensor_uuid].view(nb, ng, no, np, nd)
        obstacles_index = torch.arange(0, no).to(target_encoding.device).long()
        obstacles_meta_hidden = self.meta_embedding(obstacles_index)
        obstacles_rot_hidden = self.rotation_encoding(keypoints.view(nb, ng, no, np*nd))
        obstacles_state_hidden = self.state_encoding(keypoints.mean(3))

        na = self.num_actions
        actions_index = torch.arange(0, na).to(target_encoding.device).long()
        a_feature = self.action_embedding(actions_index).view(-1, na, nh)

        keypoints = keypoints.view(nb, ng, no, 1, np, nd).repeat(1, 1, 1, na, 1, 1)
        keypoints_homo = torch.cat((keypoints, torch.ones(nb, ng, no, na, np, 1).to(target_encoding.device)), 5)
        obstacles_meta_hidden = obstacles_meta_hidden.view(1, 1, no, 1, nh).repeat(nb, ng, 1, na, 1)
        obstacles_rot_hidden = obstacles_rot_hidden.view(nb, ng, no, 1, nh).repeat(1, 1, 1, na, 1)
        obstacles_state_hidden = obstacles_state_hidden.view(nb, ng, no, 1, nh).repeat(1, 1, 1, na, 1)
        a_feature = a_feature.view(1, 1, 1, na, nh).repeat(nb, ng, no, 1, 1)

        hidden_feature = torch.cat((obstacles_rot_hidden, obstacles_meta_hidden, a_feature), dim=4)
        NPM_hidden = self.NPM(hidden_feature)
        NPM_hidden = NPM_hidden
        M = NPM_hidden.view(nb, ng, no, na, 3, 4)
        """
        M_test = M.clone()
        M_test[:, :, :, 7] = torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]).to(M.device)
        out, rnn_hidden_states, new_keypoints = self.get_action(M, keypoints_homo, obstacles_rot_hidden,
                                                                obstacles_meta_hidden, a_feature, obstacles_state_hidden,
                                                                nb, ng, x, memory, masks, keypoints)
        out_test, _, new_keypoints_test = self.get_action(M_test, keypoints_homo, obstacles_rot_hidden,
                                                                obstacles_meta_hidden, a_feature, obstacles_state_hidden,
                                                                nb, ng, x, memory, masks, keypoints, True)
        """
        new_keypoints = torch.matmul(M, keypoints_homo.transpose(4, 5)).transpose(4, 5)
        new_obstacles_state_hidden = self.state_encoding(new_keypoints.mean(4))

        atten_feature = torch.cat((obstacles_rot_hidden, obstacles_meta_hidden, a_feature), dim=4)
        hidden_feature = torch.cat((obstacles_meta_hidden, obstacles_state_hidden, new_obstacles_state_hidden,
                                    a_feature), dim=4)
        NPM_atten_score = self.NPM_atten(atten_feature)
        NPM_atten_prob = nn.functional.softmax(NPM_atten_score, 2)
        NPM_atten_hidden = (hidden_feature * NPM_atten_prob).sum(2)
        NPM_atten_hidden = self.NPM_summary(NPM_atten_hidden)
        NPM_atten_hidden = NPM_atten_hidden.view(nb, ng, -1)
        x.append(NPM_atten_hidden)

        x = torch.cat(x, dim=-1)
        x, rnn_hidden_states = self.state_encoder(x, memory.tensor("rnn"), masks)

        out = ActorCriticOutput(
            distributions=self.actor(x), values=self.critic(x), extras={}
        )

        if not isinstance(current_actions, type(None)):
            new_keypoints = new_keypoints.view(nb * ng, no, na, 8, 3).transpose(1, 2)
            current_actions = current_actions.view(nb * ng, 1, 1).squeeze()
            NPM_out = new_keypoints[torch.arange(nb * ng), current_actions].reshape(nb, ng, no, 8, 3)
            out = {"ac_output": out, "npm_output": NPM_out}

        return out, memory.set_tensor("rnn", rnn_hidden_states)

    def get_action(self, M, keypoints_homo, obstacles_rot_hidden, obstacles_meta_hidden, a_feature, obstacles_state_hidden,
                   nb, ng, x, memory, masks, keypoints, second=False):
        new_keypoints = torch.matmul(M, keypoints_homo.transpose(4, 5)).transpose(4, 5)
        if second:
            new_keypoints[:,:,:,7] = keypoints[:,:,:,7].clone()
        new_obstacles_state_hidden = self.state_encoding(new_keypoints.mean(4))

        atten_feature = torch.cat((obstacles_rot_hidden, obstacles_meta_hidden, a_feature), dim=4)
        hidden_feature = torch.cat((obstacles_meta_hidden, obstacles_state_hidden, new_obstacles_state_hidden,
                                    a_feature), dim=4)
        NPM_atten_score = self.NPM_atten(atten_feature)
        NPM_atten_prob = nn.functional.softmax(NPM_atten_score, 2)
        NPM_atten_hidden = (hidden_feature * NPM_atten_prob).sum(2)
        NPM_atten_hidden = self.NPM_summary(NPM_atten_hidden)
        NPM_atten_hidden = NPM_atten_hidden.view(nb, ng, -1)
        if second:
            x[-1] = NPM_atten_hidden
        else:
            x.append(NPM_atten_hidden)

        y = torch.cat(x, dim=-1)
        y, rnn_hidden_states = self.state_encoder(y, memory.tensor("rnn"), masks)

        out = ActorCriticOutput(
            distributions=self.actor(y), values=self.critic(y), extras={}
        )
        return out, rnn_hidden_states, new_keypoints


class PointNavKeyPointsVisualNPMActorCriticSimpleConvRNN(ActorCriticModel[CategoricalDistr]):
    def __init__(
            self,
            action_space: gym.spaces.Discrete,
            observation_space: SpaceDict,
            goal_sensor_uuid: str,
            obstacle_keypoints_sensor_uuid: str,
            hidden_size=512,
            embed_coordinates=False,
            coordinate_embedding_dim=8,
            coordinate_dims=2,
            obstacle_type_embedding_dim=8,
            obstacle_state_hidden_dim=16,
            num_obstacle_types=20,
            num_rnn_layers=1,
            rnn_type="GRU",
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)

        self.goal_sensor_uuid = goal_sensor_uuid
        self._hidden_size = hidden_size
        self.embed_coordinates = embed_coordinates
        if self.embed_coordinates:
            self.coorinate_embedding_size = coordinate_embedding_dim
        else:
            self.coorinate_embedding_size = coordinate_dims
        self.obstacle_keypoints_sensor_uuid = obstacle_keypoints_sensor_uuid
        self.obstacle_type_embedding_size = obstacle_type_embedding_dim
        self.obstacle_state_hidden_dim = obstacle_state_hidden_dim

        self.sensor_fusion = False
        if "rgb" in observation_space.spaces and "depth" in observation_space.spaces:
            self.sensor_fuser = nn.Linear(hidden_size * 2, hidden_size)
            self.sensor_fusion = True

        self.visual_encoder = SimpleCNN(observation_space, hidden_size)

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self.recurrent_hidden_state_size)
            + self.coorinate_embedding_size + obstacle_state_hidden_dim * action_space.n,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
            )

        self.actor = LinearActorHead(self._hidden_size, action_space.n)
        self.critic = LinearCriticHead(self._hidden_size)

        if self.embed_coordinates:
            self.coordinate_embedding = nn.Linear(
                coordinate_dims, coordinate_embedding_dim
            )

        # Action embedding
        self.action_embedding = nn.Embedding(
            num_embeddings=action_space.n, embedding_dim=self.obstacle_state_hidden_dim
        )
        self.num_actions = self.action_space.n

        # Object hidden state encoding
        self.meta_embedding = nn.Embedding(
            num_embeddings=num_obstacle_types, embedding_dim=self.obstacle_state_hidden_dim
        )
        self.rotation_encoding = nn.Sequential(
            nn.Linear(24, self.obstacle_state_hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim // 2, self.obstacle_state_hidden_dim),
        )
        self.state_encoding = nn.Sequential(
            nn.Linear(3, self.obstacle_state_hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim // 2, self.obstacle_state_hidden_dim),
        )

        # NPM
        self.NPM = nn.Sequential(
            nn.Linear(hidden_size + self.obstacle_state_hidden_dim * 3, self.obstacle_state_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim, self.obstacle_state_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim, 12)
        )
        self.NPM[4].weight.data.zero_()
        self.NPM[4].bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float))

        # NPM attention
        self.NPM_atten = nn.Sequential(
            nn.Linear(self.obstacle_state_hidden_dim * 3, self.obstacle_state_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim, 1)
        )

        # NPM Summary
        self.NPM_summary = nn.Sequential(
            nn.Linear(self.obstacle_state_hidden_dim * 4, self.obstacle_state_hidden_dim * 3),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim * 3, self.obstacle_state_hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim * 2, self.obstacle_state_hidden_dim),
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_target_coordinates_encoding(self, observations):
        if self.embed_coordinates:
            return self.coordinate_embedding(
                observations[self.goal_sensor_uuid].to(torch.float32)
            )
        else:
            return observations[self.goal_sensor_uuid].to(torch.float32)

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size

    def _recurrent_memory_specification(self):
        return dict(
            rnn=(
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            )
        )

    def forward(  # type:ignore
            self,
            observations: ObservationType,
            memory: Memory,
            prev_actions: torch.Tensor,
            masks: torch.FloatTensor,
            current_actions: torch.FloatTensor = None,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        target_encoding = self.get_target_coordinates_encoding(observations)
        x: Union[torch.Tensor, List[torch.Tensor]]
        x = [target_encoding]

        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            if self.sensor_fusion:
                perception_embed = self.sensor_fuser(perception_embed)
            x = [perception_embed] + x

        nb, ng, no, np, nd = observations[self.obstacle_keypoints_sensor_uuid].shape
        nh = self.obstacle_state_hidden_dim

        keypoints = observations[self.obstacle_keypoints_sensor_uuid].view(nb, ng, no, np, nd)
        obstacles_index = torch.arange(0, no).to(target_encoding.device).long()
        obstacles_meta_hidden = self.meta_embedding(obstacles_index)
        obstacles_rot_hidden = self.rotation_encoding(keypoints.view(nb, ng, no, np*nd))
        obstacles_state_hidden = self.state_encoding(keypoints.mean(3))

        na = self.num_actions
        actions_index = torch.arange(0, na).to(target_encoding.device).long()
        a_feature = self.action_embedding(actions_index).view(-1, na, nh)

        keypoints = keypoints.view(nb, ng, no, 1, np, nd).repeat(1, 1, 1, na, 1, 1)
        keypoints_homo = torch.cat((keypoints, torch.ones(nb, ng, no, na, np, 1).to(target_encoding.device)), 5)
        obstacles_meta_hidden = obstacles_meta_hidden.view(1, 1, no, 1, nh).repeat(nb, ng, 1, na, 1)
        obstacles_rot_hidden = obstacles_rot_hidden.view(nb, ng, no, 1, nh).repeat(1, 1, 1, na, 1)
        obstacles_state_hidden = obstacles_state_hidden.view(nb, ng, no, 1, nh).repeat(1, 1, 1, na, 1)
        a_feature = a_feature.view(1, 1, 1, na, nh).repeat(nb, ng, no, 1, 1)
        perception_embed_hidden = perception_embed.view(nb, ng, 1, 1, self._hidden_size).repeat(1, 1, no, na, 1)

        hidden_feature = torch.cat((perception_embed_hidden, obstacles_rot_hidden, obstacles_meta_hidden, a_feature), dim=4)
        NPM_hidden = self.NPM(hidden_feature)
        NPM_hidden = NPM_hidden
        M = NPM_hidden.view(nb, ng, no, na, 3, 4)
        M_test = M.clone()
        M_test[:, :, :, 7] = torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]).to(M.device)
        M_test[:, :, :, 5] = torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]).to(M.device)
        out, rnn_hidden_states, new_keypoints = self.get_action(M, keypoints_homo, obstacles_rot_hidden,
                                                                obstacles_meta_hidden, a_feature, obstacles_state_hidden,
                                                                nb, ng, x, memory, masks, keypoints)
        out_test, _, new_keypoints_test = self.get_action(M_test, keypoints_homo, obstacles_rot_hidden,
                                                                obstacles_meta_hidden, a_feature, obstacles_state_hidden,
                                                                nb, ng, x, memory, masks, keypoints, True)
        """
        new_keypoints = torch.matmul(M, keypoints_homo.transpose(4, 5)).transpose(4, 5)
        new_obstacles_state_hidden = self.state_encoding(new_keypoints.mean(4))

        atten_feature = torch.cat((obstacles_rot_hidden, obstacles_meta_hidden, a_feature), dim=4)
        hidden_feature = torch.cat((obstacles_meta_hidden, obstacles_state_hidden, new_obstacles_state_hidden,
                                    a_feature), dim=4)
        NPM_atten_score = self.NPM_atten(atten_feature)
        NPM_atten_prob = nn.functional.softmax(NPM_atten_score, 2)
        NPM_atten_hidden = (hidden_feature * NPM_atten_prob).sum(2)
        NPM_atten_hidden = self.NPM_summary(NPM_atten_hidden)
        NPM_atten_hidden = NPM_atten_hidden.view(nb, ng, -1)
        x.append(NPM_atten_hidden)

        x = torch.cat(x, dim=-1)
        x, rnn_hidden_states = self.state_encoder(x, memory.tensor("rnn"), masks)

        out = ActorCriticOutput(
            distributions=self.actor(x), values=self.critic(x), extras={}
        )
        """

        if not isinstance(current_actions, type(None)):
            new_keypoints = new_keypoints.view(nb * ng, no, na, 8, 3).transpose(1, 2)
            current_actions = current_actions.view(nb * ng, 1, 1).squeeze()
            NPM_out = new_keypoints[torch.arange(nb * ng), current_actions].reshape(nb, ng, no, 8, 3)
            out = {"ac_output": out, "npm_output": NPM_out}

        return out, memory.set_tensor("rnn", rnn_hidden_states)

    def get_action(self, M, keypoints_homo, obstacles_rot_hidden, obstacles_meta_hidden, a_feature, obstacles_state_hidden,
                   nb, ng, x, memory, masks, keypoints, second=False):
        new_keypoints = torch.matmul(M, keypoints_homo.transpose(4, 5)).transpose(4, 5)
        if second:
            new_keypoints[:,:,:,7] = keypoints[:,:,:,7].clone()
            new_keypoints[:, :, :, 5] = keypoints[:, :, :, 5].clone()
        new_obstacles_state_hidden = self.state_encoding(new_keypoints.mean(4))

        atten_feature = torch.cat((obstacles_rot_hidden, obstacles_meta_hidden, a_feature), dim=4)
        hidden_feature = torch.cat((obstacles_meta_hidden, obstacles_state_hidden, new_obstacles_state_hidden,
                                    a_feature), dim=4)
        NPM_atten_score = self.NPM_atten(atten_feature)
        NPM_atten_prob = nn.functional.softmax(NPM_atten_score, 2)
        NPM_atten_hidden = (hidden_feature * NPM_atten_prob).sum(2)
        NPM_atten_hidden = self.NPM_summary(NPM_atten_hidden)
        NPM_atten_hidden = NPM_atten_hidden.view(nb, ng, -1)
        if second:
            x[-1] = NPM_atten_hidden
        else:
            x.append(NPM_atten_hidden)

        y = torch.cat(x, dim=-1)
        y, rnn_hidden_states = self.state_encoder(y, memory.tensor("rnn"), masks)

        out = ActorCriticOutput(
            distributions=self.actor(y), values=self.critic(y), extras={}
        )
        return out, rnn_hidden_states, new_keypoints


class PlacementActorCriticSimpleConvRNN(ActorCriticModel[CategoricalDistr]):
    def __init__(
            self,
            action_space: gym.spaces.Discrete,
            observation_space: SpaceDict,
            goal_sensor_uuid: str,
            object_sensor_uuid: str,
            hidden_size=512,
            embed_coordinates=False,
            coordinate_embedding_dim=8,
            coordinate_dims=2,
            object_type_embedding_dim=8,
            num_rnn_layers=1,
            rnn_type="GRU",
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)

        self.goal_sensor_uuid = goal_sensor_uuid
        self.object_sensor_uuid = object_sensor_uuid
        self._hidden_size = hidden_size
        self.embed_coordinates = embed_coordinates
        if self.embed_coordinates:
            self.coorinate_embedding_size = coordinate_embedding_dim
        else:
            self.coorinate_embedding_size = coordinate_dims
        self._n_object_types = self.observation_space.spaces[self.object_sensor_uuid].n
        self.object_type_embedding_size = object_type_embedding_dim

        self.sensor_fusion = False
        if "rgb" in observation_space.spaces and "depth" in observation_space.spaces:
            self.sensor_fuser = nn.Linear(hidden_size * 2, hidden_size)
            self.sensor_fusion = True

        self.visual_encoder = SimpleCNN(observation_space, hidden_size)

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self.recurrent_hidden_state_size)
            + self.object_type_embedding_size + self.coorinate_embedding_size,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
            )

        self.actor = LinearActorHead(self._hidden_size, action_space.n)
        self.critic = LinearCriticHead(self._hidden_size)

        if self.embed_coordinates:
            self.coordinate_embedding = nn.Linear(
                coordinate_dims, coordinate_embedding_dim
            )

        self.object_type_embedding = nn.Embedding(
            num_embeddings=self._n_object_types,
            embedding_dim=object_type_embedding_dim,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_target_coordinates_encoding(self, observations):
        if self.embed_coordinates:
            return self.coordinate_embedding(
                observations[self.goal_sensor_uuid].to(torch.float32)
            )
        else:
            return observations[self.goal_sensor_uuid].to(torch.float32)

    def get_object_type_encoding(
        self, observations: Dict[str, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """Get the object type encoding from input batched observations."""
        # noinspection PyTypeChecker
        return self.object_type_embedding(  # type:ignore
            observations[self.object_sensor_uuid].to(torch.int64)
        )

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size

    def _recurrent_memory_specification(self):
        return dict(
            rnn=(
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            )
        )

    def forward(  # type:ignore
            self,
            observations: ObservationType,
            memory: Memory,
            prev_actions: torch.Tensor,
            masks: torch.FloatTensor,
            current_actions: torch.FloatTensor = None,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        target_encoding = self.get_target_coordinates_encoding(observations)
        object_encoding = self.get_object_type_encoding(
            cast(Dict[str, torch.FloatTensor], observations)
        )
        x: Union[torch.Tensor, List[torch.Tensor]]
        x = [target_encoding, object_encoding]

        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            if self.sensor_fusion:
                perception_embed = self.sensor_fuser(perception_embed)
            x = [perception_embed] + x

        x = torch.cat(x, dim=-1)
        x, rnn_hidden_states = self.state_encoder(x, memory.tensor("rnn"), masks)

        ac_output = ActorCriticOutput(
            distributions=self.actor(x), values=self.critic(x), extras={}
        )

        return ac_output, memory.set_tensor("rnn", rnn_hidden_states)


class PlacementPBLActorCriticSimpleConvRNN(ActorCriticModel[CategoricalDistr]):
    def __init__(
            self,
            action_space: gym.spaces.Discrete,
            observation_space: SpaceDict,
            goal_sensor_uuid: str,
            object_sensor_uuid: str,
            hidden_size=512,
            embed_coordinates=False,
            coordinate_embedding_dim=8,
            coordinate_dims=2,
            object_type_embedding_dim=8,
            num_rnn_layers=1,
            rnn_type="GRU",
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)

        self.goal_sensor_uuid = goal_sensor_uuid
        self.object_sensor_uuid = object_sensor_uuid
        self._hidden_size = hidden_size
        self.embed_coordinates = embed_coordinates
        if self.embed_coordinates:
            self.coorinate_embedding_size = coordinate_embedding_dim
        else:
            self.coorinate_embedding_size = coordinate_dims
        self._n_object_types = self.observation_space.spaces[self.object_sensor_uuid].n
        self.object_type_embedding_size = object_type_embedding_dim

        self.sensor_fusion = False
        if "rgb" in observation_space.spaces and "depth" in observation_space.spaces:
            self.sensor_fuser = nn.Linear(hidden_size * 2, hidden_size)
            self.sensor_fusion = True
        self.sensor_fuser_2 = nn.Linear(self.object_type_embedding_size + self.coorinate_embedding_size + hidden_size, hidden_size)

        self.visual_encoder = SimpleCNN(observation_space, hidden_size)

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self.recurrent_hidden_state_size) + 1,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
            )

        self.state_encoder_predict = RNNStateEncoder(
            1,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
        )

        self.g_forward = nn.Sequential(
            nn.Linear(self._hidden_size, self._hidden_size),
            nn.ReLU(),
            nn.Linear(self._hidden_size, self._hidden_size),
        )

        self.g_reverse = nn.Sequential(
            nn.Linear(self._hidden_size, self._hidden_size),
            nn.ReLU(),
            nn.Linear(self._hidden_size, self._hidden_size),
        )

        self.actor = LinearActorHead(self._hidden_size, action_space.n)
        self.critic = LinearCriticHead(self._hidden_size)

        if self.embed_coordinates:
            self.coordinate_embedding = nn.Linear(
                coordinate_dims, coordinate_embedding_dim
            )

        self.object_type_embedding = nn.Embedding(
            num_embeddings=self._n_object_types,
            embedding_dim=object_type_embedding_dim,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_target_coordinates_encoding(self, observations):
        if self.embed_coordinates:
            return self.coordinate_embedding(
                observations[self.goal_sensor_uuid].to(torch.float32)
            )
        else:
            return observations[self.goal_sensor_uuid].to(torch.float32)

    def get_object_type_encoding(
            self, observations: Dict[str, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """Get the object type encoding from input batched observations."""
        # noinspection PyTypeChecker
        return self.object_type_embedding(  # type:ignore
            observations[self.object_sensor_uuid].to(torch.int64)
        )

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size

    def _recurrent_memory_specification(self):
        return dict(
            rnn=(
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            )
        )

    def forward(  # type:ignore
            self,
            observations: ObservationType,
            memory: Memory,
            prev_actions: torch.Tensor,
            masks: torch.FloatTensor,
            current_actions: torch.FloatTensor = None,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        target_encoding = self.get_target_coordinates_encoding(observations)
        object_encoding = self.get_object_type_encoding(
            cast(Dict[str, torch.FloatTensor], observations)
        )
        x: Union[torch.Tensor, List[torch.Tensor]]
        x = [target_encoding, object_encoding]

        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            if self.sensor_fusion:
                perception_embed = self.sensor_fuser(perception_embed)
            x = torch.cat([perception_embed] + x, dim=-1)
            x_obs = self.sensor_fuser_2(x)

        x = [x_obs] + [prev_actions.squeeze(-1)]
        x = torch.cat(x, dim=-1)
        x, rnn_hidden_states = self.state_encoder(x, memory.tensor("rnn"), masks)

        out = ActorCriticOutput(
            distributions=self.actor(x), values=self.critic(x), extras={}
        )

        if not isinstance(current_actions, type(None)):
            nt, ng = current_actions.shape[0], current_actions.shape[1]
            x_action = current_actions.squeeze(-1).view(1, nt * ng, 1).float()
            b = x.view(1, nt * ng, -1)
            m = masks.view(1, nt * ng, 1, 1)
            x_pred, _ = self.state_encoder_predict(x_action, b, m)
            x_pred = x_pred.view(nt, ng, -1)
            x_forward = self.g_forward(x_pred)
            #x_forward = 0.02 * (((x_forward ** 2) - 1) ** 2)
            x_forward = x_forward / (torch.norm(x_forward, dim=-1, keepdim=True) + 10**(-8))
            #x_obs = 0.02 * (((x_obs ** 2) - 1) ** 2)
            x_obs = x_obs / (torch.norm(x_obs, dim=-1, keepdim=True) + 10**(-8))
            x_reverse = self.g_reverse(x_obs)
            b = b.view(nt, ng, -1)
            out = {"ac_output": out, "x_obs": x_obs,
                   "x_forward": x_forward, "x_reverse": x_reverse, "b": b}

        return out, memory.set_tensor("rnn", rnn_hidden_states)


class PlacementCPCActorCriticSimpleConvRNN(ActorCriticModel[CategoricalDistr]):
    def __init__(
            self,
            action_space: gym.spaces.Discrete,
            observation_space: SpaceDict,
            goal_sensor_uuid: str,
            object_sensor_uuid: str,
            hidden_size=512,
            embed_coordinates=False,
            coordinate_embedding_dim=8,
            coordinate_dims=2,
            object_type_embedding_dim=8,
            num_rnn_layers=1,
            rnn_type="GRU",
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)

        self.goal_sensor_uuid = goal_sensor_uuid
        self.object_sensor_uuid = object_sensor_uuid
        self._hidden_size = hidden_size
        self.embed_coordinates = embed_coordinates
        if self.embed_coordinates:
            self.coorinate_embedding_size = coordinate_embedding_dim
        else:
            self.coorinate_embedding_size = coordinate_dims
        self._n_object_types = self.observation_space.spaces[self.object_sensor_uuid].n
        self.object_type_embedding_size = object_type_embedding_dim

        self.sensor_fusion = False
        if "rgb" in observation_space.spaces and "depth" in observation_space.spaces:
            self.sensor_fuser = nn.Linear(hidden_size * 2, hidden_size)
            self.sensor_fusion = True
        self.sensor_fuser_2 = nn.Linear(self.object_type_embedding_size + self.coorinate_embedding_size + hidden_size, hidden_size)

        self.visual_encoder = SimpleCNN(observation_space, hidden_size)

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self.recurrent_hidden_state_size) + 1,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
            )

        self.state_encoder_predict = RNNStateEncoder(
            1,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
        )

        self.classifier = nn.Sequential(
            nn.Linear(self._hidden_size * 2, self._hidden_size),
            nn.ReLU(),
            nn.Linear(self._hidden_size, 1),
        )

        self.actor = LinearActorHead(self._hidden_size, action_space.n)
        self.critic = LinearCriticHead(self._hidden_size)

        if self.embed_coordinates:
            self.coordinate_embedding = nn.Linear(
                coordinate_dims, coordinate_embedding_dim
            )

        self.object_type_embedding = nn.Embedding(
            num_embeddings=self._n_object_types,
            embedding_dim=object_type_embedding_dim,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_target_coordinates_encoding(self, observations):
        if self.embed_coordinates:
            return self.coordinate_embedding(
                observations[self.goal_sensor_uuid].to(torch.float32)
            )
        else:
            return observations[self.goal_sensor_uuid].to(torch.float32)

    def get_object_type_encoding(
            self, observations: Dict[str, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """Get the object type encoding from input batched observations."""
        # noinspection PyTypeChecker
        return self.object_type_embedding(  # type:ignore
            observations[self.object_sensor_uuid].to(torch.int64)
        )

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size

    def _recurrent_memory_specification(self):
        return dict(
            rnn=(
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            )
        )

    def forward(  # type:ignore
            self,
            observations: ObservationType,
            memory: Memory,
            prev_actions: torch.Tensor,
            masks: torch.FloatTensor,
            current_actions: torch.FloatTensor = None,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        target_encoding = self.get_target_coordinates_encoding(observations)
        object_encoding = self.get_object_type_encoding(
            cast(Dict[str, torch.FloatTensor], observations)
        )
        x: Union[torch.Tensor, List[torch.Tensor]]
        x = [target_encoding, object_encoding]

        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            if self.sensor_fusion:
                perception_embed = self.sensor_fuser(perception_embed)
            x = torch.cat([perception_embed] + x, dim=-1)
            x_obs = self.sensor_fuser_2(x)

        x = [x_obs] + [prev_actions.squeeze(-1)]
        x = torch.cat(x, dim=-1)
        x, rnn_hidden_states = self.state_encoder(x, memory.tensor("rnn"), masks)

        out = ActorCriticOutput(
            distributions=self.actor(x), values=self.critic(x), extras={}
        )

        if not isinstance(current_actions, type(None)):
            nt, ng = current_actions.shape[0], current_actions.shape[1]
            x_action = current_actions.squeeze(-1).view(1, nt * ng, 1).float()
            b = x.view(1, nt * ng, -1)
            m = masks.view(1, nt * ng, 1, 1)
            x_pred, _ = self.state_encoder_predict(x_action, b, m)
            x_pred = x_pred.view(nt, ng, -1)[:-1]

            x_obs_positive = x_obs[1:]
            concat_positive = torch.cat([x_pred, x_obs_positive], dim=-1)
            logit_positive = self.classifier(concat_positive)

            idx = torch.randperm(nt).to(x_obs.device)
            x_obs_negative = x_obs[idx, :, :][1:]
            concat_negative = torch.cat([x_pred, x_obs_negative], dim=-1)
            logit_negative = self.classifier(concat_negative)

            out = {"ac_output": out, "logit_positive": logit_positive, "logit_negative": logit_negative}

        return out, memory.set_tensor("rnn", rnn_hidden_states)


class PlacementRGBDSActorCriticSimpleConvRNN(ActorCriticModel[CategoricalDistr]):
    def __init__(
            self,
            action_space: gym.spaces.Discrete,
            observation_space: SpaceDict,
            goal_sensor_uuid: str,
            object_sensor_uuid: str,
            hidden_size=512,
            embed_coordinates=False,
            coordinate_embedding_dim=8,
            coordinate_dims=2,
            object_type_embedding_dim=8,
            num_rnn_layers=1,
            rnn_type="GRU",
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)

        self.goal_sensor_uuid = goal_sensor_uuid
        self.object_sensor_uuid = object_sensor_uuid
        self._hidden_size = hidden_size
        self.embed_coordinates = embed_coordinates
        if self.embed_coordinates:
            self.coorinate_embedding_size = coordinate_embedding_dim
        else:
            self.coorinate_embedding_size = coordinate_dims
        self._n_object_types = self.observation_space.spaces[self.object_sensor_uuid].n
        self.object_type_embedding_size = object_type_embedding_dim

        self.sensor_fusion = False
        if "rgb" in observation_space.spaces and "depth" in observation_space.spaces:
            self.sensor_fuser = nn.Linear(hidden_size * 3, hidden_size)
            self.sensor_fusion = True

        self.visual_encoder = RGBDSCNN(observation_space, hidden_size)

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self.recurrent_hidden_state_size)
            + self.object_type_embedding_size + self.coorinate_embedding_size,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
            )

        self.actor = LinearActorHead(self._hidden_size, action_space.n)
        self.critic = LinearCriticHead(self._hidden_size)

        if self.embed_coordinates:
            self.coordinate_embedding = nn.Linear(
                coordinate_dims, coordinate_embedding_dim
            )

        self.object_type_embedding = nn.Embedding(
            num_embeddings=self._n_object_types,
            embedding_dim=object_type_embedding_dim,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_target_coordinates_encoding(self, observations):
        if self.embed_coordinates:
            return self.coordinate_embedding(
                observations[self.goal_sensor_uuid].to(torch.float32)
            )
        else:
            return observations[self.goal_sensor_uuid].to(torch.float32)

    def get_object_type_encoding(
            self, observations: Dict[str, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """Get the object type encoding from input batched observations."""
        # noinspection PyTypeChecker
        return self.object_type_embedding(  # type:ignore
            observations[self.object_sensor_uuid].to(torch.int64)
        )

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size

    def _recurrent_memory_specification(self):
        return dict(
            rnn=(
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            )
        )

    def forward(  # type:ignore
            self,
            observations: ObservationType,
            memory: Memory,
            prev_actions: torch.Tensor,
            masks: torch.FloatTensor,
            current_actions: torch.FloatTensor = None,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        target_encoding = self.get_target_coordinates_encoding(observations)
        object_encoding = self.get_object_type_encoding(
            cast(Dict[str, torch.FloatTensor], observations)
        )
        x: Union[torch.Tensor, List[torch.Tensor]]
        x = [target_encoding, object_encoding]

        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            if self.sensor_fusion:
                perception_embed = self.sensor_fuser(perception_embed)
            x = [perception_embed] + x

        x = torch.cat(x, dim=-1)
        x, rnn_hidden_states = self.state_encoder(x, memory.tensor("rnn"), masks)

        ac_output = ActorCriticOutput(
            distributions=self.actor(x), values=self.critic(x), extras={}
        )

        return ac_output, memory.set_tensor("rnn", rnn_hidden_states)


class PlacementKeyPointsNPMActorCriticSimpleConvRNN(ActorCriticModel[CategoricalDistr]):
    def __init__(
            self,
            action_space: gym.spaces.Discrete,
            observation_space: SpaceDict,
            goal_sensor_uuid: str,
            object_sensor_uuid: str,
            obstacle_keypoints_sensor_uuid: str,
            hidden_size=512,
            embed_coordinates=False,
            coordinate_embedding_dim=8,
            coordinate_dims=2,
            object_type_embedding_dim=8,
            obstacle_type_embedding_dim=8,
            obstacle_state_hidden_dim=16,
            num_obstacle_types=20,
            num_rnn_layers=1,
            rnn_type="GRU",
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)

        self.goal_sensor_uuid = goal_sensor_uuid
        self.object_sensor_uuid = object_sensor_uuid
        self._hidden_size = hidden_size
        self.embed_coordinates = embed_coordinates
        if self.embed_coordinates:
            self.coorinate_embedding_size = coordinate_embedding_dim
        else:
            self.coorinate_embedding_size = coordinate_dims
        self._n_object_types = self.observation_space.spaces[self.object_sensor_uuid].n
        self.object_type_embedding_size = object_type_embedding_dim
        self.obstacle_keypoints_sensor_uuid = obstacle_keypoints_sensor_uuid
        self.obstacle_type_embedding_size = obstacle_type_embedding_dim
        self.obstacle_state_hidden_dim = obstacle_state_hidden_dim

        self.sensor_fusion = False
        if "rgb" in observation_space.spaces and "depth" in observation_space.spaces:
            self.sensor_fuser = nn.Linear(hidden_size * 2, hidden_size)
            self.sensor_fusion = True

        self.visual_encoder = SimpleCNN(observation_space, hidden_size)

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self.recurrent_hidden_state_size)
            + self.object_type_embedding_size + self.coorinate_embedding_size + obstacle_state_hidden_dim * action_space.n,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
            )

        self.actor = LinearActorHead(self._hidden_size, action_space.n)
        self.critic = LinearCriticHead(self._hidden_size)

        if self.embed_coordinates:
            self.coordinate_embedding = nn.Linear(
                coordinate_dims, coordinate_embedding_dim
            )

        self.object_type_embedding = nn.Embedding(
            num_embeddings=self._n_object_types,
            embedding_dim=object_type_embedding_dim,
        )

        # Action embedding
        self.action_embedding = nn.Embedding(
            num_embeddings=action_space.n, embedding_dim=self.obstacle_state_hidden_dim
        )
        self.num_actions = self.action_space.n

        # Object hidden state encoding
        self.meta_embedding = nn.Embedding(
            num_embeddings=num_obstacle_types, embedding_dim=self.obstacle_state_hidden_dim
        )
        self.rotation_encoding = nn.Sequential(
            nn.Linear(24, self.obstacle_state_hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim // 2, self.obstacle_state_hidden_dim),
        )
        self.state_encoding = nn.Sequential(
            nn.Linear(3, self.obstacle_state_hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim // 2, self.obstacle_state_hidden_dim),
        )

        # NPM
        self.NPM = nn.Sequential(
            nn.Linear(self.obstacle_state_hidden_dim * 3, self.obstacle_state_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim, self.obstacle_state_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim, 12)
        )
        self.NPM[4].weight.data.zero_()
        self.NPM[4].bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float))

        # NPM attention
        self.NPM_atten = nn.Sequential(
            nn.Linear(self.obstacle_state_hidden_dim * 3, self.obstacle_state_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim, 1)
        )

        # NPM Summary
        self.NPM_summary = nn.Sequential(
            nn.Linear(self.obstacle_state_hidden_dim * 4, self.obstacle_state_hidden_dim * 3),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim * 3, self.obstacle_state_hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim * 2, self.obstacle_state_hidden_dim),
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_target_coordinates_encoding(self, observations):
        if self.embed_coordinates:
            return self.coordinate_embedding(
                observations[self.goal_sensor_uuid].to(torch.float32)
            )
        else:
            return observations[self.goal_sensor_uuid].to(torch.float32)

    def get_object_type_encoding(
        self, observations: Dict[str, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """Get the object type encoding from input batched observations."""
        # noinspection PyTypeChecker
        return self.object_type_embedding(  # type:ignore
            observations[self.object_sensor_uuid].to(torch.int64)
        )

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size

    def _recurrent_memory_specification(self):
        return dict(
            rnn=(
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            )
        )

    def forward(  # type:ignore
            self,
            observations: ObservationType,
            memory: Memory,
            prev_actions: torch.Tensor,
            masks: torch.FloatTensor,
            current_actions: torch.FloatTensor = None,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        target_encoding = self.get_target_coordinates_encoding(observations)
        object_encoding = self.get_object_type_encoding(
            cast(Dict[str, torch.FloatTensor], observations)
        )
        x: Union[torch.Tensor, List[torch.Tensor]]
        x = [target_encoding, object_encoding]

        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            if self.sensor_fusion:
                perception_embed = self.sensor_fuser(perception_embed)
            x = [perception_embed] + x

        nb, ng, no, np, nd = observations[self.obstacle_keypoints_sensor_uuid].shape
        nh = self.obstacle_state_hidden_dim

        keypoints = observations[self.obstacle_keypoints_sensor_uuid].view(nb, ng, no, np, nd)
        obstacles_index = torch.arange(0, no).to(target_encoding.device).long()
        obstacles_meta_hidden = self.meta_embedding(obstacles_index)
        obstacles_rot_hidden = self.rotation_encoding(keypoints.view(nb, ng, no, np*nd))
        obstacles_state_hidden = self.state_encoding(keypoints.mean(3))

        na = self.num_actions
        actions_index = torch.arange(0, na).to(target_encoding.device).long()
        a_feature = self.action_embedding(actions_index).view(-1, na, nh)

        keypoints = keypoints.view(nb, ng, no, 1, np, nd).repeat(1, 1, 1, na, 1, 1)
        keypoints_homo = torch.cat((keypoints, torch.ones(nb, ng, no, na, np, 1).to(target_encoding.device)), 5)
        obstacles_meta_hidden = obstacles_meta_hidden.view(1, 1, no, 1, nh).repeat(nb, ng, 1, na, 1)
        obstacles_rot_hidden = obstacles_rot_hidden.view(nb, ng, no, 1, nh).repeat(1, 1, 1, na, 1)
        obstacles_state_hidden = obstacles_state_hidden.view(nb, ng, no, 1, nh).repeat(1, 1, 1, na, 1)
        a_feature = a_feature.view(1, 1, 1, na, nh).repeat(nb, ng, no, 1, 1)

        hidden_feature = torch.cat((obstacles_rot_hidden, obstacles_meta_hidden, a_feature), dim=4)
        NPM_hidden = self.NPM(hidden_feature)
        NPM_hidden = NPM_hidden
        M = NPM_hidden.view(nb, ng, no, na, 3, 4)
        """
        M_test = M.clone()
        M_test[:, :, :, 7] = torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]).to(M.device)
        out, rnn_hidden_states, new_keypoints = self.get_action(M, keypoints_homo, obstacles_rot_hidden,
                                                                obstacles_meta_hidden, a_feature, obstacles_state_hidden,
                                                                nb, ng, x, memory, masks, keypoints)
        out_test, _, new_keypoints_test = self.get_action(M_test, keypoints_homo, obstacles_rot_hidden,
                                                                obstacles_meta_hidden, a_feature, obstacles_state_hidden,
                                                                nb, ng, x, memory, masks, keypoints, True)
        """
        new_keypoints = torch.matmul(M, keypoints_homo.transpose(4, 5)).transpose(4, 5)
        new_obstacles_state_hidden = self.state_encoding(new_keypoints.mean(4))

        atten_feature = torch.cat((obstacles_rot_hidden, obstacles_meta_hidden, a_feature), dim=4)
        hidden_feature = torch.cat((obstacles_meta_hidden, obstacles_state_hidden, new_obstacles_state_hidden,
                                    a_feature), dim=4)
        NPM_atten_score = self.NPM_atten(atten_feature)
        NPM_atten_prob = nn.functional.softmax(NPM_atten_score, 2)
        NPM_atten_hidden = (hidden_feature * NPM_atten_prob).sum(2)
        NPM_atten_hidden = self.NPM_summary(NPM_atten_hidden)
        NPM_atten_hidden = NPM_atten_hidden.view(nb, ng, -1)
        x.append(NPM_atten_hidden)

        x = torch.cat(x, dim=-1)
        x, rnn_hidden_states = self.state_encoder(x, memory.tensor("rnn"), masks)

        out = ActorCriticOutput(
            distributions=self.actor(x), values=self.critic(x), extras={}
        )

        if not isinstance(current_actions, type(None)):
            new_keypoints = new_keypoints.view(nb * ng, no, na, 8, 3).transpose(1, 2)
            current_actions = current_actions.view(nb * ng, 1, 1).squeeze()
            NPM_out = new_keypoints[torch.arange(nb * ng), current_actions].reshape(nb, ng, no, 8, 3)
            out = {"ac_output": out, "npm_output": NPM_out}

        return out, memory.set_tensor("rnn", rnn_hidden_states)

    def get_action(self, M, keypoints_homo, obstacles_rot_hidden, obstacles_meta_hidden, a_feature, obstacles_state_hidden,
                   nb, ng, x, memory, masks, keypoints, second=False):
        new_keypoints = torch.matmul(M, keypoints_homo.transpose(4, 5)).transpose(4, 5)
        if second:
            new_keypoints[:,:,:,7] = keypoints[:,:,:,7].clone()
        new_obstacles_state_hidden = self.state_encoding(new_keypoints.mean(4))

        atten_feature = torch.cat((obstacles_rot_hidden, obstacles_meta_hidden, a_feature), dim=4)
        hidden_feature = torch.cat((obstacles_meta_hidden, obstacles_state_hidden, new_obstacles_state_hidden,
                                    a_feature), dim=4)
        NPM_atten_score = self.NPM_atten(atten_feature)
        NPM_atten_prob = nn.functional.softmax(NPM_atten_score, 2)
        NPM_atten_hidden = (hidden_feature * NPM_atten_prob).sum(2)
        NPM_atten_hidden = self.NPM_summary(NPM_atten_hidden)
        NPM_atten_hidden = NPM_atten_hidden.view(nb, ng, -1)
        if second:
            x[-1] = NPM_atten_hidden
        else:
            x.append(NPM_atten_hidden)

        y = torch.cat(x, dim=-1)
        y, rnn_hidden_states = self.state_encoder(y, memory.tensor("rnn"), masks)

        out = ActorCriticOutput(
            distributions=self.actor(y), values=self.critic(y), extras={}
        )
        return out, rnn_hidden_states, new_keypoints


class PlacementKeyPointsVisualNPMActorCriticSimpleConvRNN(ActorCriticModel[CategoricalDistr]):
    def __init__(
            self,
            action_space: gym.spaces.Discrete,
            observation_space: SpaceDict,
            goal_sensor_uuid: str,
            object_sensor_uuid: str,
            obstacle_keypoints_sensor_uuid: str,
            hidden_size=512,
            embed_coordinates=False,
            coordinate_embedding_dim=8,
            coordinate_dims=2,
            object_type_embedding_dim=8,
            obstacle_type_embedding_dim=8,
            obstacle_state_hidden_dim=16,
            num_obstacle_types=20,
            num_rnn_layers=1,
            rnn_type="GRU",
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)

        self.goal_sensor_uuid = goal_sensor_uuid
        self.object_sensor_uuid = object_sensor_uuid
        self._hidden_size = hidden_size
        self.embed_coordinates = embed_coordinates
        if self.embed_coordinates:
            self.coorinate_embedding_size = coordinate_embedding_dim
        else:
            self.coorinate_embedding_size = coordinate_dims
        self._n_object_types = self.observation_space.spaces[self.object_sensor_uuid].n
        self.object_type_embedding_size = object_type_embedding_dim
        self.obstacle_keypoints_sensor_uuid = obstacle_keypoints_sensor_uuid
        self.obstacle_type_embedding_size = obstacle_type_embedding_dim
        self.obstacle_state_hidden_dim = obstacle_state_hidden_dim

        self.sensor_fusion = False
        if "rgb" in observation_space.spaces and "depth" in observation_space.spaces:
            self.sensor_fuser = nn.Linear(hidden_size * 2, hidden_size)
            self.sensor_fusion = True

        self.visual_encoder = SimpleCNN(observation_space, hidden_size)

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self.recurrent_hidden_state_size)
            + self.object_type_embedding_size + self.coorinate_embedding_size + obstacle_state_hidden_dim * action_space.n,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
            )

        self.actor = LinearActorHead(self._hidden_size, action_space.n)
        self.critic = LinearCriticHead(self._hidden_size)

        if self.embed_coordinates:
            self.coordinate_embedding = nn.Linear(
                coordinate_dims, coordinate_embedding_dim
            )

        self.object_type_embedding = nn.Embedding(
            num_embeddings=self._n_object_types,
            embedding_dim=object_type_embedding_dim,
        )

        # Action embedding
        self.action_embedding = nn.Embedding(
            num_embeddings=action_space.n, embedding_dim=self.obstacle_state_hidden_dim
        )
        self.num_actions = self.action_space.n

        # Object hidden state encoding
        self.meta_embedding = nn.Embedding(
            num_embeddings=num_obstacle_types, embedding_dim=self.obstacle_state_hidden_dim
        )
        self.rotation_encoding = nn.Sequential(
            nn.Linear(24, self.obstacle_state_hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim // 2, self.obstacle_state_hidden_dim),
        )
        self.state_encoding = nn.Sequential(
            nn.Linear(3, self.obstacle_state_hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim // 2, self.obstacle_state_hidden_dim),
        )

        # NPM
        self.NPM = nn.Sequential(
            nn.Linear(hidden_size + self.obstacle_state_hidden_dim * 3, self.obstacle_state_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim, self.obstacle_state_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim, 12)
        )
        self.NPM[4].weight.data.zero_()
        self.NPM[4].bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float))

        # NPM attention
        self.NPM_atten = nn.Sequential(
            nn.Linear(self.obstacle_state_hidden_dim * 3, self.obstacle_state_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim, 1)
        )

        # NPM Summary
        self.NPM_summary = nn.Sequential(
            nn.Linear(self.obstacle_state_hidden_dim * 4, self.obstacle_state_hidden_dim * 3),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim * 3, self.obstacle_state_hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim * 2, self.obstacle_state_hidden_dim),
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_target_coordinates_encoding(self, observations):
        if self.embed_coordinates:
            return self.coordinate_embedding(
                observations[self.goal_sensor_uuid].to(torch.float32)
            )
        else:
            return observations[self.goal_sensor_uuid].to(torch.float32)

    def get_object_type_encoding(
        self, observations: Dict[str, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """Get the object type encoding from input batched observations."""
        # noinspection PyTypeChecker
        return self.object_type_embedding(  # type:ignore
            observations[self.object_sensor_uuid].to(torch.int64)
        )

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size

    def _recurrent_memory_specification(self):
        return dict(
            rnn=(
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            )
        )

    def forward(  # type:ignore
            self,
            observations: ObservationType,
            memory: Memory,
            prev_actions: torch.Tensor,
            masks: torch.FloatTensor,
            current_actions: torch.FloatTensor = None,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        target_encoding = self.get_target_coordinates_encoding(observations)
        object_encoding = self.get_object_type_encoding(
            cast(Dict[str, torch.FloatTensor], observations)
        )
        x: Union[torch.Tensor, List[torch.Tensor]]
        x = [target_encoding, object_encoding]

        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            if self.sensor_fusion:
                perception_embed = self.sensor_fuser(perception_embed)
            x = [perception_embed] + x

        nb, ng, no, np, nd = observations[self.obstacle_keypoints_sensor_uuid].shape
        nh = self.obstacle_state_hidden_dim

        keypoints = observations[self.obstacle_keypoints_sensor_uuid].view(nb, ng, no, np, nd)
        obstacles_index = torch.arange(0, no).to(target_encoding.device).long()
        obstacles_meta_hidden = self.meta_embedding(obstacles_index)
        obstacles_rot_hidden = self.rotation_encoding(keypoints.view(nb, ng, no, np*nd))
        obstacles_state_hidden = self.state_encoding(keypoints.mean(3))

        na = self.num_actions
        actions_index = torch.arange(0, na).to(target_encoding.device).long()
        a_feature = self.action_embedding(actions_index).view(-1, na, nh)

        keypoints = keypoints.view(nb, ng, no, 1, np, nd).repeat(1, 1, 1, na, 1, 1)
        keypoints_homo = torch.cat((keypoints, torch.ones(nb, ng, no, na, np, 1).to(target_encoding.device)), 5)
        obstacles_meta_hidden = obstacles_meta_hidden.view(1, 1, no, 1, nh).repeat(nb, ng, 1, na, 1)
        obstacles_rot_hidden = obstacles_rot_hidden.view(nb, ng, no, 1, nh).repeat(1, 1, 1, na, 1)
        obstacles_state_hidden = obstacles_state_hidden.view(nb, ng, no, 1, nh).repeat(1, 1, 1, na, 1)
        a_feature = a_feature.view(1, 1, 1, na, nh).repeat(nb, ng, no, 1, 1)
        perception_embed_hidden = perception_embed.view(nb, ng, 1, 1, self._hidden_size).repeat(1, 1, no, na, 1)

        hidden_feature = torch.cat((perception_embed_hidden, obstacles_rot_hidden, obstacles_meta_hidden, a_feature), dim=4)
        NPM_hidden = self.NPM(hidden_feature)
        NPM_hidden = NPM_hidden
        M = NPM_hidden.view(nb, ng, no, na, 3, 4)
        M_test = M.clone()
        M_test[:, :, :, 7] = torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]).to(M.device)
        M_test[:, :, :, 5] = torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]).to(M.device)
        out, rnn_hidden_states, new_keypoints = self.get_action(M, keypoints_homo, obstacles_rot_hidden,
                                                                obstacles_meta_hidden, a_feature, obstacles_state_hidden,
                                                                nb, ng, x, memory, masks, keypoints)
        out_test, _, new_keypoints_test = self.get_action(M_test, keypoints_homo, obstacles_rot_hidden,
                                                          obstacles_meta_hidden, a_feature, obstacles_state_hidden,
                                                          nb, ng, x, memory, masks, keypoints, True)
        """
        new_keypoints = torch.matmul(M, keypoints_homo.transpose(4, 5)).transpose(4, 5)
        new_obstacles_state_hidden = self.state_encoding(new_keypoints.mean(4))

        atten_feature = torch.cat((obstacles_rot_hidden, obstacles_meta_hidden, a_feature), dim=4)
        hidden_feature = torch.cat((obstacles_meta_hidden, obstacles_state_hidden, new_obstacles_state_hidden,
                                    a_feature), dim=4)
        NPM_atten_score = self.NPM_atten(atten_feature)
        NPM_atten_prob = nn.functional.softmax(NPM_atten_score, 2)
        NPM_atten_hidden = (hidden_feature * NPM_atten_prob).sum(2)
        NPM_atten_hidden = self.NPM_summary(NPM_atten_hidden)
        NPM_atten_hidden = NPM_atten_hidden.view(nb, ng, -1)
        x.append(NPM_atten_hidden)

        x = torch.cat(x, dim=-1)
        x, rnn_hidden_states = self.state_encoder(x, memory.tensor("rnn"), masks)

        out = ActorCriticOutput(
            distributions=self.actor(x), values=self.critic(x), extras={}
        )
        """

        if not isinstance(current_actions, type(None)):
            new_keypoints = new_keypoints.view(nb * ng, no, na, 8, 3).transpose(1, 2)
            current_actions = current_actions.view(nb * ng, 1, 1).squeeze()
            NPM_out = new_keypoints[torch.arange(nb * ng), current_actions].reshape(nb, ng, no, 8, 3)
            out = {"ac_output": out, "npm_output": NPM_out}

        return out, memory.set_tensor("rnn", rnn_hidden_states)

    def get_action(self, M, keypoints_homo, obstacles_rot_hidden, obstacles_meta_hidden, a_feature, obstacles_state_hidden,
                   nb, ng, x, memory, masks, keypoints, second=False):
        new_keypoints = torch.matmul(M, keypoints_homo.transpose(4, 5)).transpose(4, 5)
        if second:
            new_keypoints[:,:,:,7] = keypoints[:,:,:,7].clone()
            new_keypoints[:, :, :, 5] = keypoints[:, :, :, 5].clone()
        new_obstacles_state_hidden = self.state_encoding(new_keypoints.mean(4))

        atten_feature = torch.cat((obstacles_rot_hidden, obstacles_meta_hidden, a_feature), dim=4)
        hidden_feature = torch.cat((obstacles_meta_hidden, obstacles_state_hidden, new_obstacles_state_hidden,
                                    a_feature), dim=4)
        NPM_atten_score = self.NPM_atten(atten_feature)
        NPM_atten_prob = nn.functional.softmax(NPM_atten_score, 2)
        NPM_atten_hidden = (hidden_feature * NPM_atten_prob).sum(2)
        NPM_atten_hidden = self.NPM_summary(NPM_atten_hidden)
        NPM_atten_hidden = NPM_atten_hidden.view(nb, ng, -1)
        if second:
            x[-1] = NPM_atten_hidden
        else:
            x.append(NPM_atten_hidden)

        y = torch.cat(x, dim=-1)
        y, rnn_hidden_states = self.state_encoder(y, memory.tensor("rnn"), masks)

        out = ActorCriticOutput(
            distributions=self.actor(y), values=self.critic(y), extras={}
        )
        return out, rnn_hidden_states, new_keypoints


class PlacementCPCKeyPointsVisualNPMActorCriticSimpleConvRNN(ActorCriticModel[CategoricalDistr]):
    def __init__(
            self,
            action_space: gym.spaces.Discrete,
            observation_space: SpaceDict,
            goal_sensor_uuid: str,
            object_sensor_uuid: str,
            obstacle_keypoints_sensor_uuid: str,
            hidden_size=512,
            embed_coordinates=False,
            coordinate_embedding_dim=8,
            coordinate_dims=2,
            object_type_embedding_dim=8,
            obstacle_type_embedding_dim=8,
            obstacle_state_hidden_dim=16,
            num_obstacle_types=20,
            num_rnn_layers=1,
            rnn_type="GRU",
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)

        self.goal_sensor_uuid = goal_sensor_uuid
        self.object_sensor_uuid = object_sensor_uuid
        self._hidden_size = hidden_size
        self.embed_coordinates = embed_coordinates
        if self.embed_coordinates:
            self.coorinate_embedding_size = coordinate_embedding_dim
        else:
            self.coorinate_embedding_size = coordinate_dims
        self._n_object_types = self.observation_space.spaces[self.object_sensor_uuid].n
        self.object_type_embedding_size = object_type_embedding_dim
        self.obstacle_keypoints_sensor_uuid = obstacle_keypoints_sensor_uuid
        self.obstacle_type_embedding_size = obstacle_type_embedding_dim
        self.obstacle_state_hidden_dim = obstacle_state_hidden_dim

        self.sensor_fusion = False
        if "rgb" in observation_space.spaces and "depth" in observation_space.spaces:
            self.sensor_fuser = nn.Linear(hidden_size * 2, hidden_size)
            self.sensor_fusion = True
        self.sensor_fuser_2 = nn.Linear(self.object_type_embedding_size +
                                        self.coorinate_embedding_size +
                                        hidden_size +
                                        obstacle_state_hidden_dim * action_space.n,
                                        hidden_size)

        self.visual_encoder = SimpleCNN(observation_space, hidden_size)

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self.recurrent_hidden_state_size) + 1,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
            )

        self.actor = LinearActorHead(self._hidden_size, action_space.n)
        self.critic = LinearCriticHead(self._hidden_size)

        if self.embed_coordinates:
            self.coordinate_embedding = nn.Linear(
                coordinate_dims, coordinate_embedding_dim
            )

        self.object_type_embedding = nn.Embedding(
            num_embeddings=self._n_object_types,
            embedding_dim=object_type_embedding_dim,
        )

        # Action embedding
        self.action_embedding = nn.Embedding(
            num_embeddings=action_space.n, embedding_dim=self.obstacle_state_hidden_dim
        )
        self.num_actions = self.action_space.n

        # Object hidden state encoding
        self.meta_embedding = nn.Embedding(
            num_embeddings=num_obstacle_types, embedding_dim=self.obstacle_state_hidden_dim
        )
        self.rotation_encoding = nn.Sequential(
            nn.Linear(24, self.obstacle_state_hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim // 2, self.obstacle_state_hidden_dim),
        )
        self.state_encoding = nn.Sequential(
            nn.Linear(3, self.obstacle_state_hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim // 2, self.obstacle_state_hidden_dim),
        )

        # NPM
        self.NPM = nn.Sequential(
            nn.Linear(hidden_size + self.obstacle_state_hidden_dim * 3, self.obstacle_state_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim, self.obstacle_state_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim, 12)
        )
        self.NPM[4].weight.data.zero_()
        self.NPM[4].bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float))

        # NPM attention
        self.NPM_atten = nn.Sequential(
            nn.Linear(self.obstacle_state_hidden_dim * 3, self.obstacle_state_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim, 1)
        )

        # NPM Summary
        self.NPM_summary = nn.Sequential(
            nn.Linear(self.obstacle_state_hidden_dim * 4, self.obstacle_state_hidden_dim * 3),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim * 3, self.obstacle_state_hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim * 2, self.obstacle_state_hidden_dim),
        )

        self.state_encoder_predict = RNNStateEncoder(
            1,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
        )

        self.classifier = nn.Sequential(
            nn.Linear(self._hidden_size * 2, self._hidden_size),
            nn.ReLU(),
            nn.Linear(self._hidden_size, 1),
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_target_coordinates_encoding(self, observations):
        if self.embed_coordinates:
            return self.coordinate_embedding(
                observations[self.goal_sensor_uuid].to(torch.float32)
            )
        else:
            return observations[self.goal_sensor_uuid].to(torch.float32)

    def get_object_type_encoding(
            self, observations: Dict[str, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """Get the object type encoding from input batched observations."""
        # noinspection PyTypeChecker
        return self.object_type_embedding(  # type:ignore
            observations[self.object_sensor_uuid].to(torch.int64)
        )

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size

    def _recurrent_memory_specification(self):
        return dict(
            rnn=(
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            )
        )

    def forward(  # type:ignore
            self,
            observations: ObservationType,
            memory: Memory,
            prev_actions: torch.Tensor,
            masks: torch.FloatTensor,
            current_actions: torch.FloatTensor = None,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        target_encoding = self.get_target_coordinates_encoding(observations)
        object_encoding = self.get_object_type_encoding(
            cast(Dict[str, torch.FloatTensor], observations)
        )
        x: Union[torch.Tensor, List[torch.Tensor]]
        x = [target_encoding, object_encoding]

        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            if self.sensor_fusion:
                perception_embed = self.sensor_fuser(perception_embed)
            x = [perception_embed] + x

        nb, ng, no, np, nd = observations[self.obstacle_keypoints_sensor_uuid].shape
        nh = self.obstacle_state_hidden_dim

        keypoints = observations[self.obstacle_keypoints_sensor_uuid].view(nb, ng, no, np, nd)
        obstacles_index = torch.arange(0, no).to(target_encoding.device).long()
        obstacles_meta_hidden = self.meta_embedding(obstacles_index)
        obstacles_rot_hidden = self.rotation_encoding(keypoints.view(nb, ng, no, np*nd))
        obstacles_state_hidden = self.state_encoding(keypoints.mean(3))

        na = self.num_actions
        actions_index = torch.arange(0, na).to(target_encoding.device).long()
        a_feature = self.action_embedding(actions_index).view(-1, na, nh)

        keypoints = keypoints.view(nb, ng, no, 1, np, nd).repeat(1, 1, 1, na, 1, 1)
        keypoints_homo = torch.cat((keypoints, torch.ones(nb, ng, no, na, np, 1).to(target_encoding.device)), 5)
        obstacles_meta_hidden = obstacles_meta_hidden.view(1, 1, no, 1, nh).repeat(nb, ng, 1, na, 1)
        obstacles_rot_hidden = obstacles_rot_hidden.view(nb, ng, no, 1, nh).repeat(1, 1, 1, na, 1)
        obstacles_state_hidden = obstacles_state_hidden.view(nb, ng, no, 1, nh).repeat(1, 1, 1, na, 1)
        a_feature = a_feature.view(1, 1, 1, na, nh).repeat(nb, ng, no, 1, 1)
        perception_embed_hidden = perception_embed.view(nb, ng, 1, 1, self._hidden_size).repeat(1, 1, no, na, 1)

        hidden_feature = torch.cat((perception_embed_hidden, obstacles_rot_hidden, obstacles_meta_hidden, a_feature), dim=4)
        NPM_hidden = self.NPM(hidden_feature)
        NPM_hidden = NPM_hidden
        M = NPM_hidden.view(nb, ng, no, na, 3, 4)
        M_test = M.clone()
        M_test[:, :, :, 7] = torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]).to(M.device)
        M_test[:, :, :, 5] = torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]).to(M.device)

        new_keypoints = torch.matmul(M, keypoints_homo.transpose(4, 5)).transpose(4, 5)
        new_obstacles_state_hidden = self.state_encoding(new_keypoints.mean(4))

        atten_feature = torch.cat((obstacles_rot_hidden, obstacles_meta_hidden, a_feature), dim=4)
        hidden_feature = torch.cat((obstacles_meta_hidden, obstacles_state_hidden, new_obstacles_state_hidden,
                                    a_feature), dim=4)
        NPM_atten_score = self.NPM_atten(atten_feature)
        NPM_atten_prob = nn.functional.softmax(NPM_atten_score, 2)
        NPM_atten_hidden = (hidden_feature * NPM_atten_prob).sum(2)
        NPM_atten_hidden = self.NPM_summary(NPM_atten_hidden)
        NPM_atten_hidden = NPM_atten_hidden.view(nb, ng, -1)
        x.append(NPM_atten_hidden)

        x_obs = self.sensor_fuser_2(torch.cat(x, dim=-1))
        x = [x_obs] + [prev_actions.squeeze(-1)]
        x = torch.cat(x, dim=-1)
        x, rnn_hidden_states = self.state_encoder(x, memory.tensor("rnn"), masks)

        out = ActorCriticOutput(
            distributions=self.actor(x), values=self.critic(x), extras={}
        )

        if not isinstance(current_actions, type(None)):
            nt, ng = current_actions.shape[0], current_actions.shape[1]
            x_action = current_actions.squeeze(-1).view(1, nt * ng, 1).float()
            b = x.view(1, nt * ng, -1)
            m = masks.view(1, nt * ng, 1, 1)
            x_pred, _ = self.state_encoder_predict(x_action, b, m)
            x_pred = x_pred.view(nt, ng, -1)[:-1]

            x_obs_positive = x_obs[1:]
            concat_positive = torch.cat([x_pred, x_obs_positive], dim=-1)
            logit_positive = self.classifier(concat_positive)

            idx = torch.randperm(nt).to(x_obs.device)
            x_obs_negative = x_obs[idx, :, :][1:]
            concat_negative = torch.cat([x_pred, x_obs_negative], dim=-1)
            logit_negative = self.classifier(concat_negative)

            new_keypoints = new_keypoints.view(nb * ng, no, na, 8, 3).transpose(1, 2)
            current_actions = current_actions.view(nb * ng, 1, 1).squeeze()
            NPM_out = new_keypoints[torch.arange(nb * ng), current_actions].reshape(nb, ng, no, 8, 3)
            out = {"ac_output": out, "npm_output": NPM_out,
                   "logit_positive": logit_positive, "logit_negative": logit_negative}

        return out, memory.set_tensor("rnn", rnn_hidden_states)


class PlacementPBLKeyPointsVisualNPMActorCriticSimpleConvRNN(ActorCriticModel[CategoricalDistr]):
    def __init__(
            self,
            action_space: gym.spaces.Discrete,
            observation_space: SpaceDict,
            goal_sensor_uuid: str,
            object_sensor_uuid: str,
            obstacle_keypoints_sensor_uuid: str,
            hidden_size=512,
            embed_coordinates=False,
            coordinate_embedding_dim=8,
            coordinate_dims=2,
            object_type_embedding_dim=8,
            obstacle_type_embedding_dim=8,
            obstacle_state_hidden_dim=16,
            num_obstacle_types=20,
            num_rnn_layers=1,
            rnn_type="GRU",
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)

        self.goal_sensor_uuid = goal_sensor_uuid
        self.object_sensor_uuid = object_sensor_uuid
        self._hidden_size = hidden_size
        self.embed_coordinates = embed_coordinates
        if self.embed_coordinates:
            self.coorinate_embedding_size = coordinate_embedding_dim
        else:
            self.coorinate_embedding_size = coordinate_dims
        self._n_object_types = self.observation_space.spaces[self.object_sensor_uuid].n
        self.object_type_embedding_size = object_type_embedding_dim
        self.obstacle_keypoints_sensor_uuid = obstacle_keypoints_sensor_uuid
        self.obstacle_type_embedding_size = obstacle_type_embedding_dim
        self.obstacle_state_hidden_dim = obstacle_state_hidden_dim

        self.sensor_fusion = False
        if "rgb" in observation_space.spaces and "depth" in observation_space.spaces:
            self.sensor_fuser = nn.Linear(hidden_size * 2, hidden_size)
            self.sensor_fusion = True
        self.sensor_fuser_2 = nn.Linear(self.object_type_embedding_size +
                                        self.coorinate_embedding_size +
                                        hidden_size +
                                        obstacle_state_hidden_dim * action_space.n,
                                        hidden_size)

        self.visual_encoder = SimpleCNN(observation_space, hidden_size)

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self.recurrent_hidden_state_size) + 1,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
            )

        self.actor = LinearActorHead(self._hidden_size, action_space.n)
        self.critic = LinearCriticHead(self._hidden_size)

        if self.embed_coordinates:
            self.coordinate_embedding = nn.Linear(
                coordinate_dims, coordinate_embedding_dim
            )

        self.object_type_embedding = nn.Embedding(
            num_embeddings=self._n_object_types,
            embedding_dim=object_type_embedding_dim,
        )

        # Action embedding
        self.action_embedding = nn.Embedding(
            num_embeddings=action_space.n, embedding_dim=self.obstacle_state_hidden_dim
        )
        self.num_actions = self.action_space.n

        # Object hidden state encoding
        self.meta_embedding = nn.Embedding(
            num_embeddings=num_obstacle_types, embedding_dim=self.obstacle_state_hidden_dim
        )
        self.rotation_encoding = nn.Sequential(
            nn.Linear(24, self.obstacle_state_hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim // 2, self.obstacle_state_hidden_dim),
        )
        self.state_encoding = nn.Sequential(
            nn.Linear(3, self.obstacle_state_hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim // 2, self.obstacle_state_hidden_dim),
        )

        # NPM
        self.NPM = nn.Sequential(
            nn.Linear(hidden_size + self.obstacle_state_hidden_dim * 3, self.obstacle_state_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim, self.obstacle_state_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim, 12)
        )
        self.NPM[4].weight.data.zero_()
        self.NPM[4].bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float))

        # NPM attention
        self.NPM_atten = nn.Sequential(
            nn.Linear(self.obstacle_state_hidden_dim * 3, self.obstacle_state_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim, 1)
        )

        # NPM Summary
        self.NPM_summary = nn.Sequential(
            nn.Linear(self.obstacle_state_hidden_dim * 4, self.obstacle_state_hidden_dim * 3),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim * 3, self.obstacle_state_hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim * 2, self.obstacle_state_hidden_dim),
        )

        self.state_encoder_predict = RNNStateEncoder(
            1,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
        )

        self.g_forward = nn.Sequential(
            nn.Linear(self._hidden_size, self._hidden_size),
            nn.ReLU(),
            nn.Linear(self._hidden_size, self._hidden_size),
        )

        self.g_reverse = nn.Sequential(
            nn.Linear(self._hidden_size, self._hidden_size),
            nn.ReLU(),
            nn.Linear(self._hidden_size, self._hidden_size),
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_target_coordinates_encoding(self, observations):
        if self.embed_coordinates:
            return self.coordinate_embedding(
                observations[self.goal_sensor_uuid].to(torch.float32)
            )
        else:
            return observations[self.goal_sensor_uuid].to(torch.float32)

    def get_object_type_encoding(
            self, observations: Dict[str, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """Get the object type encoding from input batched observations."""
        # noinspection PyTypeChecker
        return self.object_type_embedding(  # type:ignore
            observations[self.object_sensor_uuid].to(torch.int64)
        )

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size

    def _recurrent_memory_specification(self):
        return dict(
            rnn=(
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            )
        )

    def forward(  # type:ignore
            self,
            observations: ObservationType,
            memory: Memory,
            prev_actions: torch.Tensor,
            masks: torch.FloatTensor,
            current_actions: torch.FloatTensor = None,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        target_encoding = self.get_target_coordinates_encoding(observations)
        object_encoding = self.get_object_type_encoding(
            cast(Dict[str, torch.FloatTensor], observations)
        )
        x: Union[torch.Tensor, List[torch.Tensor]]
        x = [target_encoding, object_encoding]

        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            if self.sensor_fusion:
                perception_embed = self.sensor_fuser(perception_embed)
            x = [perception_embed] + x

        nb, ng, no, np, nd = observations[self.obstacle_keypoints_sensor_uuid].shape
        nh = self.obstacle_state_hidden_dim

        keypoints = observations[self.obstacle_keypoints_sensor_uuid].view(nb, ng, no, np, nd)
        obstacles_index = torch.arange(0, no).to(target_encoding.device).long()
        obstacles_meta_hidden = self.meta_embedding(obstacles_index)
        obstacles_rot_hidden = self.rotation_encoding(keypoints.view(nb, ng, no, np*nd))
        obstacles_state_hidden = self.state_encoding(keypoints.mean(3))

        na = self.num_actions
        actions_index = torch.arange(0, na).to(target_encoding.device).long()
        a_feature = self.action_embedding(actions_index).view(-1, na, nh)

        keypoints = keypoints.view(nb, ng, no, 1, np, nd).repeat(1, 1, 1, na, 1, 1)
        keypoints_homo = torch.cat((keypoints, torch.ones(nb, ng, no, na, np, 1).to(target_encoding.device)), 5)
        obstacles_meta_hidden = obstacles_meta_hidden.view(1, 1, no, 1, nh).repeat(nb, ng, 1, na, 1)
        obstacles_rot_hidden = obstacles_rot_hidden.view(nb, ng, no, 1, nh).repeat(1, 1, 1, na, 1)
        obstacles_state_hidden = obstacles_state_hidden.view(nb, ng, no, 1, nh).repeat(1, 1, 1, na, 1)
        a_feature = a_feature.view(1, 1, 1, na, nh).repeat(nb, ng, no, 1, 1)
        perception_embed_hidden = perception_embed.view(nb, ng, 1, 1, self._hidden_size).repeat(1, 1, no, na, 1)

        hidden_feature = torch.cat((perception_embed_hidden, obstacles_rot_hidden, obstacles_meta_hidden, a_feature), dim=4)
        NPM_hidden = self.NPM(hidden_feature)
        NPM_hidden = NPM_hidden
        M = NPM_hidden.view(nb, ng, no, na, 3, 4)
        M_test = M.clone()
        M_test[:, :, :, 7] = torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]).to(M.device)
        M_test[:, :, :, 5] = torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]).to(M.device)

        new_keypoints = torch.matmul(M, keypoints_homo.transpose(4, 5)).transpose(4, 5)
        new_obstacles_state_hidden = self.state_encoding(new_keypoints.mean(4))

        atten_feature = torch.cat((obstacles_rot_hidden, obstacles_meta_hidden, a_feature), dim=4)
        hidden_feature = torch.cat((obstacles_meta_hidden, obstacles_state_hidden, new_obstacles_state_hidden,
                                    a_feature), dim=4)
        NPM_atten_score = self.NPM_atten(atten_feature)
        NPM_atten_prob = nn.functional.softmax(NPM_atten_score, 2)
        NPM_atten_hidden = (hidden_feature * NPM_atten_prob).sum(2)
        NPM_atten_hidden = self.NPM_summary(NPM_atten_hidden)
        NPM_atten_hidden = NPM_atten_hidden.view(nb, ng, -1)
        x.append(NPM_atten_hidden)

        x_obs = self.sensor_fuser_2(torch.cat(x, dim=-1))
        x = [x_obs] + [prev_actions.squeeze(-1)]
        x = torch.cat(x, dim=-1)
        x, rnn_hidden_states = self.state_encoder(x, memory.tensor("rnn"), masks)

        out = ActorCriticOutput(
            distributions=self.actor(x), values=self.critic(x), extras={}
        )

        if not isinstance(current_actions, type(None)):
            nt, ng = current_actions.shape[0], current_actions.shape[1]
            x_action = current_actions.squeeze(-1).view(1, nt * ng, 1).float()
            b = x.view(1, nt * ng, -1)
            m = masks.view(1, nt * ng, 1, 1)
            x_pred, _ = self.state_encoder_predict(x_action, b, m)
            x_pred = x_pred.view(nt, ng, -1)
            x_forward = self.g_forward(x_pred)
            #x_forward = 0.02 * (((x_forward ** 2) - 1) ** 2)
            x_forward = x_forward / (torch.norm(x_forward, dim=-1, keepdim=True) + 10**(-8))
            #x_obs = 0.02 * (((x_obs ** 2) - 1) ** 2)
            x_obs = x_obs / (torch.norm(x_obs, dim=-1, keepdim=True) + 10**(-8))
            x_reverse = self.g_reverse(x_obs)
            b = b.view(nt, ng, -1)

            new_keypoints = new_keypoints.view(nb * ng, no, na, 8, 3).transpose(1, 2)
            current_actions = current_actions.view(nb * ng, 1, 1).squeeze()
            NPM_out = new_keypoints[torch.arange(nb * ng), current_actions].reshape(nb, ng, no, 8, 3)
            out = {"ac_output": out, "npm_output": NPM_out, "x_obs": x_obs,
                   "x_forward": x_forward, "x_reverse": x_reverse, "b": b}

        return out, memory.set_tensor("rnn", rnn_hidden_states)


class PlacementKeyPointsVisualNPMSAActorCriticSimpleConvRNN(ActorCriticModel[CategoricalDistr]):
    def __init__(
            self,
            action_space: gym.spaces.Discrete,
            observation_space: SpaceDict,
            goal_sensor_uuid: str,
            object_sensor_uuid: str,
            obstacle_keypoints_sensor_uuid: str,
            hidden_size=512,
            embed_coordinates=False,
            coordinate_embedding_dim=8,
            coordinate_dims=2,
            object_type_embedding_dim=8,
            obstacle_type_embedding_dim=8,
            obstacle_state_hidden_dim=16,
            num_obstacle_types=20,
            num_rnn_layers=1,
            rnn_type="GRU",
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)

        self.goal_sensor_uuid = goal_sensor_uuid
        self.object_sensor_uuid = object_sensor_uuid
        self._hidden_size = hidden_size
        self.embed_coordinates = embed_coordinates
        if self.embed_coordinates:
            self.coorinate_embedding_size = coordinate_embedding_dim
        else:
            self.coorinate_embedding_size = coordinate_dims
        self._n_object_types = self.observation_space.spaces[self.object_sensor_uuid].n
        self.object_type_embedding_size = object_type_embedding_dim
        self.obstacle_keypoints_sensor_uuid = obstacle_keypoints_sensor_uuid
        self.obstacle_type_embedding_size = obstacle_type_embedding_dim
        self.obstacle_state_hidden_dim = obstacle_state_hidden_dim

        self.sensor_fusion = False
        if "rgb" in observation_space.spaces and "depth" in observation_space.spaces:
            self.sensor_fuser = nn.Linear(hidden_size * 2, hidden_size)
            self.sensor_fusion = True

        self.visual_encoder = SimpleCNN(observation_space, hidden_size)

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self.recurrent_hidden_state_size)
            + self.object_type_embedding_size + self.coorinate_embedding_size + obstacle_state_hidden_dim * action_space.n,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
            )

        self.actor = LinearActorHead(self._hidden_size, action_space.n)
        self.critic = LinearCriticHead(self._hidden_size)

        if self.embed_coordinates:
            self.coordinate_embedding = nn.Linear(
                coordinate_dims, coordinate_embedding_dim
            )

        self.object_type_embedding = nn.Embedding(
            num_embeddings=self._n_object_types,
            embedding_dim=object_type_embedding_dim,
        )

        # Action embedding
        self.action_embedding = nn.Embedding(
            num_embeddings=action_space.n, embedding_dim=self.obstacle_state_hidden_dim
        )
        self.num_actions = self.action_space.n

        # Object hidden state encoding
        self.meta_embedding = nn.Embedding(
            num_embeddings=num_obstacle_types, embedding_dim=self.obstacle_state_hidden_dim
        )
        self.rotation_encoding = nn.Sequential(
            nn.Linear(24, self.obstacle_state_hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim // 2, self.obstacle_state_hidden_dim),
        )
        self.state_encoding = nn.Sequential(
            nn.Linear(3, self.obstacle_state_hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim // 2, self.obstacle_state_hidden_dim),
        )

        # NPM
        self.NPM = nn.Sequential(
            nn.Linear(hidden_size + self.obstacle_state_hidden_dim * 3, self.obstacle_state_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim, self.obstacle_state_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim, 12)
        )
        self.NPM[4].weight.data.zero_()
        self.NPM[4].bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float))

        # NPM attention
        self.NPM_atten_a = SA(obstacle_state_hidden_dim * 5, obstacle_state_hidden_dim * 5)
        self.NPM_atten_b = SA(obstacle_state_hidden_dim * 5, obstacle_state_hidden_dim * 5)

        # NPM Summary
        self.NPM_summary = nn.Sequential(
            nn.Linear(self.obstacle_state_hidden_dim * 5, self.obstacle_state_hidden_dim * 3),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim * 3, self.obstacle_state_hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim * 2, self.obstacle_state_hidden_dim),
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_target_coordinates_encoding(self, observations):
        if self.embed_coordinates:
            return self.coordinate_embedding(
                observations[self.goal_sensor_uuid].to(torch.float32)
            )
        else:
            return observations[self.goal_sensor_uuid].to(torch.float32)

    def get_object_type_encoding(
            self, observations: Dict[str, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """Get the object type encoding from input batched observations."""
        # noinspection PyTypeChecker
        return self.object_type_embedding(  # type:ignore
            observations[self.object_sensor_uuid].to(torch.int64)
        )

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size

    def _recurrent_memory_specification(self):
        return dict(
            rnn=(
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            )
        )

    def forward(  # type:ignore
            self,
            observations: ObservationType,
            memory: Memory,
            prev_actions: torch.Tensor,
            masks: torch.FloatTensor,
            current_actions: torch.FloatTensor = None,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        target_encoding = self.get_target_coordinates_encoding(observations)
        object_encoding = self.get_object_type_encoding(
            cast(Dict[str, torch.FloatTensor], observations)
        )
        x: Union[torch.Tensor, List[torch.Tensor]]
        x = [target_encoding, object_encoding]

        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            if self.sensor_fusion:
                perception_embed = self.sensor_fuser(perception_embed)
            x = [perception_embed] + x

        nb, ng, no, np, nd = observations[self.obstacle_keypoints_sensor_uuid].shape
        nh = self.obstacle_state_hidden_dim

        keypoints = observations[self.obstacle_keypoints_sensor_uuid].view(nb, ng, no, np, nd)
        obstacles_index = torch.arange(0, no).to(target_encoding.device).long()
        obstacles_meta_hidden = self.meta_embedding(obstacles_index)
        obstacles_rot_hidden = self.rotation_encoding(keypoints.view(nb, ng, no, np*nd))
        obstacles_state_hidden = self.state_encoding(keypoints.mean(3))

        na = self.num_actions
        actions_index = torch.arange(0, na).to(target_encoding.device).long()
        a_feature = self.action_embedding(actions_index).view(-1, na, nh)

        keypoints = keypoints.view(nb, ng, no, 1, np, nd).repeat(1, 1, 1, na, 1, 1)
        keypoints_homo = torch.cat((keypoints, torch.ones(nb, ng, no, na, np, 1).to(target_encoding.device)), 5)
        obstacles_meta_hidden = obstacles_meta_hidden.view(1, 1, no, 1, nh).repeat(nb, ng, 1, na, 1)
        obstacles_rot_hidden = obstacles_rot_hidden.view(nb, ng, no, 1, nh).repeat(1, 1, 1, na, 1)
        obstacles_state_hidden = obstacles_state_hidden.view(nb, ng, no, 1, nh).repeat(1, 1, 1, na, 1)
        a_feature = a_feature.view(1, 1, 1, na, nh).repeat(nb, ng, no, 1, 1)
        perception_embed_hidden = perception_embed.view(nb, ng, 1, 1, self._hidden_size).repeat(1, 1, no, na, 1)

        hidden_feature = torch.cat((perception_embed_hidden, obstacles_rot_hidden, obstacles_meta_hidden, a_feature), dim=4)
        NPM_hidden = self.NPM(hidden_feature)
        NPM_hidden = NPM_hidden
        M = NPM_hidden.view(nb, ng, no, na, 3, 4)
        M_test = M.clone()
        M_test[:, :, :, 7] = torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]).to(M.device)
        M_test[:, :, :, 5] = torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]).to(M.device)
        out, rnn_hidden_states, new_keypoints = self.get_action(M, keypoints_homo, obstacles_rot_hidden,
                                                                obstacles_meta_hidden, a_feature, obstacles_state_hidden,
                                                                nb, ng, x, memory, masks, keypoints)
        #out_test, _, new_keypoints_test = self.get_action(M_test, keypoints_homo, obstacles_rot_hidden,
        #                                                  obstacles_meta_hidden, a_feature, obstacles_state_hidden,
        #                                                  nb, ng, x, memory, masks, keypoints, True)
        """
        new_keypoints = torch.matmul(M, keypoints_homo.transpose(4, 5)).transpose(4, 5)
        new_obstacles_state_hidden = self.state_encoding(new_keypoints.mean(4))

        atten_feature = torch.cat((obstacles_rot_hidden, obstacles_meta_hidden, a_feature), dim=4)
        hidden_feature = torch.cat((obstacles_meta_hidden, obstacles_state_hidden, new_obstacles_state_hidden,
                                    a_feature), dim=4)
        NPM_atten_score = self.NPM_atten(atten_feature)
        NPM_atten_prob = nn.functional.softmax(NPM_atten_score, 2)
        NPM_atten_hidden = (hidden_feature * NPM_atten_prob).sum(2)
        NPM_atten_hidden = self.NPM_summary(NPM_atten_hidden)
        NPM_atten_hidden = NPM_atten_hidden.view(nb, ng, -1)
        x.append(NPM_atten_hidden)

        x = torch.cat(x, dim=-1)
        x, rnn_hidden_states = self.state_encoder(x, memory.tensor("rnn"), masks)

        out = ActorCriticOutput(
            distributions=self.actor(x), values=self.critic(x), extras={}
        )
        """

        if not isinstance(current_actions, type(None)):
            new_keypoints = new_keypoints.view(nb * ng, no, na, 8, 3).transpose(1, 2)
            current_actions = current_actions.view(nb * ng, 1, 1).squeeze()
            NPM_out = new_keypoints[torch.arange(nb * ng), current_actions].reshape(nb, ng, no, 8, 3)
            out = {"ac_output": out, "npm_output": NPM_out}

        return out, memory.set_tensor("rnn", rnn_hidden_states)

    def get_action(self, M, keypoints_homo, obstacles_rot_hidden, obstacles_meta_hidden, a_feature, obstacles_state_hidden,
                   nb, ng, x, memory, masks, keypoints, second=False):
        new_keypoints = torch.matmul(M, keypoints_homo.transpose(4, 5)).transpose(4, 5)
        if second:
            new_keypoints[:,:,:,7] = keypoints[:,:,:,7].clone()
            new_keypoints[:, :, :, 5] = keypoints[:, :, :, 5].clone()
        new_obstacles_state_hidden = self.state_encoding(new_keypoints.mean(4))

        hidden_feature = torch.cat((obstacles_rot_hidden, obstacles_meta_hidden, obstacles_state_hidden, new_obstacles_state_hidden,
                                    a_feature), dim=4)

        NPM_atten_hidden = self.NPM_atten_a(hidden_feature)
        NPM_atten_hidden = self.NPM_atten_b(NPM_atten_hidden)
        NPM_atten_hidden = NPM_atten_hidden.mean(2)
        NPM_atten_hidden = self.NPM_summary(NPM_atten_hidden)
        NPM_atten_hidden = NPM_atten_hidden.view(nb, ng, -1)
        if second:
            x[-1] = NPM_atten_hidden
        else:
            x.append(NPM_atten_hidden)

        y = torch.cat(x, dim=-1)
        y, rnn_hidden_states = self.state_encoder(y, memory.tensor("rnn"), masks)

        out = ActorCriticOutput(
            distributions=self.actor(y), values=self.critic(y), extras={}
        )
        return out, rnn_hidden_states, new_keypoints


class ResnetTensorPointNavActorCritic(ActorCriticModel[CategoricalDistr]):
    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        goal_sensor_uuid: str,
        rgb_resnet_preprocessor_uuid: Optional[str] = None,
        depth_resnet_preprocessor_uuid: Optional[str] = None,
        hidden_size: int = 512,
        goal_dims: int = 32,
        resnet_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
        combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
    ):

        super().__init__(
            action_space=action_space, observation_space=observation_space,
        )

        self._hidden_size = hidden_size
        if (
            rgb_resnet_preprocessor_uuid is None
            or depth_resnet_preprocessor_uuid is None
        ):
            resnet_preprocessor_uuid = (
                rgb_resnet_preprocessor_uuid
                if rgb_resnet_preprocessor_uuid is not None
                else depth_resnet_preprocessor_uuid
            )
            self.goal_visual_encoder = ResnetTensorGoalEncoder(
                self.observation_space,
                goal_sensor_uuid,
                resnet_preprocessor_uuid,
                goal_dims,
                resnet_compressor_hidden_out_dims,
                combiner_hidden_out_dims,
            )
        else:
            self.goal_visual_encoder = ResnetDualTensorGoalEncoder(  # type:ignore
                self.observation_space,
                goal_sensor_uuid,
                rgb_resnet_preprocessor_uuid,
                depth_resnet_preprocessor_uuid,
                goal_dims,
                resnet_compressor_hidden_out_dims,
                combiner_hidden_out_dims,
            )
        self.state_encoder = RNNStateEncoder(
            self.goal_visual_encoder.output_dims, self._hidden_size,
        )
        self.actor = LinearActorHead(self._hidden_size, action_space.n)
        self.critic = LinearCriticHead(self._hidden_size)
        self.train()
        self.memory_key = "rnn"

    @property
    def recurrent_hidden_state_size(self) -> int:
        """The recurrent hidden state size of the model."""
        return self._hidden_size

    @property
    def is_blind(self) -> bool:
        """True if the model is blind (e.g. neither 'depth' or 'rgb' is an
        input observation type)."""
        return self.goal_visual_encoder.is_blind

    @property
    def num_recurrent_layers(self) -> int:
        """Number of recurrent hidden layers."""
        return self.state_encoder.num_recurrent_layers

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
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        x = self.goal_visual_encoder(observations)
        x, rnn_hidden_states = self.state_encoder(
            x, memory.tensor(self.memory_key), masks
        )
        return (
            ActorCriticOutput(
                distributions=self.actor(x), values=self.critic(x), extras={}
            ),
            memory.set_tensor(self.memory_key, rnn_hidden_states),
        )


class ResnetTensorGoalEncoder(nn.Module):
    def __init__(
        self,
        observation_spaces: SpaceDict,
        goal_sensor_uuid: str,
        resnet_preprocessor_uuid: str,
        goal_dims: int = 32,
        resnet_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
        combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
    ) -> None:
        super().__init__()
        self.goal_uuid = goal_sensor_uuid
        self.resnet_uuid = resnet_preprocessor_uuid
        self.goal_dims = goal_dims
        self.resnet_hid_out_dims = resnet_compressor_hidden_out_dims
        self.combine_hid_out_dims = combiner_hidden_out_dims
        self.embed_goal = nn.Linear(2, self.goal_dims)
        self.blind = self.resnet_uuid not in observation_spaces.spaces
        if not self.blind:
            self.resnet_tensor_shape = observation_spaces.spaces[self.resnet_uuid].shape
            self.resnet_compressor = nn.Sequential(
                nn.Conv2d(self.resnet_tensor_shape[0], self.resnet_hid_out_dims[0], 1),
                nn.ReLU(),
                nn.Conv2d(*self.resnet_hid_out_dims[0:2], 1),
                nn.ReLU(),
            )
            self.target_obs_combiner = nn.Sequential(
                nn.Conv2d(
                    self.resnet_hid_out_dims[1] + self.goal_dims,
                    self.combine_hid_out_dims[0],
                    1,
                ),
                nn.ReLU(),
                nn.Conv2d(*self.combine_hid_out_dims[0:2], 1),
            )

    @property
    def is_blind(self):
        return self.blind

    @property
    def output_dims(self):
        if self.blind:
            return self.goal_dims
        else:
            return (
                self.combine_hid_out_dims[-1]
                * self.resnet_tensor_shape[1]
                * self.resnet_tensor_shape[2]
            )

    def get_object_type_encoding(
        self, observations: Dict[str, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """Get the object type encoding from input batched observations."""
        return typing.cast(
            torch.FloatTensor,
            self.embed_goal(observations[self.goal_uuid].to(torch.int64)),
        )

    def compress_resnet(self, observations):
        return self.resnet_compressor(observations[self.resnet_uuid])

    def distribute_target(self, observations):
        target_emb = self.embed_goal(observations[self.goal_uuid])
        return target_emb.view(-1, self.goal_dims, 1, 1).expand(
            -1, -1, self.resnet_tensor_shape[-2], self.resnet_tensor_shape[-1]
        )

    def adapt_input(self, observations):
        resnet = observations[self.resnet_uuid]

        use_agent = False
        nagent = 1

        if len(resnet.shape) == 6:
            use_agent = True
            nstep, nsampler, nagent = resnet.shape[:3]
        else:
            nstep, nsampler = resnet.shape[:2]

        observations[self.resnet_uuid] = resnet.view(-1, *resnet.shape[-3:])
        observations[self.goal_uuid] = observations[self.goal_uuid].view(-1, 2)

        return observations, use_agent, nstep, nsampler, nagent

    @staticmethod
    def adapt_output(x, use_agent, nstep, nsampler, nagent):
        if use_agent:
            return x.view(nstep, nsampler, nagent, -1)
        return x.view(nstep, nsampler * nagent, -1)

    def forward(self, observations):
        observations, use_agent, nstep, nsampler, nagent = self.adapt_input(
            observations
        )

        if self.blind:
            return self.embed_goal(observations[self.goal_uuid])
        embs = [
            self.compress_resnet(observations),
            self.distribute_target(observations),
        ]
        x = self.target_obs_combiner(torch.cat(embs, dim=1,))
        x = x.reshape(x.size(0), -1)  # flatten

        return self.adapt_output(x, use_agent, nstep, nsampler, nagent)


class ResnetDualTensorGoalEncoder(nn.Module):
    def __init__(
        self,
        observation_spaces: SpaceDict,
        goal_sensor_uuid: str,
        rgb_resnet_preprocessor_uuid: str,
        depth_resnet_preprocessor_uuid: str,
        goal_dims: int = 32,
        resnet_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
        combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
    ) -> None:
        super().__init__()
        self.goal_uuid = goal_sensor_uuid
        self.rgb_resnet_uuid = rgb_resnet_preprocessor_uuid
        self.depth_resnet_uuid = depth_resnet_preprocessor_uuid
        self.goal_dims = goal_dims
        self.resnet_hid_out_dims = resnet_compressor_hidden_out_dims
        self.combine_hid_out_dims = combiner_hidden_out_dims
        self.embed_goal = nn.Linear(2, self.goal_dims)
        self.blind = (
            self.rgb_resnet_uuid not in observation_spaces.spaces
            or self.depth_resnet_uuid not in observation_spaces.spaces
        )
        if not self.blind:
            self.resnet_tensor_shape = observation_spaces.spaces[
                self.rgb_resnet_uuid
            ].shape
            self.rgb_resnet_compressor = nn.Sequential(
                nn.Conv2d(self.resnet_tensor_shape[0], self.resnet_hid_out_dims[0], 1),
                nn.ReLU(),
                nn.Conv2d(*self.resnet_hid_out_dims[0:2], 1),
                nn.ReLU(),
            )
            self.depth_resnet_compressor = nn.Sequential(
                nn.Conv2d(self.resnet_tensor_shape[0], self.resnet_hid_out_dims[0], 1),
                nn.ReLU(),
                nn.Conv2d(*self.resnet_hid_out_dims[0:2], 1),
                nn.ReLU(),
            )
            self.rgb_target_obs_combiner = nn.Sequential(
                nn.Conv2d(
                    self.resnet_hid_out_dims[1] + self.goal_dims,
                    self.combine_hid_out_dims[0],
                    1,
                ),
                nn.ReLU(),
                nn.Conv2d(*self.combine_hid_out_dims[0:2], 1),
            )
            self.depth_target_obs_combiner = nn.Sequential(
                nn.Conv2d(
                    self.resnet_hid_out_dims[1] + self.goal_dims,
                    self.combine_hid_out_dims[0],
                    1,
                ),
                nn.ReLU(),
                nn.Conv2d(*self.combine_hid_out_dims[0:2], 1),
            )

    @property
    def is_blind(self):
        return self.blind

    @property
    def output_dims(self):
        if self.blind:
            return self.goal_dims
        else:
            return (
                2
                * self.combine_hid_out_dims[-1]
                * self.resnet_tensor_shape[1]
                * self.resnet_tensor_shape[2]
            )

    def get_object_type_encoding(
        self, observations: Dict[str, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """Get the object type encoding from input batched observations."""
        return typing.cast(
            torch.FloatTensor,
            self.embed_goal(observations[self.goal_uuid].to(torch.int64)),
        )

    def compress_rgb_resnet(self, observations):
        return self.rgb_resnet_compressor(observations[self.rgb_resnet_uuid])

    def compress_depth_resnet(self, observations):
        return self.depth_resnet_compressor(observations[self.depth_resnet_uuid])

    def distribute_target(self, observations):
        target_emb = self.embed_goal(observations[self.goal_uuid])
        return target_emb.view(-1, self.goal_dims, 1, 1).expand(
            -1, -1, self.resnet_tensor_shape[-2], self.resnet_tensor_shape[-1]
        )

    def adapt_input(self, observations):
        rgb = observations[self.rgb_resnet_uuid]
        depth = observations[self.depth_resnet_uuid]

        use_agent = False
        nagent = 1

        if len(rgb.shape) == 6:
            use_agent = True
            nstep, nsampler, nagent = rgb.shape[:3]
        else:
            nstep, nsampler = rgb.shape[:2]

        observations[self.rgb_resnet_uuid] = rgb.view(-1, *rgb.shape[-3:])
        observations[self.depth_resnet_uuid] = depth.view(-1, *depth.shape[-3:])
        observations[self.goal_uuid] = observations[self.goal_uuid].view(-1, 2)

        return observations, use_agent, nstep, nsampler, nagent

    @staticmethod
    def adapt_output(x, use_agent, nstep, nsampler, nagent):
        if use_agent:
            return x.view(nstep, nsampler, nagent, -1)
        return x.view(nstep, nsampler, -1)

    def forward(self, observations):
        observations, use_agent, nstep, nsampler, nagent = self.adapt_input(
            observations
        )

        if self.blind:
            return self.embed_goal(observations[self.goal_uuid])
        rgb_embs = [
            self.compress_rgb_resnet(observations),
            self.distribute_target(observations),
        ]
        rgb_x = self.rgb_target_obs_combiner(torch.cat(rgb_embs, dim=1,))
        depth_embs = [
            self.compress_depth_resnet(observations),
            self.distribute_target(observations),
        ]
        depth_x = self.depth_target_obs_combiner(torch.cat(depth_embs, dim=1,))
        x = torch.cat([rgb_x, depth_x], dim=1)
        x = x.reshape(x.size(0), -1)  # flatten

        return self.adapt_output(x, use_agent, nstep, nsampler, nagent)

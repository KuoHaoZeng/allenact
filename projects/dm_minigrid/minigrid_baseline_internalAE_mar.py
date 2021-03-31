"""Experiment Config for MiniGrid tutorial."""

from typing import Dict, Optional, List, Any

import gym
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import LambdaLR

from plugins.minigrid_plugin.minigrid_models import MiniGridMAInternalAESimpleConvRNN
from plugins.minigrid_plugin.minigrid_sensors import EgocentricMiniGridSensor, PrevEgocentricMiniGridSensor
from plugins.minigrid_plugin.minigrid_tasks import DynamicsCorruptionEmptyTask
from core.algorithms.onpolicy_sync.losses.ppo import PPO, PPOConfig
from core.base_abstractions.sensor import SensorSuite
from utils.experiment_utils import TrainingPipeline, Builder, PipelineStage, LinearDecay

from projects.dm_minigrid.minigrid_baseline import MiniGridTutorialExperimentConfig


class MiniGridMAExperimentConfig(MiniGridTutorialExperimentConfig):
    def __init__(self):
        super().__init__()
        self.REWARD_CONFIG = dict(missing_action_penalty=-0.01)

    @classmethod
    def tag(cls) -> str:
        return "MiniGrid_ppo_internal_mar"

    SENSORS = [
        EgocentricMiniGridSensor(agent_view_size=5, view_channels=3),
        PrevEgocentricMiniGridSensor(agent_view_size=5, view_channels=3),
    ]

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return MiniGridMAInternalAESimpleConvRNN(
            action_space=gym.spaces.Discrete(len(DynamicsCorruptionEmptyTask.class_action_names())),
            observation_space=SensorSuite(cls.SENSORS).observation_spaces,
            num_objects=cls.SENSORS[0].num_objects,
            num_colors=cls.SENSORS[0].num_colors,
            num_states=cls.SENSORS[0].num_states,
            prev_action_embedding_dim=32,
        )
    @classmethod
    def training_pipeline(cls, **kwargs) -> TrainingPipeline:
        ppo_steps = int(10000000)
        return TrainingPipeline(
            named_losses=dict(ppo_loss=PPO(**PPOConfig),),
            pipeline_stages=[
                PipelineStage(loss_names=["ppo_loss"], max_stage_steps=ppo_steps)
            ],
            optimizer_builder=Builder(optim.Adam, dict(lr=1e-4)),
            num_mini_batch=4,
            update_repeats=3,
            max_grad_norm=0.5,
            num_steps=16,
            gamma=0.99,
            use_gae=True,
            gae_lambda=0.95,
            advance_scene_rollout_period=None,
            save_interval=1000000,
            metric_accumulate_interval=1,
            lr_scheduler_builder=Builder(
                LambdaLR, {"lr_lambda": LinearDecay(steps=ppo_steps)}  # type:ignore
            ),
        )

    @classmethod
    def machine_params(cls, mode="train", **kwargs) -> Dict[str, Any]:
        return {
            "nprocesses": 128 if mode == "train" else 16,
            "gpu_ids": [0],
        }


import gym
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from core.algorithms.onpolicy_sync.losses import PPO
from core.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from plugins.ithor_plugin.ithor_sensors import RGBSensorThor
from plugins.ithor_plugin.ithor_sensors import (
    DepthSensorIThor,
    GPSCompassSensorIThor,
    MissingActionSensor,
)
from plugins.ithor_plugin.ithor_tasks import PointNavObstaclesMissingActionTask
from projects.pointnav_baselines.experiments.ithor_obstacles.pointnav_ma_ithor_base import (
    PointNaviThorBaseConfig,
)
from projects.pointnav_baselines.models.point_nav_models import (
    PointNavMAActorCriticSimpleConvRNN,
)
from utils.experiment_utils import Builder, PipelineStage, TrainingPipeline, LinearDecay


class PointNaviThorRGBPPOExperimentConfig(PointNaviThorBaseConfig):
    """An Point Navigation experiment configuration in iThor with RGBD
    input."""

    def __init__(self):
        super().__init__()

        self.SENSORS = [
            RGBSensorThor(
                height=self.SCREEN_SIZE,
                width=self.SCREEN_SIZE,
                use_resnet_normalization=True,
                uuid="rgb",
            ),
            DepthSensorIThor(
                height=self.SCREEN_SIZE,
                width=self.SCREEN_SIZE,
                use_normalization=True,
                uuid="depth",
            ),
            GPSCompassSensorIThor(),
            MissingActionSensor(
                nactions=len(PointNavObstaclesMissingActionTask.class_action_names()),
                uuid="missing_action"
            ),
        ]

        self.PREPROCESSORS = []

        self.OBSERVATIONS = [
            "rgb",
            "depth",
            "target_coordinates_ind",
            "missing_action",
        ]

    @classmethod
    def tag(cls):
        return "Pointnav-ma-iTHOR-RGBDM-SimpleConv-DDPPO"

    @classmethod
    def training_pipeline(cls, **kwargs):
        ppo_steps = int(10000000)
        lr = 3e-4
        num_mini_batch = 1
        update_repeats = 3
        num_steps = 30
        save_interval = 100000
        log_interval = 100
        gamma = 0.99
        use_gae = True
        gae_lambda = 0.95
        max_grad_norm = 0.5
        return TrainingPipeline(
            save_interval=save_interval,
            metric_accumulate_interval=log_interval,
            optimizer_builder=Builder(optim.Adam, dict(lr=lr)),
            num_mini_batch=num_mini_batch,
            update_repeats=update_repeats,
            max_grad_norm=max_grad_norm,
            num_steps=num_steps,
            named_losses={"ppo_loss": PPO(**PPOConfig)},
            gamma=gamma,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
            advance_scene_rollout_period=cls.ADVANCE_SCENE_ROLLOUT_PERIOD,
            pipeline_stages=[
                PipelineStage(loss_names=["ppo_loss"], max_stage_steps=ppo_steps)
            ],
            lr_scheduler_builder=Builder(
                LambdaLR, {"lr_lambda": LinearDecay(steps=ppo_steps)}
            ),
        )

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return PointNavMAActorCriticSimpleConvRNN(
            action_space=gym.spaces.Discrete(len(PointNavObstaclesMissingActionTask.class_action_names())),
            observation_space=kwargs["observation_set"].observation_spaces,
            goal_sensor_uuid="target_coordinates_ind",
            hidden_size=512,
            embed_coordinates=False,
            coordinate_dims=2,
            missing_action_embedding_dim=32,
            missing_action_uuid="missing_action",
            num_rnn_layers=1,
            rnn_type="GRU",
        )

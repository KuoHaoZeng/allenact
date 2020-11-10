import gym
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from core.algorithms.onpolicy_sync.losses import PPO, NPM_Reg
from core.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from core.base_abstractions.sensor import ExpertActionSensor
from plugins.ithor_plugin.ithor_sensors import RGBSensorThor
from plugins.ithor_plugin.ithor_sensors import (
    DepthSensorIThor,
    GPSCompassSensorIThor,
    LocalKeyPoints3DSensorThor,
    GlobalKeyPoints3DSensorThor,
    GlobalObjPoseSensorThor,
    GlobalAgentPoseSensorThor,
    GlobalObjUpdateMaskSensorThor,
    GlobalObjActionMaskSensorThor,
)
from plugins.ithor_plugin.ithor_tasks import PointNavObstaclesTask
from projects.pointnav_baselines.experiments.ithor_obstacles.pointnav_ithor_base import (
    PointNaviThorBaseConfig,
)
from projects.pointnav_baselines.models.point_nav_models import (
    PointNavKeyPointsVisualNPMActorCriticSimpleConvRNN,
)
from utils.experiment_utils import Builder, PipelineStage, TrainingPipeline, LinearDecay
from plugins.ithor_plugin.ithor_constants import END

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
            LocalKeyPoints3DSensorThor(
                objectTypes=self.OBSTACLES_TYPES,
                uuid="3Dkeypoints_local"
            ),
            GlobalKeyPoints3DSensorThor(
                objectTypes=self.OBSTACLES_TYPES,
                uuid="3Dkeypoints_global"
            ),
            GlobalObjPoseSensorThor(
                objectTypes=self.OBSTACLES_TYPES,
                uuid="object_pose_global"
            ),
            GlobalAgentPoseSensorThor(
                uuid="agent_pose_global"
            ),
            GlobalObjUpdateMaskSensorThor(
                objectTypes=self.OBSTACLES_TYPES,
                uuid="object_update_mask"
            ),
            GlobalObjActionMaskSensorThor(
                objectTypes=self.OBSTACLES_TYPES,
                uuid="object_action_mask"
            ),
        ]

        self.PREPROCESSORS = []

        self.OBSERVATIONS = [
            "rgb",
            "depth",
            "target_coordinates_ind",
            "3Dkeypoints_local",
            "3Dkeypoints_global",
            "object_pose_global",
            "agent_pose_global",
            "object_update_mask",
            "object_action_mask",
        ]

    @classmethod
    def tag(cls):
        return "Pointnav-iTHOR-RGBD-KeyPoints-wVisual-SimpleConv-DDPPO"

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
            named_losses={
                "ppo_loss": PPO(**PPOConfig),
                "npm_loss": NPM_Reg(agent_pose_uuid="agent_pose_global",
                                    pose_uuid="object_pose_global",
                                    local_keypoints_uuid="3Dkeypoints_local",
                                    global_keypoints_uuid="3Dkeypoints_global",
                                    obj_update_mask_uuid="object_update_mask",
                                    obj_action_mask_uuid="object_action_mask",),
            },
            gamma=gamma,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
            advance_scene_rollout_period=cls.ADVANCE_SCENE_ROLLOUT_PERIOD,
            pipeline_stages=[
                PipelineStage(loss_names=["ppo_loss", "npm_loss"], max_stage_steps=ppo_steps)
            ],
            lr_scheduler_builder=Builder(
                LambdaLR, {"lr_lambda": LinearDecay(steps=ppo_steps)}
            ),
        )

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return PointNavKeyPointsVisualNPMActorCriticSimpleConvRNN(
            action_space=gym.spaces.Discrete(len(PointNavObstaclesTask.class_action_names())),
            observation_space=kwargs["observation_set"].observation_spaces,
            goal_sensor_uuid="target_coordinates_ind",
            obstacle_keypoints_sensor_uuid="3Dkeypoints_local",
            hidden_size=512,
            embed_coordinates=False,
            coordinate_dims=2,
            obstacle_type_embedding_dim=8,
            obstacle_state_hidden_dim=16,
            num_obstacle_types=20,
            num_rnn_layers=1,
            rnn_type="GRU",
        )

from typing import Any, Dict, Optional, List

import gym
import numpy as np

from plugins.ithor_plugin.ithor_environment import IThorEnvironment, IThorArmEnvironment
from plugins.ithor_plugin.ithor_tasks import ObjectNavTask, ObjectManipTask
from core.base_abstractions.sensor import Sensor, RGBSensor
from core.base_abstractions.task import Task
from utils.misc_utils import prepare_locals_for_super


class RGBSensorThor(RGBSensor[IThorEnvironment, Task[IThorEnvironment]]):
    """Sensor for RGB images in iTHOR.

    Returns from a running IThorEnvironment instance, the current RGB
    frame corresponding to the agent's egocentric view.
    """

    def frame_from_env(self, env: IThorEnvironment) -> np.ndarray:
        return env.current_frame.copy()

class GoalObjectTypeThorSensor(Sensor):
    def __init__(
        self,
        object_types: List[str],
        target_to_detector_map: Optional[Dict[str, str]] = None,
        detector_types: Optional[List[str]] = None,
        uuid: str = "goal_object_type_ind",
        **kwargs: Any
    ):
        self.ordered_object_types = list(object_types)
        assert self.ordered_object_types == sorted(
            self.ordered_object_types
        ), "object types input to goal object type sensor must be ordered"

        if target_to_detector_map is None:
            self.object_type_to_ind = {
                ot: i for i, ot in enumerate(self.ordered_object_types)
            }

            observation_space = gym.spaces.Discrete(len(self.ordered_object_types))
        else:
            assert (
                detector_types is not None
            ), "Missing detector_types for map {}".format(target_to_detector_map)
            self.target_to_detector = target_to_detector_map
            self.detector_types = detector_types

            detector_index = {ot: i for i, ot in enumerate(self.detector_types)}
            self.object_type_to_ind = {
                ot: detector_index[self.target_to_detector[ot]]
                for ot in self.ordered_object_types
            }

            observation_space = gym.spaces.Discrete(len(self.detector_types))

        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self,
        env: IThorEnvironment,
        task: Optional[ObjectNavTask],
        *args: Any,
        **kwargs: Any
    ) -> Any:
        return self.object_type_to_ind[task.task_info["object_type"]]

class GoalActionTypeThorSensor(Sensor):
    def __init__(
        self,
        action_types: List[str],
        uuid: str = "goal_action_type_ind",
        **kwargs: Any
    ):
        self.ordered_action_types = list(action_types)
        assert self.ordered_action_types == sorted(
            self.ordered_action_types
        ), "action types input to goal action type sensor must be ordered"

        self.action_type_to_ind = {
            ot: i for i, ot in enumerate(self.ordered_action_types)
        }

        observation_space = gym.spaces.Discrete(len(self.ordered_action_types))

        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self,
        env: IThorEnvironment,
        task: Optional[ObjectNavTask],
        *args: Any,
        **kwargs: Any
    ) -> Any:
        return self.action_type_to_ind[task.task_info["action_type"]]

class CurrentArmStateThorSensor(Sensor):
    """
    Get current hand target and arm locations.
    """
    def __init__(
        self,
        uuid: str = "current_arm_state",
        **kwargs: Any
    ):
        observation_space = gym.spaces.Box(low=-1.0, high=2.0, shape=(3,), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self,
        env: IThorArmEnvironment,
        task: Optional[ObjectManipTask],
        *args: Any,
        **kwargs: Any
    ):
        state = env.get_current_arm_coordinate()
        return [state['x'], state['y'], state['z']]

class ArmCollisionSensor(Sensor):
    """
    if last action is move arm, check whether the arm is moving succesfully.
    """
    def __init__(
        self,
        uuid: str = "arm_collision_state",
        **kwargs: Any
    ):
        observation_space = gym.spaces.Discrete(3)
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self,
        env: IThorArmEnvironment,
        task: Optional[ObjectManipTask],
        *args: Any,
        **kwargs: Any
    ):
        if "MoveMidLevelArm" == env.last_action:
            if env.last_action_success:
                state = 0
            else:
                state = 1
        else:
            state = 2

        # print("arm collision state %d:" %state)
        return state
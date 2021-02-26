import random
import math
import warnings
from typing import Dict, Tuple, List, Any, Optional, Union, Sequence, cast

import gym
import numpy as np

from plugins.ithor_plugin.ithor_environment import IThorEnvironment
from plugins.ithor_plugin.ithor_util import round_to_factor
from plugins.ithor_plugin.ithor_constants import (
    MOVE_AHEAD,
    ROTATE_LEFT,
    ROTATE_RIGHT,
    LOOK_DOWN,
    LOOK_UP,
    END,
    DIRECTIONAL_AHEAD_PUSH,
    DIRECTIONAL_BACK_PUSH,
    DIRECTIONAL_RIGHT_PUSH,
    DIRECTIONAL_LEFT_PUSH,
    MOVE_BACK
)
from core.base_abstractions.misc import RLStepResult
from core.base_abstractions.sensor import Sensor
from core.base_abstractions.task import Task
from utils.system import get_logger

from PIL import Image
import os, json
from utils.utils_3d_torch import project_3d_points_to_2d, draw_point
import torch

class ObjectNavTask(Task[IThorEnvironment]):
    """Defines the object navigation task in AI2-THOR.

    In object navigation an agent is randomly initialized into an AI2-THOR scene and must
    find an object of a given type (e.g. tomato, television, etc). An object is considered
    found if the agent takes an `End` action and the object is visible to the agent (see
    [here](https://ai2thor.allenai.org/documentation/concepts) for a definition of visibiliy
    in AI2-THOR).

    The actions available to an agent in this task are:

    1. Move ahead
        * Moves agent ahead by 0.25 meters.
    1. Rotate left / rotate right
        * Rotates the agent by 90 degrees counter-clockwise / clockwise.
    1. Look down / look up
        * Changes agent view angle by 30 degrees up or down. An agent cannot look more than 30
          degrees above horizontal or less than 60 degrees below horizontal.
    1. End
        * Ends the task and the agent receives a positive reward if the object type is visible to the agent,
        otherwise it receives a negative reward.

    # Attributes

    env : The ai2thor environment.
    sensor_suite: Collection of sensors formed from the `sensors` argument in the initializer.
    task_info : The task info. Must contain a field "object_type" that specifies, as a string,
        the goal object type.
    max_steps : The maximum number of steps an agent can take an in the task before it is considered failed.
    observation_space: The observation space returned on each step from the sensors.
    """

    _actions = (MOVE_AHEAD, ROTATE_LEFT, ROTATE_RIGHT, LOOK_DOWN, LOOK_UP, END)

    _CACHED_LOCATIONS_FROM_WHICH_OBJECT_IS_VISIBLE: Dict[
        Tuple[str, str], List[Tuple[float, float, int, int]]
    ] = {}

    def __init__(
        self,
        env: IThorEnvironment,
        sensors: List[Sensor],
        task_info: Dict[str, Any],
        max_steps: int,
        **kwargs
    ) -> None:
        """Initializer.

        See class documentation for parameter definitions.
        """
        super().__init__(
            env=env, sensors=sensors, task_info=task_info, max_steps=max_steps, **kwargs
        )
        self._took_end_action: bool = False
        self._success: Optional[bool] = False
        self._subsampled_locations_from_which_obj_visible: Optional[
            List[Tuple[float, float, int, int]]
        ] = None

    @property
    def action_space(self):
        return gym.spaces.Discrete(len(self._actions))

    def reached_terminal_state(self) -> bool:
        return self._took_end_action

    @classmethod
    def class_action_names(cls, **kwargs) -> Tuple[str, ...]:
        return cls._actions

    def close(self) -> None:
        self.env.stop()

    def _step(self, action: Union[int, Sequence[int]]) -> RLStepResult:
        assert isinstance(action, int)
        action = cast(int, action)

        action_str = self.class_action_names()[action]

        if action_str == END:
            self._took_end_action = True
            self._success = self._is_goal_object_visible()
            self.last_action_success = self._success
        else:
            self.env.step({"action": action_str})
            self.last_action_success = self.env.last_action_success

            if (
                not self.last_action_success
            ) and self._CACHED_LOCATIONS_FROM_WHICH_OBJECT_IS_VISIBLE is not None:
                self.env.update_graph_with_failed_action(failed_action=action_str)

        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=self.judge(),
            done=self.is_done(),
            info={"last_action_success": self.last_action_success},
        )
        return step_result

    def render(self, mode: str = "rgb", *args, **kwargs) -> np.ndarray:
        assert mode == "rgb", "only rgb rendering is implemented"
        return self.env.current_frame

    def _is_goal_object_visible(self) -> bool:
        """Is the goal object currently visible?"""
        return any(
            o["objectType"] == self.task_info["object_type"]
            for o in self.env.visible_objects()
        )

    def judge(self) -> float:
        """Compute the reward after having taken a step."""
        reward = -0.01

        if not self.last_action_success:
            reward += -0.03

        if self._took_end_action:
            reward += 1.0 if self._success else -1.0

        return float(reward)

    def metrics(self) -> Dict[str, Any]:
        if not self.is_done():
            return {}
        else:
            return {"success": self._success, **super(ObjectNavTask, self).metrics()}

    def query_expert(self, **kwargs) -> Tuple[int, bool]:
        target = self.task_info["object_type"]

        if self._is_goal_object_visible():
            return self.class_action_names().index(END), True
        else:
            key = (self.env.scene_name, target)
            if self._subsampled_locations_from_which_obj_visible is None:
                if key not in self._CACHED_LOCATIONS_FROM_WHICH_OBJECT_IS_VISIBLE:
                    obj_ids: List[str] = []
                    obj_ids.extend(
                        o["objectId"]
                        for o in self.env.last_event.metadata["objects"]
                        if o["objectType"] == target
                    )

                    assert len(obj_ids) != 0, "No objects to get an expert path to."

                    locations_from_which_object_is_visible: List[
                        Tuple[float, float, int, int]
                    ] = []
                    y = self.env.last_event.metadata["agent"]["position"]["y"]
                    positions_to_check_interactionable_from = [
                        {"x": x, "y": y, "z": z}
                        for x, z in set((x, z) for x, z, _, _ in self.env.graph.nodes)
                    ]
                    for obj_id in set(obj_ids):
                        self.env.controller.step(
                            {
                                "action": "PositionsFromWhichItemIsInteractable",
                                "objectId": obj_id,
                                "positions": positions_to_check_interactionable_from,
                            }
                        )
                        assert (
                            self.env.last_action_success
                        ), "Could not get positions from which item was interactable."

                        returned = self.env.last_event.metadata["actionReturn"]
                        locations_from_which_object_is_visible.extend(
                            (
                                round(x, 2),
                                round(z, 2),
                                round_to_factor(rot, 90) % 360,
                                round_to_factor(hor, 30) % 360,
                            )
                            for x, z, rot, hor, standing in zip(
                                returned["x"],
                                returned["z"],
                                returned["rotation"],
                                returned["horizon"],
                                returned["standing"],
                            )
                            if standing == 1
                        )

                    self._CACHED_LOCATIONS_FROM_WHICH_OBJECT_IS_VISIBLE[
                        key
                    ] = locations_from_which_object_is_visible

                self._subsampled_locations_from_which_obj_visible = self._CACHED_LOCATIONS_FROM_WHICH_OBJECT_IS_VISIBLE[
                    key
                ]
                if len(self._subsampled_locations_from_which_obj_visible) > 5:
                    self._subsampled_locations_from_which_obj_visible = random.sample(
                        self._CACHED_LOCATIONS_FROM_WHICH_OBJECT_IS_VISIBLE[key], 5
                    )

            current_loc_key = self.env.get_key(self.env.last_event.metadata["agent"])
            paths = []

            for goal_key in self._subsampled_locations_from_which_obj_visible:
                path = self.env.shortest_state_path(
                    source_state_key=current_loc_key, goal_state_key=goal_key
                )
                if path is not None:
                    paths.append(path)
            if len(paths) == 0:
                return 0, False

            shortest_path_ind = int(np.argmin([len(p) for p in paths]))

            if len(paths[shortest_path_ind]) == 1:
                warnings.warn(
                    "Shortest path computations suggest we are at the target but episode does not think so."
                )
                return 0, False

            next_key_on_shortest_path = paths[shortest_path_ind][1]
            return (
                self.class_action_names().index(
                    self.env.action_transitioning_between_keys(
                        current_loc_key, next_key_on_shortest_path
                    )
                ),
                True,
            )

class PointNavTask(Task[IThorEnvironment]):
    _actions = (MOVE_AHEAD, ROTATE_LEFT, ROTATE_RIGHT, END)

    def __init__(
            self,
            env: IThorEnvironment,
            sensors: List[Sensor],
            task_info: Dict[str, Any],
            max_steps: int,
            reward_configs: Dict[str, Any],
            **kwargs,
    ) -> None:
        super().__init__(
            env=env, sensors=sensors, task_info=task_info, max_steps=max_steps, **kwargs
        )
        self.reward_configs = reward_configs
        self._took_end_action: bool = False
        self._success: Optional[bool] = False
        self.last_geodesic_distance = self.env.distance_to_point(
            self.task_info["target"]
        )

        self.optimal_distance = self.last_geodesic_distance
        self._rewards: List[float] = []
        self._distance_to_goal: List[float] = []
        self._metrics = None
        self.path: List[Any] = (
            []
        )  # the initial coordinate will be directly taken from the optimal path

        self.task_info["followed_path"] = [self.env.agent_state()]
        self.task_info["action_names"] = self.action_names()
        self.num_moves_made = 0

    @property
    def action_space(self):
        return gym.spaces.Discrete(len(self._actions))

    def reached_terminal_state(self) -> bool:
        return self._took_end_action

    @classmethod
    def class_action_names(cls, **kwargs) -> Tuple[str, ...]:
        return cls._actions

    def close(self) -> None:
        self.env.stop()

    def _step(self, action: Union[int, Sequence[int]]) -> RLStepResult:
        assert isinstance(action, int)
        action = cast(int, action)

        action_str = self.action_names()[action]

        if action_str == END:
            self._took_end_action = True
            self._success = self._is_goal_in_range()
            self.last_action_success = self._success
        else:
            self.env.step({"action": action_str})
            self.last_action_success = self.env.last_action_success
            pose = self.env.agent_state()
            self.path.append({k: pose[k] for k in ["x", "y", "z"]})
            self.task_info["followed_path"].append(pose)
        if len(self.path) > 1 and self.path[-1] != self.path[-2]:
            self.num_moves_made += 1
        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=self.judge(),
            done=self.is_done(),
            info={"last_action_success": self.last_action_success, "action": action},
        )
        return step_result

    def render(self, mode: str = "rgb", *args, **kwargs) -> np.ndarray:
        assert mode in ["rgb", "depth"], "only rgb and depth rendering is implemented"
        if mode == "rgb":
            return self.env.current_frame
        elif mode == "depth":
            return self.env.current_depth

    def _is_goal_in_range(self) -> Optional[bool]:
        tget = self.task_info["target"]
        dist = self.dist_to_target()

        if -0.5 < dist <= 0.2:
            return True
        elif dist > 0.2:
            return False
        else:
            get_logger().debug(
                "No path for {} from {} to {}".format(
                    self.env.scene_name, self.env.agent_state(), tget
                )
            )
            return None

    def shaping(self) -> float:
        rew = 0.0

        if self.reward_configs["shaping_weight"] == 0.0:
            return rew

        geodesic_distance = self.dist_to_target()

        if geodesic_distance == -1.0:
            geodesic_distance = self.last_geodesic_distance
        if (
                self.last_geodesic_distance > -0.5 and geodesic_distance > -0.5
        ):  # (robothor limits)
            rew += self.last_geodesic_distance - geodesic_distance
        self.last_geodesic_distance = geodesic_distance

        return rew * self.reward_configs["shaping_weight"]

    def judge(self) -> float:
        """Judge the last event."""
        reward = self.reward_configs["step_penalty"]

        reward += self.shaping()

        if self._took_end_action:
            if self._success is not None:
                reward += (
                    self.reward_configs["goal_success_reward"]
                    if self._success
                    else self.reward_configs["failed_stop_reward"]
                )

        self._rewards.append(float(reward))
        return float(reward)

    def spl(self):
        if not self._success:
            return 0.0
        li = self.optimal_distance
        pi = self.num_moves_made * self.env.config["gridSize"]
        res = li / (max(pi, li) + 1e-8)
        return res

    def dist_to_target(self):
        return self.env.distance_to_point(self.task_info["target"])

    def metrics(self) -> Dict[str, Any]:
        if not self.is_done():
            return {}

        total_reward = float(np.sum(self._rewards))
        self._rewards = []

        if self._success is None:
            return {}

        dist2tget = self.dist_to_target()
        spl = self.spl()

        return {
            **super(PointNavTask, self).metrics(),
            "success": self._success,  # False also if no path to target
            "total_reward": total_reward,
            "dist_to_target": dist2tget,
            "spl": spl,
        }


class PointNavObstaclesTask(Task[IThorEnvironment]):
    _actions = (MOVE_AHEAD, ROTATE_LEFT, ROTATE_RIGHT, LOOK_DOWN, LOOK_UP,
                DIRECTIONAL_AHEAD_PUSH, DIRECTIONAL_BACK_PUSH, DIRECTIONAL_RIGHT_PUSH, DIRECTIONAL_LEFT_PUSH,
                END)

    def __init__(
            self,
            env: IThorEnvironment,
            sensors: List[Sensor],
            task_info: Dict[str, Any],
            max_steps: int,
            reward_configs: Dict[str, Any],
            **kwargs,
    ) -> None:
        super().__init__(
            env=env, sensors=sensors, task_info=task_info, max_steps=max_steps, **kwargs
        )
        self.reward_configs = reward_configs
        self._took_end_action: bool = False
        self._success: Optional[bool] = False
        self.last_geodesic_distance = self.env.distance_to_point(
            self.task_info["target"]
        )
        self.last_tget_in_path = False

        self.optimal_distance = self.last_geodesic_distance
        self._rewards: List[float] = []
        self._distance_to_goal: List[float] = []
        self._metrics = None
        self.path: List[Any] = (
            []
        )  # the initial coordinate will be directly taken from the optimal path

        self.task_info["followed_path"] = [self.env.agent_state()]
        self.task_info["action_names"] = self.action_names()
        self.num_moves_made = 0

    @property
    def action_space(self):
        return gym.spaces.Discrete(len(self._actions))

    def reached_terminal_state(self) -> bool:
        return self._took_end_action

    @classmethod
    def class_action_names(cls, **kwargs) -> Tuple[str, ...]:
        return cls._actions

    def close(self) -> None:
        self.env.stop()

    def _step(self, action: Union[int, Sequence[int]]) -> RLStepResult:

        assert isinstance(action, int)
        action = cast(int, action)

        action_str = self.action_names()[action]

        if action_str == END:
            self._took_end_action = True
            self._success = self._is_goal_in_range()
            self.last_action_success = self._success
        elif action_str in [DIRECTIONAL_AHEAD_PUSH, DIRECTIONAL_BACK_PUSH,
                            DIRECTIONAL_RIGHT_PUSH, DIRECTIONAL_LEFT_PUSH]:
            if action_str not in [DIRECTIONAL_RIGHT_PUSH, DIRECTIONAL_LEFT_PUSH]:
                angle = [0.001, 180, 90, 270][action - 5]
                obj = self.env.moveable_closest_obj_by_types(self.task_info["obstacles_types"])
                if obj != None:
                    self.env.step({"action": action_str,
                                   "objectId": obj["objectId"],
                                   "moveMagnitude": obj["mass"] * 100,
                                   "pushAngle": angle,
                                   "autoSimulation": False})
                    self.last_action_success = self.env.last_action_success
                else:
                    self.last_action_success = False
            else:
                angle = [0.001, 180, 90, 270][action - 5]
                obj = self.env.moveable_closest_obj_by_types(self.task_info["obstacles_types"])
                if obj != None:
                    self.env.step({"action": action_str,
                                   "objectId": obj["objectId"],
                                   "moveMagnitude": obj["mass"] * 100,
                                   "pushAngle": angle,
                                   "autoSimulation": False})
                    self.last_action_success = self.env.last_action_success
                else:
                    self.last_action_success = False
        elif action_str in [LOOK_UP, LOOK_DOWN]:
            self.env.step({"action": action_str})
            self.last_action_success = self.env.last_action_success
        else:
            self.env.step({"action": action_str})
            self.last_action_success = self.env.last_action_success
            pose = self.env.agent_state()
            self.path.append({k: pose[k] for k in ["x", "y", "z"]})
            self.task_info["followed_path"].append(pose)
        if len(self.path) > 1 and self.path[-1] != self.path[-2]:
            self.num_moves_made += 1
        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=self.judge(),
            done=self.is_done(),
            info={"last_action_success": self.last_action_success, "action": action},
        )

        if False:
            tgts = []
            for i in range(10):
                tgt = [self.task_info["target"]["x"], i * 0.1, self.task_info["target"]["z"]]
                tgts.append(tgt)
            tgts = torch.Tensor(tgts).unsqueeze(0)
            tgts_2d = project_3d_points_to_2d([self.env.last_event.metadata], tgts)
            tgts_2d = tgts_2d.squeeze(0).numpy()
            tgts_2d = np.array([[y, x] for (x, y) in tgts_2d])

            self.env.counter += 1
            #method_name = "rgbd_kp_1xNPM"
            method_name = "rgbd"
            folder = "qualitative_results/{}/{}/{}".format(method_name,
                                                           self.task_info["scene"],
                                                           self.task_info["id"])
            if not os.path.isdir("{}".format(folder)):
                os.makedirs("{}".format(folder))
            img = Image.fromarray(self.env.current_frame, "RGB")
            img = draw_point(img, tgts_2d)
            img.save("{}/{:05}.png".format(folder, self.env.counter))

        return step_result

    def render(self, mode: str = "rgb", *args, **kwargs) -> np.ndarray:
        assert mode in ["rgb", "depth"], "only rgb and depth rendering is implemented"
        if mode == "rgb":
            return self.env.current_frame
        elif mode == "depth":
            return self.env.current_depth

    def _is_goal_in_range(self) -> Optional[bool]:
        tget = self.task_info["target"]
        dist = self.dist_to_target()

        if -0.5 < dist <= 0.2:
            return True
        elif dist > 0.2:
            return False
        else:
            get_logger().debug(
                "No path for {} from {} to {}".format(
                    self.env.scene_name, self.env.agent_state(), tget
                )
            )
            return None

    def shaping(self) -> float:
        rew = 0.0

        if self.reward_configs["shaping_weight"] == 0.0:
            return rew

        geodesic_distance = self.dist_to_target()

        if geodesic_distance == -1.0:
            geodesic_distance = self.last_geodesic_distance
        if (
                self.last_geodesic_distance > -0.5 and geodesic_distance > -0.5
        ):  # (robothor limits)
            rew += self.last_geodesic_distance - geodesic_distance
        self.last_geodesic_distance = geodesic_distance

        return rew * self.reward_configs["shaping_weight"]

    def shaping_by_path(self) -> float:
        reward = 0.0
        if self.env.last_action in [DIRECTIONAL_AHEAD_PUSH, DIRECTIONAL_BACK_PUSH,
                                    DIRECTIONAL_RIGHT_PUSH, DIRECTIONAL_LEFT_PUSH]:
            tget_in_path = self.env.target_in_reachable_points(self.task_info["target"])
            if tget_in_path and not self.last_tget_in_path:
                reward += 0.5
            elif not tget_in_path and self.last_tget_in_path:
                reward -= 0.5
            self.last_tget_in_path = tget_in_path
        return reward

    def judge(self) -> float:
        """Judge the last event."""
        reward = self.reward_configs["step_penalty"]

        reward += self.shaping()
        reward += self.shaping_by_path()

        if self._took_end_action:
            if self._success is not None:
                reward += (
                    self.reward_configs["goal_success_reward"]
                    if self._success
                    else self.reward_configs["failed_stop_reward"]
                )

        self._rewards.append(float(reward))
        return float(reward)

    def spl(self):
        if not self._success:
            return 0.0
        li = self.optimal_distance
        pi = self.num_moves_made * self.env._grid_size
        res = li / (max(pi, li) + 1e-8)
        return res

    def dist_to_target(self):
        return self.env.distance_to_point(self.task_info["target"])

    def metrics(self) -> Dict[str, Any]:
        if not self.is_done():
            return {}

        total_reward = float(np.sum(self._rewards))
        self._rewards = []

        if self._success is None:
            return {}

        dist2tget = self.dist_to_target()
        spl = self.spl()

        return {
            **super(PointNavObstaclesTask, self).metrics(),
            "success": self._success,  # False also if no path to target
            "total_reward": total_reward,
            "dist_to_target": dist2tget,
            "spl": spl,
            "target_in_reachable_points": self.last_tget_in_path,
        }

    def query_expert(self, end_action_only: bool = False, **kwargs) -> Tuple[int, bool]:
        if self._is_goal_in_range():
            return self.class_action_names().index(END), True
        if end_action_only:
            return 0, False
        else:
            raise NotImplementedError


class PlacementTask(Task[IThorEnvironment]):
    _actions = (MOVE_AHEAD, ROTATE_LEFT, ROTATE_RIGHT, LOOK_DOWN, LOOK_UP,
                DIRECTIONAL_AHEAD_PUSH, DIRECTIONAL_BACK_PUSH, DIRECTIONAL_RIGHT_PUSH, DIRECTIONAL_LEFT_PUSH,
                END)

    def __init__(
            self,
            env: IThorEnvironment,
            sensors: List[Sensor],
            task_info: Dict[str, Any],
            max_steps: int,
            reward_configs: Dict[str, Any],
            **kwargs,
    ) -> None:
        super().__init__(
            env=env, sensors=sensors, task_info=task_info, max_steps=max_steps, **kwargs
        )
        self.reward_configs = reward_configs
        self._took_end_action: bool = False
        self._success: Optional[bool] = False
        self.last_geodesic_distance = self.env.distance_to_point(
            self.task_info["target"]
        )
        self.obj_last_geodesic_distance = self.obj_dist_to_target()
        self.last_both_in_path = False

        self.optimal_distance = self.last_geodesic_distance + self.obj_last_geodesic_distance
        self._rewards: List[float] = []
        self._distance_to_goal: List[float] = []
        self._metrics = None
        self.path: List[Any] = (
            []
        )  # the initial coordinate will be directly taken from the optimal path

        self.task_info["followed_path"] = [self.env.agent_state()]
        self.task_info["action_names"] = self.action_names()
        self.num_moves_made = 0

    @property
    def action_space(self):
        return gym.spaces.Discrete(len(self._actions))

    def reached_terminal_state(self) -> bool:
        return self._took_end_action

    @classmethod
    def class_action_names(cls, **kwargs) -> Tuple[str, ...]:
        return cls._actions

    def close(self) -> None:
        self.env.stop()

    def _step(self, action: Union[int, Sequence[int]]) -> RLStepResult:

        assert isinstance(action, int)
        action = cast(int, action)

        action_str = self.action_names()[action]

        if action_str == END:
            self._took_end_action = True
            self._success = self._is_goal_in_range()
            self.last_action_success = self._success
        elif action_str in [DIRECTIONAL_AHEAD_PUSH, DIRECTIONAL_BACK_PUSH,
                            DIRECTIONAL_RIGHT_PUSH, DIRECTIONAL_LEFT_PUSH]:
            if action_str not in [DIRECTIONAL_RIGHT_PUSH, DIRECTIONAL_LEFT_PUSH]:
                angle = [0.001, 180, 90, 270][action - 5]
                obj = self.env.moveable_closest_obj_by_types(self.task_info["obstacles_types"])
                if obj != None:
                    self.env.step({"action": action_str,
                                   "objectId": obj["objectId"],
                                   "moveMagnitude": obj["mass"] * 100,
                                   "pushAngle": angle,
                                   "autoSimulation": False})
                    self.last_action_success = self.env.last_action_success
                else:
                    self.last_action_success = False
            else:
                angle = [0.001, 180, 90, 270][action - 5]
                obj = self.env.moveable_closest_obj_by_types(self.task_info["obstacles_types"])
                if obj != None:
                    self.env.step({"action": action_str,
                                   "objectId": obj["objectId"],
                                   "moveMagnitude": obj["mass"] * 100,
                                   "pushAngle": angle,
                                   "autoSimulation": False})
                    self.last_action_success = self.env.last_action_success
                else:
                    self.last_action_success = False
        elif action_str in [LOOK_UP, LOOK_DOWN]:
            self.env.step({"action": action_str})
            self.last_action_success = self.env.last_action_success
        else:
            self.env.step({"action": action_str})
            self.last_action_success = self.env.last_action_success
            pose = self.env.agent_state()
            self.path.append({k: pose[k] for k in ["x", "y", "z"]})
            self.task_info["followed_path"].append(pose)
        if len(self.path) > 1 and self.path[-1] != self.path[-2]:
            self.num_moves_made += 1
        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=self.judge(),
            done=self.is_done(),
            info={"last_action_success": self.last_action_success, "action": action},
        )

        if False:
            tgts = []
            for i in range(10):
                tgt = [self.task_info["target"]["x"], i * 0.1, self.task_info["target"]["z"]]
                tgts.append(tgt)
            tgts = torch.Tensor(tgts).unsqueeze(0)
            tgts_2d = project_3d_points_to_2d([self.env.last_event.metadata], tgts)
            tgts_2d = tgts_2d.squeeze(0).numpy()
            tgts_2d = np.array([[y, x] for (x, y) in tgts_2d])

            self.env.counter += 1
            method_name = "rgbd_kp_3xNPM_visual_large"
            #method_name = "rgbd"
            folder = "qualitative_results/{}/{}/{}".format(method_name,
                                                           self.task_info["scene"],
                                                           self.task_info["id"])
            if not os.path.isdir("{}".format(folder)):
                os.makedirs("{}".format(folder))
            img = Image.fromarray(self.env.current_frame, "RGB")
            img = draw_point(img, tgts_2d)
            img.save("{}/{:05}.png".format(folder, self.env.counter))

        return step_result

    def render(self, mode: str = "rgb", *args, **kwargs) -> np.ndarray:
        assert mode in ["rgb", "depth"], "only rgb and depth rendering is implemented"
        if mode == "rgb":
            return self.env.current_frame
        elif mode == "depth":
            return self.env.current_depth

    def _is_goal_in_range(self) -> Optional[bool]:
        objs = self.env.get_objects_by_type(self.task_info["object_type"])
        tgt_obj = self.env.get_objects_by_type(self.task_info["target_type"])[0]
        for obj in objs:
            if obj["objectId"] in tgt_obj["receptacleObjectIds"]:
                return True

        tget = self.task_info["target"]
        dist = self.obj_dist_to_target()

        if -0.5 < dist <= 0.2:
            return True
        elif dist > 0.2:
            return False
        else:
            get_logger().debug(
                "No path for {} from {} to {}".format(
                    self.env.scene_name, self.env.agent_state(), tget
                )
            )
            return None

    def shaping(self) -> float:
        rew = 0.0

        if self.reward_configs["shaping_weight"] == 0.0:
            return rew

        geodesic_distance = self.obj_dist_to_target()

        if geodesic_distance == -1.0:
            geodesic_distance = self.obj_last_geodesic_distance
        if (
                self.obj_last_geodesic_distance > -0.5 and geodesic_distance > -0.5
        ):  # (robothor limits)
            rew += self.obj_last_geodesic_distance - geodesic_distance
        self.obj_last_geodesic_distance = geodesic_distance
        """
        if geodesic_distance < self.obj_last_geodesic_distance:
            rew += 0.02
        self.obj_last_geodesic_distance = geodesic_distance
        """

        """
        geodesic_distance = self.dist_to_target()

        if geodesic_distance == -1.0:
            geodesic_distance = self.last_geodesic_distance
        if (
                self.last_geodesic_distance > -0.5 and geodesic_distance > -0.5
        ):  # (robothor limits)
            rew += self.last_geodesic_distance - geodesic_distance
        self.last_geodesic_distance = geodesic_distance
        """

        return rew * self.reward_configs["shaping_weight"]

    def shaping_by_path(self) -> float:
        reward = 0.0
        if self.env.last_action in [DIRECTIONAL_AHEAD_PUSH, DIRECTIONAL_BACK_PUSH,
                                    DIRECTIONAL_RIGHT_PUSH, DIRECTIONAL_LEFT_PUSH]:
            objs = self.env.get_objects_by_type(self.task_info["object_type"])
            objs_in_path = False
            for obj in objs:
                if self.env.target_in_reachable_points(obj["position"]):
                    objs_in_path = True
                    break

            tget_in_path = self.env.target_in_reachable_points(self.task_info["target"])

            both_in_path = objs_in_path and tget_in_path

            if both_in_path and not self.last_both_in_path:
                reward += 0.5
            elif not both_in_path and self.last_both_in_path:
                reward -= 0.5
            self.last_both_in_path = both_in_path
        return reward

    def judge(self) -> float:
        """Judge the last event."""
        reward = self.reward_configs["step_penalty"]
        #reward = 0.

        #reward += self.shaping_by_path()
        reward += self.shaping()

        if self._took_end_action:
            if self._success is not None:
                reward += (
                    self.reward_configs["goal_success_reward"]
                    if self._success
                    else self.reward_configs["failed_stop_reward"]
                )

        self._rewards.append(float(reward))
        return float(reward)

    def spl(self):
        if not self._success:
            return 0.0
        li = self.optimal_distance
        pi = self.num_moves_made * self.env._grid_size
        res = li / (max(pi, li) + 1e-8)
        return res

    def dist_to_target(self):
        objs, idx = self.env.get_objects_and_idx_by_type(self.task_info["object_type"])
        dis = []
        for id in idx:
            #dis.append(self.env.distance_to_point(self.env.all_objects()[id]["position"]))
            dis.append(self.env.object_distance_to_point(id, self.task_info["target"]))
        id = idx[np.argmin(dis)]
        #return self.env.object_distance_to_point(id, self.task_info["target"])
        return self.env.distance_to_point(self.env.all_objects()[id]["position"])

    def obj_dist_to_target(self):
        objs, idx = self.env.get_objects_and_idx_by_type(self.task_info["object_type"])
        dis = []
        for id in idx:
            dis.append(self.env.object_distance_to_point(id, self.task_info["target"]))
        return min(dis)

    def metrics(self) -> Dict[str, Any]:
        if not self.is_done():
            return {}

        total_reward = float(np.sum(self._rewards))
        self._rewards = []

        if self._success is None:
            return {}

        dist2tget = self.obj_dist_to_target()
        spl = self.spl()

        return {
            **super(PlacementTask, self).metrics(),
            "success": self._success,  # False also if no path to target
            "total_reward": total_reward,
            "dist_to_target": dist2tget,
            "spl": spl,
            "both_in_reachable_points": self.last_both_in_path,
        }

    def query_expert(self, end_action_only: bool = False, **kwargs) -> Tuple[int, bool]:
        if self._is_goal_in_range():
            return self.class_action_names().index(END), True
        if end_action_only:
            return 0, False
        else:
            raise NotImplementedError


class PointNavObstaclesMissingActionTask(Task[IThorEnvironment]):
    _actions = (MOVE_AHEAD, MOVE_BACK, ROTATE_LEFT, ROTATE_RIGHT, LOOK_DOWN, LOOK_UP,
                DIRECTIONAL_AHEAD_PUSH, DIRECTIONAL_BACK_PUSH, DIRECTIONAL_RIGHT_PUSH, DIRECTIONAL_LEFT_PUSH,
                END)

    def __init__(
            self,
            env: IThorEnvironment,
            sensors: List[Sensor],
            task_info: Dict[str, Any],
            max_steps: int,
            reward_configs: Dict[str, Any],
            **kwargs,
    ) -> None:
        super().__init__(
            env=env, sensors=sensors, task_info=task_info, max_steps=max_steps, **kwargs
        )
        self.reward_configs = reward_configs
        self._took_end_action: bool = False
        self._success: Optional[bool] = False
        self.last_geodesic_distance = self.env.distance_to_point(
            self.task_info["target"]
        )
        self.last_tget_in_path = False

        self.optimal_distance = self.last_geodesic_distance
        self._rewards: List[float] = []
        self._distance_to_goal: List[float] = []
        self._metrics = None
        self.path: List[Any] = (
            []
        )  # the initial coordinate will be directly taken from the optimal path

        self.task_info["followed_path"] = [self.env.agent_state()]
        self.task_info["action_names"] = self.action_names()
        self.num_moves_made = 0
        self.last_missing_action_made = False
        self.num_missing_action_made = 0
        self.num_action_made = 0

    @property
    def action_space(self):
        return gym.spaces.Discrete(len(self._actions))

    def reached_terminal_state(self) -> bool:
        return self._took_end_action

    @classmethod
    def class_action_names(cls, **kwargs) -> Tuple[str, ...]:
        return cls._actions

    def close(self) -> None:
        self.env.stop()

    def _step(self, action: Union[int, Sequence[int]]) -> RLStepResult:

        assert isinstance(action, int)
        action = cast(int, action)

        if action != self.task_info["missing_action"]:
            action_str = self.action_names()[action]

            if action_str == END:
                self._took_end_action = True
                self._success = self._is_goal_in_range()
                self.last_action_success = self._success
            elif action_str in [DIRECTIONAL_AHEAD_PUSH, DIRECTIONAL_BACK_PUSH,
                                DIRECTIONAL_RIGHT_PUSH, DIRECTIONAL_LEFT_PUSH]:
                angle = [0.001, 180, 90, 270][action - 6]
                obj = self.env.moveable_closest_obj_by_types(self.task_info["obstacles_types"])
                if obj != None:
                    self.env.step({"action": action_str,
                                   "objectId": obj["objectId"],
                                   "moveMagnitude": obj["mass"] * 100,
                                   "pushAngle": angle,
                                   "autoSimulation": False})
                    self.last_action_success = self.env.last_action_success
                else:
                    self.last_action_success = False
            elif action_str in [LOOK_UP, LOOK_DOWN]:
                self.env.step({"action": action_str})
                self.last_action_success = self.env.last_action_success
            else:
                self.env.step({"action": action_str})
                self.last_action_success = self.env.last_action_success
                pose = self.env.agent_state()
                self.path.append({k: pose[k] for k in ["x", "y", "z"]})
                self.task_info["followed_path"].append(pose)
        else:
            self.last_action_success = False
            self.last_missing_action_made = True
            self.num_missing_action_made += 1
        self.num_action_made += 1

        if len(self.path) > 1 and self.path[-1] != self.path[-2]:
            self.num_moves_made += 1
        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=self.judge(),
            done=self.is_done(),
            info={"last_action_success": self.last_action_success, "action": action},
        )

        if False:
            tgts = []
            for i in range(10):
                tgt = [self.task_info["target"]["x"], i * 0.1, self.task_info["target"]["z"]]
                tgts.append(tgt)
            tgts = torch.Tensor(tgts).unsqueeze(0)
            tgts_2d = project_3d_points_to_2d([self.env.last_event.metadata], tgts)
            tgts_2d = tgts_2d.squeeze(0).numpy()
            tgts_2d = np.array([[y, x] for (x, y) in tgts_2d])

            self.env.counter += 1
            #method_name = "rgbd_kp_1xNPM"
            method_name = "rgbd"
            folder = "qualitative_results/{}/{}/{}".format(method_name,
                                                           self.task_info["scene"],
                                                           self.task_info["id"])
            if not os.path.isdir("{}".format(folder)):
                os.makedirs("{}".format(folder))
            img = Image.fromarray(self.env.current_frame, "RGB")
            img = draw_point(img, tgts_2d)
            img.save("{}/{:05}.png".format(folder, self.env.counter))

        return step_result

    def render(self, mode: str = "rgb", *args, **kwargs) -> np.ndarray:
        assert mode in ["rgb", "depth"], "only rgb and depth rendering is implemented"
        if mode == "rgb":
            return self.env.current_frame
        elif mode == "depth":
            return self.env.current_depth

    def _is_goal_in_range(self) -> Optional[bool]:
        tget = self.task_info["target"]
        dist = self.dist_to_target()

        if -0.5 < dist <= 0.2:
            return True
        elif dist > 0.2:
            return False
        else:
            get_logger().debug(
                "No path for {} from {} to {}".format(
                    self.env.scene_name, self.env.agent_state(), tget
                )
            )
            return None

    def shaping(self) -> float:
        rew = 0.0

        if self.reward_configs["shaping_weight"] == 0.0:
            return rew

        geodesic_distance = self.dist_to_target()

        if geodesic_distance == -1.0:
            geodesic_distance = self.last_geodesic_distance
        if (
                self.last_geodesic_distance > -0.5 and geodesic_distance > -0.5
        ):  # (robothor limits)
            rew += self.last_geodesic_distance - geodesic_distance
        self.last_geodesic_distance = geodesic_distance

        return rew * self.reward_configs["shaping_weight"]

    def shaping_by_path(self) -> float:
        reward = 0.0
        if self.env.last_action in [DIRECTIONAL_AHEAD_PUSH, DIRECTIONAL_BACK_PUSH,
                                    DIRECTIONAL_RIGHT_PUSH, DIRECTIONAL_LEFT_PUSH]:
            tget_in_path = self.env.target_in_reachable_points(self.task_info["target"])
            if tget_in_path and not self.last_tget_in_path:
                reward += 0.5
            elif not tget_in_path and self.last_tget_in_path:
                reward -= 0.5
            self.last_tget_in_path = tget_in_path
        return reward

    def shaping_by_missing_action(self, penalty) -> float:
        if self.last_missing_action_made:
            self.last_missing_action_made = False
            return penalty
        else:
            return 0

    def judge(self) -> float:
        """Judge the last event."""
        reward = self.reward_configs["step_penalty"]

        reward += self.shaping()
        reward += self.shaping_by_path()
        if "missing_action_penalty" in self.reward_configs.keys():
            reward += self.shaping_by_missing_action(self.reward_configs["missing_action_penalty"])

        if self._took_end_action:
            if self._success is not None:
                reward += (
                    self.reward_configs["goal_success_reward"]
                    if self._success
                    else self.reward_configs["failed_stop_reward"]
                )

        self._rewards.append(float(reward))
        return float(reward)

    def spl(self):
        if not self._success:
            return 0.0
        li = self.optimal_distance
        pi = self.num_moves_made * self.env._grid_size
        res = li / (max(pi, li) + 1e-8)
        return res

    def dist_to_target(self):
        return self.env.distance_to_point(self.task_info["target"])

    def metrics(self) -> Dict[str, Any]:
        if not self.is_done():
            return {}

        total_reward = float(np.sum(self._rewards))
        self._rewards = []

        if self._success is None:
            return {}

        dist2tget = self.dist_to_target()
        spl = self.spl()

        return {
            **super(PointNavObstaclesMissingActionTask, self).metrics(),
            "success": self._success,  # False also if no path to target
            "total_reward": total_reward,
            "dist_to_target": dist2tget,
            "spl": spl,
            "target_in_reachable_points": self.last_tget_in_path,
            "num_missing_action": self.num_missing_action_made,
            "missing_action_ratio": self.num_missing_action_made / float(self.num_action_made),
        }

    def query_expert(self, end_action_only: bool = False, **kwargs) -> Tuple[int, bool]:
        if self._is_goal_in_range():
            return self.class_action_names().index(END), True
        if end_action_only:
            return 0, False
        else:
            raise NotImplementedError


class PointNavMissingActionTask(Task[IThorEnvironment]):
    _actions = (MOVE_AHEAD, MOVE_BACK, ROTATE_LEFT, ROTATE_RIGHT, LOOK_DOWN, LOOK_UP, END)

    def __init__(
            self,
            env: IThorEnvironment,
            sensors: List[Sensor],
            task_info: Dict[str, Any],
            max_steps: int,
            reward_configs: Dict[str, Any],
            **kwargs,
    ) -> None:
        super().__init__(
            env=env, sensors=sensors, task_info=task_info, max_steps=max_steps, **kwargs
        )
        self.reward_configs = reward_configs
        self._took_end_action: bool = False
        self._success: Optional[bool] = False
        self.last_geodesic_distance = self.env.distance_to_point(
            self.task_info["target"]
        )
        self.last_tget_in_path = False

        self.optimal_distance = self.last_geodesic_distance
        self._rewards: List[float] = []
        self._distance_to_goal: List[float] = []
        self._metrics = None
        self.path: List[Any] = (
            []
        )  # the initial coordinate will be directly taken from the optimal path

        self.task_info["followed_path"] = [self.env.agent_state()]
        self.task_info["action_names"] = self.action_names()
        self.num_moves_made = 0
        self.last_missing_action_made = False
        self.num_missing_action_made = 0
        self.num_action_made = 0

    @property
    def action_space(self):
        return gym.spaces.Discrete(len(self._actions))

    def reached_terminal_state(self) -> bool:
        return self._took_end_action

    @classmethod
    def class_action_names(cls, **kwargs) -> Tuple[str, ...]:
        return cls._actions

    def close(self) -> None:
        self.env.stop()

    def _step(self, action: Union[int, Sequence[int]]) -> RLStepResult:

        assert isinstance(action, int)
        action = cast(int, action)

        if action != self.task_info["missing_action"]:
            action_str = self.action_names()[action]

            if action_str == END:
                self._took_end_action = True
                self._success = self._is_goal_in_range()
                self.last_action_success = self._success
            elif action_str in [LOOK_UP, LOOK_DOWN]:
                self.env.step({"action": action_str})
                self.last_action_success = self.env.last_action_success
            else:
                self.env.step({"action": action_str})
                self.last_action_success = self.env.last_action_success
                pose = self.env.agent_state()
                self.path.append({k: pose[k] for k in ["x", "y", "z"]})
                self.task_info["followed_path"].append(pose)
        else:
            self.last_action_success = False
            self.last_missing_action_made = True
            self.num_missing_action_made += 1
        self.num_action_made += 1

        if len(self.path) > 1 and self.path[-1] != self.path[-2]:
            self.num_moves_made += 1
        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=self.judge(),
            done=self.is_done(),
            info={"last_action_success": self.last_action_success, "action": action},
        )

        if False:
            tgts = []
            for i in range(10):
                tgt = [self.task_info["target"]["x"], i * 0.1, self.task_info["target"]["z"]]
                tgts.append(tgt)
            tgts = torch.Tensor(tgts).unsqueeze(0)
            tgts_2d = project_3d_points_to_2d([self.env.last_event.metadata], tgts)
            tgts_2d = tgts_2d.squeeze(0).numpy()
            tgts_2d = np.array([[y, x] for (x, y) in tgts_2d])

            self.env.counter += 1
            #method_name = "rgbd_kp_1xNPM"
            method_name = "rgbd"
            folder = "qualitative_results/{}/{}/{}".format(method_name,
                                                           self.task_info["scene"],
                                                           self.task_info["id"])
            if not os.path.isdir("{}".format(folder)):
                os.makedirs("{}".format(folder))
            img = Image.fromarray(self.env.current_frame, "RGB")
            img = draw_point(img, tgts_2d)
            img.save("{}/{:05}.png".format(folder, self.env.counter))

        return step_result

    def render(self, mode: str = "rgb", *args, **kwargs) -> np.ndarray:
        assert mode in ["rgb", "depth"], "only rgb and depth rendering is implemented"
        if mode == "rgb":
            return self.env.current_frame
        elif mode == "depth":
            return self.env.current_depth

    def _is_goal_in_range(self) -> Optional[bool]:
        tget = self.task_info["target"]
        dist = self.dist_to_target()

        if -0.5 < dist <= 0.2:
            return True
        elif dist > 0.2:
            return False
        else:
            get_logger().debug(
                "No path for {} from {} to {}".format(
                    self.env.scene_name, self.env.agent_state(), tget
                )
            )
            return None

    def shaping(self) -> float:
        rew = 0.0

        if self.reward_configs["shaping_weight"] == 0.0:
            return rew

        geodesic_distance = self.dist_to_target()

        if geodesic_distance == -1.0:
            geodesic_distance = self.last_geodesic_distance
        if (
                self.last_geodesic_distance > -0.5 and geodesic_distance > -0.5
        ):  # (robothor limits)
            rew += self.last_geodesic_distance - geodesic_distance
        self.last_geodesic_distance = geodesic_distance

        return rew * self.reward_configs["shaping_weight"]

    def shaping_by_missing_action(self, penalty) -> float:
        if self.last_missing_action_made:
            self.last_missing_action_made = False
            return penalty
        else:
            return 0

    def judge(self) -> float:
        """Judge the last event."""
        reward = self.reward_configs["step_penalty"]

        reward += self.shaping()
        if "missing_action_penalty" in self.reward_configs.keys():
            reward += self.shaping_by_missing_action(self.reward_configs["missing_action_penalty"])

        if self._took_end_action:
            if self._success is not None:
                reward += (
                    self.reward_configs["goal_success_reward"]
                    if self._success
                    else self.reward_configs["failed_stop_reward"]
                )

        self._rewards.append(float(reward))
        return float(reward)

    def spl(self):
        if not self._success:
            return 0.0
        li = self.optimal_distance
        pi = self.num_moves_made * self.env._grid_size
        res = li / (max(pi, li) + 1e-8)
        return res

    def dist_to_target(self):
        return self.env.distance_to_point(self.task_info["target"])

    def metrics(self) -> Dict[str, Any]:
        if not self.is_done():
            return {}

        total_reward = float(np.sum(self._rewards))
        self._rewards = []

        if self._success is None:
            return {}

        dist2tget = self.dist_to_target()
        spl = self.spl()

        return {
            **super(PointNavMissingActionTask, self).metrics(),
            "success": self._success,  # False also if no path to target
            "total_reward": total_reward,
            "dist_to_target": dist2tget,
            "spl": spl,
            "target_in_reachable_points": self.last_tget_in_path,
            "num_missing_action": self.num_missing_action_made,
            "missing_action_ratio": self.num_missing_action_made / float(self.num_action_made),
        }

    def query_expert(self, end_action_only: bool = False, **kwargs) -> Tuple[int, bool]:
        if self._is_goal_in_range():
            return self.class_action_names().index(END), True
        if end_action_only:
            return 0, False
        else:
            raise NotImplementedError

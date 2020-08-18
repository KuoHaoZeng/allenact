import copy
import random
from typing import List, Dict, Optional, Any, Union

import gym

from plugins.ithor_plugin.ithor_environment import IThorEnvironment, IThorObjManipEnvironment
from plugins.ithor_plugin.ithor_tasks import ObjectNavTask, ObjectManipTask
from core.base_abstractions.sensor import Sensor
from core.base_abstractions.task import TaskSampler
from utils.experiment_utils import set_deterministic_cudnn, set_seed
from utils.system import get_logger


class ObjectNavTaskSampler(TaskSampler):
    def __init__(
        self,
        scenes: List[str],
        object_types: str,
        sensors: List[Sensor],
        max_steps: int,
        env_args: Dict[str, Any],
        action_space: gym.Space,
        scene_period: Optional[Union[int, str]] = None,
        max_tasks: Optional[int] = None,
        seed: Optional[int] = None,
        deterministic_cudnn: bool = False,
        fixed_tasks: Optional[List[Dict[str, Any]]] = None,
        *args,
        **kwargs
    ) -> None:
        self.env_args = env_args
        self.scenes = scenes
        self.object_types = object_types
        self.grid_size = 0.25
        self.env: Optional[IThorEnvironment] = None
        self.sensors = sensors
        self.max_steps = max_steps
        self._action_space = action_space

        self.scene_counter: Optional[int] = None
        self.scene_order: Optional[List[str]] = None
        self.scene_id: Optional[int] = None
        self.scene_period: Optional[
            Union[str, int]
        ] = scene_period  # default makes a random choice
        self.max_tasks: Optional[int] = None
        self.reset_tasks = max_tasks

        self._last_sampled_task: Optional[ObjectNavTask] = None

        self.seed: Optional[int] = None
        self.set_seed(seed)

        if deterministic_cudnn:
            set_deterministic_cudnn()

        self.reset()

    def _create_environment(self) -> IThorEnvironment:
        env = IThorEnvironment(
            make_agents_visible=False,
            object_open_speed=0.05,
            restrict_to_initially_reachable_points=True,
            **self.env_args,
        )
        return env

    @property
    def length(self) -> Union[int, float]:
        """Length.

        # Returns

        Number of total tasks remaining that can be sampled. Can be float('inf').
        """
        return float("inf") if self.max_tasks is None else self.max_tasks

    @property
    def total_unique(self) -> Optional[Union[int, float]]:
        return None

    @property
    def last_sampled_task(self) -> Optional[ObjectNavTask]:
        return self._last_sampled_task

    def close(self) -> None:
        if self.env is not None:
            self.env.stop()

    @property
    def all_observation_spaces_equal(self) -> bool:
        """Check if observation spaces equal.

        # Returns

        True if all Tasks that can be sampled by this sampler have the
            same observation space. Otherwise False.
        """
        return True

    def sample_scene(self, force_advance_scene: bool):
        if force_advance_scene:
            if self.scene_period != "manual":
                get_logger().warning(
                    "When sampling scene, have `force_advance_scene == True`"
                    "but `self.scene_period` is not equal to 'manual',"
                    "this may cause unexpected behavior."
                )
            self.scene_id = (1 + self.scene_id) % len(self.scenes)
            if self.scene_id == 0:
                random.shuffle(self.scene_order)

        if self.scene_period is None:
            # Random scene
            self.scene_id = random.randint(0, len(self.scenes) - 1)
        elif self.scene_period == "manual":
            pass
        elif self.scene_counter == self.scene_period:
            if self.scene_id == len(self.scene_order) - 1:
                # Randomize scene order for next iteration
                random.shuffle(self.scene_order)
                # Move to next scene
                self.scene_id = 0
            else:
                # Move to next scene
                self.scene_id += 1
            # Reset scene counter
            self.scene_counter = 1
        elif isinstance(self.scene_period, int):
            # Stay in current scene
            self.scene_counter += 1
        else:
            raise NotImplementedError(
                "Invalid scene_period {}".format(self.scene_period)
            )

        if self.max_tasks is not None:
            self.max_tasks -= 1

        return self.scenes[int(self.scene_order[self.scene_id])]

    def next_task(self, force_advance_scene: bool = False) -> Optional[ObjectNavTask]:
        if self.max_tasks is not None and self.max_tasks <= 0:
            return None

        scene = self.sample_scene(force_advance_scene)

        if self.env is not None:
            if scene.replace("_physics", "") != self.env.scene_name.replace(
                "_physics", ""
            ):
                self.env.reset(scene)
        else:
            self.env = self._create_environment()
            self.env.reset(scene_name=scene)

        pose = self.env.randomize_agent_location()

        object_types_in_scene = set(
            [o["objectType"] for o in self.env.last_event.metadata["objects"]]
        )

        task_info: Dict[str, Any] = {}
        for ot in random.sample(self.object_types, len(self.object_types)):
            if ot in object_types_in_scene:
                task_info["object_type"] = ot
                break

        if len(task_info) == 0:
            get_logger().warning(
                "Scene {} does not contain any"
                " objects of any of the types {}.".format(scene, self.object_types)
            )

        task_info["start_pose"] = copy.copy(pose)

        self._last_sampled_task = ObjectNavTask(
            env=self.env,
            sensors=self.sensors,
            task_info=task_info,
            max_steps=self.max_steps,
            action_space=self._action_space,
        )
        return self._last_sampled_task

    def reset(self):
        self.scene_counter = 0
        self.scene_order = list(range(len(self.scenes)))
        random.shuffle(self.scene_order)
        self.scene_id = 0
        self.max_tasks = self.reset_tasks

    def set_seed(self, seed: int):
        self.seed = seed
        if seed is not None:
            set_seed(seed)


class ObjectManipTaskSampler(TaskSampler):
    """
    """
    def __init__(
            self,
            scenes: List[str],
            sensors: List[Sensor],
            max_steps: int,
            env_args: Dict[str, Any],
            action_space: gym.Space,
            scene_period: Optional[Union[int, str]] = None,
            max_tasks: Optional[int] = None,
            seed: Optional[int] = None,
            deterministic_cudnn: bool = False,
            fixed_tasks: Optional[List[Dict[str, Any]]] = None,
            *args,
            **kwargs
    ) -> None:
        self.env_args = env_args
        self.scenes = scenes
        self.grid_size = 0.25
        self.env: Optional[IThorObjManipEnvironment] = None
        self.sensors = sensors
        self.max_steps = max_steps
        self._action_space = action_space

        self.scene_counter: Optional[int] = None
        self.scene_order: Optional[List[str]] = None
        self.scene_id: Optional[int] = None
        self.scene_period: Optional[
            Union[str, int]
        ] = scene_period  # default makes a random choice
        self.max_tasks: Optional[int] = None
        self.reset_tasks = max_tasks

        self._last_sampled_task: Optional[ObjectManipTask] = None

        self.seed: Optional[int] = None
        self.set_seed(seed)

        if deterministic_cudnn:
            set_deterministic_cudnn()

        self.reset()

    def _create_environment(self) -> IThorObjManipEnvironment:
        env = IThorObjManipEnvironment(
            make_agents_visible=False,
            object_open_speed=0.05,
            # restrict_to_initially_reachable_points=True, #TODO is this really important?
            **self.env_args,
        )

        return env

    @property
    def length(self) -> Union[int, float]:
        """Length.

        # Returns

        Number of total tasks remaining that can be sampled. Can be float('inf').
        """
        return float("inf") if self.max_tasks is None else self.max_tasks

    @property
    def total_unique(self) -> Optional[Union[int, float]]:
        return None

    @property
    def last_sampled_task(self) -> Optional[ObjectManipTask]:
        return self._last_sampled_task

    def close(self) -> None:
        if self.env is not None:
            self.env.stop()

    @property
    def all_observation_spaces_equal(self) -> bool:
        """Check if observation spaces equal.

        # Returns

        True if all Tasks that can be sampled by this sampler have the
            same observation space. Otherwise False.
        """
        return True

    def sample_scene(self, force_advance_scene: bool):
        if force_advance_scene:
            if self.scene_period != "manual":
                get_logger().warning(
                    "When sampling scene, have `force_advance_scene == True`"
                    "but `self.scene_period` is not equal to 'manual',"
                    "this may cause unexpected behavior."
                )
            self.scene_id = (1 + self.scene_id) % len(self.scenes)
            if self.scene_id == 0:
                random.shuffle(self.scene_order)

        if self.scene_period is None:
            # Random scene
            self.scene_id = random.randint(0, len(self.scenes) - 1)
        elif self.scene_period == "manual":
            pass
        elif self.scene_counter == self.scene_period:
            if self.scene_id == len(self.scene_order) - 1:
                # Randomize scene order for next iteration
                random.shuffle(self.scene_order)
                # Move to next scene
                self.scene_id = 0
            else:
                # Move to next scene
                self.scene_id += 1
            # Reset scene counter
            self.scene_counter = 1
        elif isinstance(self.scene_period, int):
            # Stay in current scene
            self.scene_counter += 1
        else:
            raise NotImplementedError(
                "Invalid scene_period {}".format(self.scene_period)
            )

        if self.max_tasks is not None:
            self.max_tasks -= 1

        return self.scenes[int(self.scene_order[self.scene_id])]

    def next_task(self, force_advance_scene: bool = False) -> Optional[ObjectManipTask]:
        if self.max_tasks is not None and self.max_tasks <= 0:
            return None


        scene = self.sample_scene(force_advance_scene)

        if self.env is not None:
            if scene.replace("_physics", "") != self.env.scene_name.replace(
                    "_physics", ""
            ):
                self.env.reset(scene)
        else:
            self.env = self._create_environment()
            self.env.reset(scene_name=scene)

        #TODO later on we might wanna just pick one pickupable object instead of visible and pickupable one
        
        self.env.controller.step(action='MoveMidLevelArm', position=dict(x=0, y=1.2, z=1), speed = 1.0, returnToStart = False, handCameraSpace = False)
        #TODO this is not optimized at all
        finallyPickedUp = False
        while(True):
            pose = self.env.randomize_agent_location(seed=0) #TODO change this seed
            pickupable_object_in_scene = [o for o in self.env.last_event.metadata["objects"] if o['visible'] and o['pickupable']]
            if len(pickupable_object_in_scene) > 0:
                task_info: Dict[str, Any] = {}
                # random.shuffle(pickupable_object_in_scene) #TODO put this back
                for object_info in pickupable_object_in_scene:
                    task_info['objectId'] = object_info['objectId']

                    #TODO only for now, you can change this as an action later
                    event = self.env.controller.step('PickupObject', objectId=object_info['objectId'],manualInteract=True)
                    if event.metadata['lastActionSuccess']:
                        finallyPickedUp = True
                        break
                    else:
                        print('Failed to pickupable', task_info['objectId'])
                    # assert event.metadata['lastActionSuccess'], 'Pick up failed'; ForkedPdb().set_trace()
                if finallyPickedUp:
                    break

        #TODO remove this
        print(task_info['objectId'])

        position = object_info['position']
        rotation = object_info['rotation']

        #TODO this should be changed lagter, too simple or maybe not possible at all
        goal_position = copy.copy(position)
        goal_rotation = copy.copy(rotation)
        goal_position['x'] += 0.1
        goal_position['y'] += 0.1
        goal_position['z'] += 0.1 #TODO all these might not be feasile at all

        task_info["start_pose"] = copy.copy(pose)

        task_info['init_obj_state'] = dict(position=position, rotation=rotation)
        task_info['goal_obj_state'] = dict(position=goal_position, rotation=goal_rotation)


        self._last_sampled_task = ObjectManipTask(
            env=self.env,
            sensors=self.sensors,
            task_info=task_info,
            max_steps=self.max_steps,
            action_space=self._action_space,
        )
        return self._last_sampled_task

    def reset(self):
        self.scene_counter = 0
        self.scene_order = list(range(len(self.scenes)))
        random.shuffle(self.scene_order)
        self.scene_id = 0
        self.max_tasks = self.reset_tasks

    def set_seed(self, seed: int):
        self.seed = seed
        if seed is not None:
            set_seed(seed)
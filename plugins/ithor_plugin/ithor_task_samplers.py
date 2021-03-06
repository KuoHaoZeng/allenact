import copy
import random
import gzip
import json
from typing import List, Dict, Optional, Any, Union, cast, Tuple

import gym

from core.base_abstractions.sensor import Sensor
from core.base_abstractions.task import TaskSampler
from plugins.ithor_plugin.ithor_environment import IThorEnvironment
from plugins.ithor_plugin.ithor_tasks import ObjectNavTask, PointNavTask, PointNavObstaclesTask, PlacementTask,\
    PointNavObstaclesMissingActionTask, PointNavMissingActionTask, PointNavDynamicsCorruptionTask
from utils.cache_utils import str_to_pos_for_cache
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
        **kwargs,
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
        elif self.scene_counter >= cast(int, self.scene_period):
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


class ObjectNavDatasetTaskSampler(TaskSampler):
    def __init__(
            self,
            scenes: List[str],
            scene_directory: str,
            sensors: List[Sensor],
            max_steps: int,
            env_args: Dict[str, Any],
            action_space: gym.Space,
            rewards_config: Dict,
            seed: Optional[int] = None,
            deterministic_cudnn: bool = False,
            loop_dataset: bool = True,
            allow_flipping=False,
            env_class=IThorEnvironment,
            **kwargs,
    ) -> None:
        self.rewards_config = rewards_config
        self.env_args = env_args
        self.scenes = scenes
        self.episodes = {
            scene: ObjectNavDatasetTaskSampler.load_dataset(
                scene, scene_directory + "/episodes"
            )
            for scene in scenes
        }
        self.env_class = env_class
        self.object_types = [
            ep["object_type"] for scene in self.episodes for ep in self.episodes[scene]
        ]
        self.env: Optional[IThorEnvironment] = None
        self.sensors = sensors
        self.max_steps = max_steps
        self._action_space = action_space
        self.allow_flipping = allow_flipping
        self.scene_counter: Optional[int] = None
        self.scene_order: Optional[List[str]] = None
        self.scene_id: Optional[int] = None
        # get the total number of tasks assigned to this process
        if loop_dataset:
            self.max_tasks = None
        else:
            self.max_tasks = sum(len(self.episodes[scene]) for scene in self.episodes)
        self.reset_tasks = self.max_tasks
        self.scene_index = 0
        self.episode_index = 0

        self._last_sampled_task: Optional[ObjectNavTask] = None

        self.seed: Optional[int] = None
        self.set_seed(seed)

        if deterministic_cudnn:
            set_deterministic_cudnn()

        self.reset()

    def _create_environment(self) -> IThorEnvironment:
        env = self.env_class(**self.env_args)
        return env

    @staticmethod
    def load_dataset(scene: str, base_directory: str) -> List[Dict]:
        filename = (
            "/".join([base_directory, scene])
            if base_directory[-1] != "/"
            else "".join([base_directory, scene])
        )
        filename += ".json.gz"
        fin = gzip.GzipFile(filename, "r")
        json_bytes = fin.read()
        fin.close()
        json_str = json_bytes.decode("utf-8")
        data = json.loads(json_str)
        random.shuffle(data)
        return data

    @staticmethod
    def load_distance_cache_from_file(scene: str, base_directory: str) -> Dict:
        filename = (
            "/".join([base_directory, scene])
            if base_directory[-1] != "/"
            else "".join([base_directory, scene])
        )
        filename += ".json.gz"
        fin = gzip.GzipFile(filename, "r")
        json_bytes = fin.read()
        fin.close()
        json_str = json_bytes.decode("utf-8")
        data = json.loads(json_str)
        return data

    @property
    def __len__(self) -> Union[int, float]:
        """Length.

        # Returns

        Number of total tasks remaining that can be sampled. Can be float('inf').
        """
        return float("inf") if self.max_tasks is None else self.max_tasks

    @property
    def total_unique(self) -> Optional[Union[int, float]]:
        return self.reset_tasks

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

    @property
    def length(self) -> Union[int, float]:
        """Length.

        # Returns

        Number of total tasks remaining that can be sampled. Can be float('inf').
        """
        return float("inf") if self.max_tasks is None else self.max_tasks

    def next_task(self, force_advance_scene: bool = False) -> Optional[ObjectNavTask]:
        if self.max_tasks is not None and self.max_tasks <= 0:
            return None

        if self.episode_index >= len(self.episodes[self.scenes[self.scene_index]]):
            self.scene_index = (self.scene_index + 1) % len(self.scenes)
            # shuffle the new list of episodes to train on
            random.shuffle(self.episodes[self.scenes[self.scene_index]])
            self.episode_index = 0
        scene = self.scenes[self.scene_index]
        episode = self.episodes[scene][self.episode_index]
        if self.env is not None:
            if scene.replace("_physics", "") != self.env.scene_name.replace(
                    "_physics", ""
            ):
                self.env.reset(
                    scene_name=scene,
                    filtered_objects=list(
                        set([e["object_id"] for e in self.episodes[scene]])
                    ),
                )
        else:
            self.env = self._create_environment()
            self.env.reset(
                scene_name=scene,
                filtered_objects=list(
                    set([e["object_id"] for e in self.episodes[scene]])
                ),
            )
        task_info = {"scene": scene, "object_type": episode["object_type"]}
        if len(task_info) == 0:
            get_logger().warning(
                "Scene {} does not contain any"
                " objects of any of the types {}.".format(scene, self.object_types)
            )
        task_info["initial_position"] = episode["initial_position"]
        task_info["initial_orientation"] = episode["initial_orientation"]
        task_info["distance_to_target"] = episode["shortest_path_length"]
        task_info["path_to_target"] = episode["shortest_path"]
        task_info["object_type"] = episode["object_type"]
        task_info["id"] = episode["id"]
        if self.allow_flipping and random.random() > 0.5:
            task_info["mirrored"] = True
        else:
            task_info["mirrored"] = False

        self.episode_index += 1
        if self.max_tasks is not None:
            self.max_tasks -= 1
        if not self.env.teleport(
                episode["initial_position"], episode["initial_orientation"]
        ):
            return self.next_task()
        self._last_sampled_task = ObjectNavTask(
            env=self.env,
            sensors=self.sensors,
            task_info=task_info,
            max_steps=self.max_steps,
            action_space=self._action_space,
            reward_configs=self.rewards_config,
        )
        return self._last_sampled_task

    def reset(self):
        self.episode_index = 0
        self.scene_index = 0
        self.max_tasks = self.reset_tasks

    def set_seed(self, seed: int):
        self.seed = seed
        if seed is not None:
            set_seed(seed)


class PointNavDatasetTaskSampler(TaskSampler):
    def __init__(
            self,
            scenes: List[str],
            scene_directory: str,
            sensors: List[Sensor],
            max_steps: int,
            env_args: Dict[str, Any],
            action_space: gym.Space,
            rewards_config: Dict,
            seed: Optional[int] = None,
            deterministic_cudnn: bool = False,
            loop_dataset: bool = True,
            shuffle_dataset: bool = True,
            allow_flipping=False,
            env_class=IThorEnvironment,
            **kwargs,
    ) -> None:
        self.rewards_config = rewards_config
        self.env_args = env_args
        self.scenes = scenes
        self.shuffle_dataset: bool = shuffle_dataset
        self.episodes = {
            scene: ObjectNavDatasetTaskSampler.load_dataset(
                scene, scene_directory + "/episodes"
            )
            for scene in scenes
        }
        self.env_class = env_class
        self.env: Optional[IThorEnvironment] = None
        self.sensors = sensors
        self.max_steps = max_steps
        self._action_space = action_space
        self.allow_flipping = allow_flipping
        self.scene_counter: Optional[int] = None
        self.scene_order: Optional[List[str]] = None
        self.scene_id: Optional[int] = None
        # get the total number of tasks assigned to this process
        if loop_dataset:
            self.max_tasks = None
        else:
            self.max_tasks = sum(len(self.episodes[scene]) for scene in self.episodes)
        self.reset_tasks = self.max_tasks
        self.scene_index = 0
        self.episode_index = 0

        self._last_sampled_task: Optional[PointNavTask] = None

        self.seed: Optional[int] = None
        self.set_seed(seed)

        if deterministic_cudnn:
            set_deterministic_cudnn()

        self.reset()

    def _create_environment(self) -> IThorEnvironment:
        env = self.env_class(**self.env_args)
        return env

    @property
    def __len__(self) -> Union[int, float]:
        """Length.

        # Returns

        Number of total tasks remaining that can be sampled. Can be float('inf').
        """
        return float("inf") if self.max_tasks is None else self.max_tasks

    @property
    def total_unique(self) -> Optional[Union[int, float]]:
        return self.reset_tasks

    @property
    def last_sampled_task(self) -> Optional[PointNavTask]:
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

    def next_task(self, force_advance_scene: bool = False) -> Optional[PointNavTask]:
        if self.max_tasks is not None and self.max_tasks <= 0:
            return None

        if self.episode_index >= len(self.episodes[self.scenes[self.scene_index]]):
            self.scene_index = (self.scene_index + 1) % len(self.scenes)
            # shuffle the new list of episodes to train on
            if self.shuffle_dataset:
                random.shuffle(self.episodes[self.scenes[self.scene_index]])
            self.episode_index = 0

        scene = self.scenes[self.scene_index]
        episode = self.episodes[scene][self.episode_index]
        if self.env is not None:
            if scene.replace("_physics", "") != self.env.scene_name.replace(
                    "_physics", ""
            ):
                self.env.reset(scene_name=scene, filtered_objects=[])
        else:
            self.env = self._create_environment()
            self.env.reset(scene_name=scene, filtered_objects=[])

        def to_pos(s):
            if isinstance(s, (Dict, Tuple)):
                return s
            if isinstance(s, float):
                return {"x": 0, "y": s, "z": 0}
            return str_to_pos_for_cache(s)

        for k in ["initial_position", "initial_orientation", "target_position"]:
            episode[k] = to_pos(episode[k])

        task_info = {
            "scene": scene,
            "initial_position": episode["initial_position"],
            "initial_orientation": episode["initial_orientation"],
            "target": episode["target_position"],
            "shortest_path": episode["shortest_path"],
            "distance_to_target": episode["shortest_path_length"],
            "id": episode["id"],
        }

        if self.allow_flipping and random.random() > 0.5:
            task_info["mirrored"] = True
        else:
            task_info["mirrored"] = False

        self.episode_index += 1
        if self.max_tasks is not None:
            self.max_tasks -= 1

        if not self.env.teleport(
                pose=episode["initial_position"], rotation=episode["initial_orientation"]
        ):
            return self.next_task()

        self._last_sampled_task = PointNavTask(
            env=self.env,
            sensors=self.sensors,
            task_info=task_info,
            max_steps=self.max_steps,
            action_space=self._action_space,
            reward_configs=self.rewards_config,
        )

        return self._last_sampled_task

    def reset(self):
        self.episode_index = 0
        self.scene_index = 0
        self.max_tasks = self.reset_tasks

    def set_seed(self, seed: int):
        self.seed = seed
        if seed is not None:
            set_seed(seed)

    @property
    def length(self) -> Union[int, float]:
        """Length.

        # Returns

        Number of total tasks remaining that can be sampled.
        Can be float('inf').
        """
        return float("inf") if self.max_tasks is None else self.max_tasks


class PointNavObstaclesDatasetTaskSampler(TaskSampler):
    def __init__(
            self,
            scenes: List[str],
            scene_directory: str,
            sensors: List[Sensor],
            max_steps: int,
            env_args: Dict[str, Any],
            action_space: gym.Space,
            rewards_config: Dict,
            seed: Optional[int] = None,
            deterministic_cudnn: bool = False,
            loop_dataset: bool = True,
            shuffle_dataset: bool = True,
            allow_flipping=False,
            env_class=IThorEnvironment,
            **kwargs,
    ) -> None:
        self.rewards_config = rewards_config
        self.env_args = env_args
        self.scenes = scenes
        self.shuffle_dataset: bool = shuffle_dataset
        self.episodes = {
            scene: ObjectNavDatasetTaskSampler.load_dataset(
                scene, scene_directory + "/episodes"
            )
            for scene in scenes
        }
        self.env_class = env_class
        self.env: Optional[IThorEnvironment] = None
        self.sensors = sensors
        self.max_steps = max_steps
        self._action_space = action_space
        self.allow_flipping = allow_flipping
        self.scene_counter: Optional[int] = None
        self.scene_order: Optional[List[str]] = None
        self.scene_id: Optional[int] = None
        # get the total number of tasks assigned to this process
        if loop_dataset:
            self.max_tasks = None
        else:
            self.max_tasks = sum(len(self.episodes[scene]) for scene in self.episodes)
        self.reset_tasks = self.max_tasks
        self.scene_index = 0
        self.episode_index = 0

        self._last_sampled_task: Optional[PointNavObstaclesTask] = None

        self.seed: Optional[int] = None
        self.set_seed(seed)

        if deterministic_cudnn:
            set_deterministic_cudnn()

        self.reset()

    def _create_environment(self) -> IThorEnvironment:
        env = self.env_class(**self.env_args)
        return env

    @property
    def __len__(self) -> Union[int, float]:
        """Length.

        # Returns

        Number of total tasks remaining that can be sampled. Can be float('inf').
        """
        return float("inf") if self.max_tasks is None else self.max_tasks

    @property
    def total_unique(self) -> Optional[Union[int, float]]:
        return self.reset_tasks

    @property
    def last_sampled_task(self) -> Optional[PointNavObstaclesTask]:
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

    def next_task(self, force_advance_scene: bool = False) -> Optional[PointNavObstaclesTask]:
        if self.max_tasks is not None and self.max_tasks <= 0:
            return None

        if self.episode_index >= len(self.episodes[self.scenes[self.scene_index]]):
            self.scene_index = (self.scene_index + 1) % len(self.scenes)
            # shuffle the new list of episodes to train on
            if self.shuffle_dataset:
                random.shuffle(self.episodes[self.scenes[self.scene_index]])
            self.episode_index = 0

        scene = self.scenes[self.scene_index]
        episode = self.episodes[scene][self.episode_index]
        if self.env is not None:
            #if scene.replace("_physics", "") != self.env.scene_name.replace(
            #        "_physics", ""
            #):
            self.env.reset(scene_name=scene, filtered_objects=[])
        else:
            self.env = self._create_environment()
            self.env.reset(scene_name=scene, filtered_objects=[])

        def to_pos(s):
            if isinstance(s, (Dict, Tuple)):
                return s
            if isinstance(s, float):
                return {"x": 0, "y": s, "z": 0}
            return str_to_pos_for_cache(s)

        for k in ["initial_position", "initial_orientation", "target_position"]:
            episode[k] = to_pos(episode[k])

        task_info = {
            "scene": scene,
            "initial_position": episode["initial_position"],
            "initial_orientation": episode["initial_orientation"],
            "target": episode["target_position"],
            "shortest_path": episode["shortest_path"],
            "distance_to_target": episode["shortest_path_length"],
            "id": episode["id"],
            "spawn_objects": episode["spawn_objects"],
            "obstacles_types": episode["obstacles_types"],
        }

        if self.allow_flipping and random.random() > 0.5:
            task_info["mirrored"] = True
        else:
            task_info["mirrored"] = False

        self.episode_index += 1
        if self.max_tasks is not None:
            self.max_tasks -= 1

        if not self.env.teleport(
                pose=episode["initial_position"], rotation=episode["initial_orientation"]
        ):
            return self.next_task()

        for obj in episode["spawn_objects"]:
            if not self.env.spawn_obj(obj):
                return self.next_task()

        self.env.initialize(0.25, renderObjectImage=True, renderDepthImage=True)

        self._last_sampled_task = PointNavObstaclesTask(
            env=self.env,
            sensors=self.sensors,
            task_info=task_info,
            max_steps=self.max_steps,
            action_space=self._action_space,
            reward_configs=self.rewards_config,
        )

        return self._last_sampled_task

    def reset(self):
        self.episode_index = 0
        self.scene_index = 0
        self.max_tasks = self.reset_tasks

    def set_seed(self, seed: int):
        self.seed = seed
        if seed is not None:
            set_seed(seed)

    @property
    def length(self) -> Union[int, float]:
        """Length.

        # Returns

        Number of total tasks remaining that can be sampled.
        Can be float('inf').
        """
        return float("inf") if self.max_tasks is None else self.max_tasks

class PlacementDatasetTaskSampler(TaskSampler):
    def __init__(
            self,
            scenes: List[str],
            scene_directory: str,
            sensors: List[Sensor],
            max_steps: int,
            env_args: Dict[str, Any],
            action_space: gym.Space,
            rewards_config: Dict,
            seed: Optional[int] = None,
            deterministic_cudnn: bool = False,
            loop_dataset: bool = True,
            shuffle_dataset: bool = True,
            allow_flipping=False,
            env_class=IThorEnvironment,
            **kwargs,
    ) -> None:
        self.rewards_config = rewards_config
        self.env_args = env_args
        self.scenes = scenes
        self.shuffle_dataset: bool = shuffle_dataset
        self.episodes = {
            scene: ObjectNavDatasetTaskSampler.load_dataset(
                scene, scene_directory + "/episodes"
            )
            for scene in scenes
        }
        self.env_class = env_class
        self.env: Optional[IThorEnvironment] = None
        self.sensors = sensors
        self.max_steps = max_steps
        self._action_space = action_space
        self.allow_flipping = allow_flipping
        self.scene_counter: Optional[int] = None
        self.scene_order: Optional[List[str]] = None
        self.scene_id: Optional[int] = None
        # get the total number of tasks assigned to this process
        if loop_dataset:
            self.max_tasks = None
        else:
            self.max_tasks = sum(len(self.episodes[scene]) for scene in self.episodes)
        self.reset_tasks = self.max_tasks
        self.scene_index = 0
        self.episode_index = 0

        self._last_sampled_task: Optional[PlacementTask] = None

        self.seed: Optional[int] = None
        self.set_seed(seed)

        if deterministic_cudnn:
            set_deterministic_cudnn()

        self.reset()

    def _create_environment(self) -> IThorEnvironment:
        env = self.env_class(**self.env_args)
        return env

    @property
    def __len__(self) -> Union[int, float]:
        """Length.

        # Returns

        Number of total tasks remaining that can be sampled. Can be float('inf').
        """
        return float("inf") if self.max_tasks is None else self.max_tasks

    @property
    def total_unique(self) -> Optional[Union[int, float]]:
        return self.reset_tasks

    @property
    def last_sampled_task(self) -> Optional[PlacementTask]:
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

    def next_task(self, force_advance_scene: bool = False) -> Optional[PlacementTask]:
        if self.max_tasks is not None and self.max_tasks <= 0:
            return None

        if self.episode_index >= len(self.episodes[self.scenes[self.scene_index]]):
            self.scene_index = (self.scene_index + 1) % len(self.scenes)
            # shuffle the new list of episodes to train on
            if self.shuffle_dataset:
                random.shuffle(self.episodes[self.scenes[self.scene_index]])
            self.episode_index = 0

        scene = self.scenes[self.scene_index]
        episode = self.episodes[scene][self.episode_index]
        if self.env is not None:
            #if scene.replace("_physics", "") != self.env.scene_name.replace(
            #        "_physics", ""
            #):
            self.env.reset(scene_name=scene, filtered_objects=[])
        else:
            self.env = self._create_environment()
            self.env.reset(scene_name=scene, filtered_objects=[])

        def to_pos(s):
            if isinstance(s, (Dict, Tuple)):
                return s
            if isinstance(s, float):
                return {"x": 0, "y": s, "z": 0}
            return str_to_pos_for_cache(s)

        for k in ["initial_position", "initial_orientation", "target_position"]:
            episode[k] = to_pos(episode[k])

        self.episode_index += 1
        if self.max_tasks is not None:
            self.max_tasks -= 1

        if not self.env.teleport(
                pose=episode["initial_position"],
                rotation=episode["initial_orientation"],
                horizon=episode["initial_horizon"]
        ):
            return self.next_task()

        for obj in episode["spawn_objects"]:
            if not "TargetCircle" in obj["objectType"]:
                if not self.env.spawn_obj(obj):
                    return self.next_task()
            else:
                if not self.env.spawn_target_circle(obj["random_seed"]):
                    return self.next_task()

        self.env.initialize(0.25, renderObjectImage=True, renderDepthImage=True)

        task_info = {
            "scene": scene,
            "initial_position": episode["initial_position"],
            "initial_orientation": episode["initial_orientation"],
            "target": episode["target_position"],
            "shortest_path": episode["shortest_path"],
            "distance_to_target": episode["shortest_path_length"],
            "id": episode["id"],
            "spawn_objects": episode["spawn_objects"],
            "obstacles_types": episode["obstacles_types"],
            "target_type": "TargetCircle",
            "object_type": episode["spawn_objects"][1]["objectType"],
        }

        if self.allow_flipping and random.random() > 0.5:
            task_info["mirrored"] = True
        else:
            task_info["mirrored"] = False

        self._last_sampled_task = PlacementTask(
            env=self.env,
            sensors=self.sensors,
            task_info=task_info,
            max_steps=self.max_steps,
            action_space=self._action_space,
            reward_configs=self.rewards_config,
        )

        return self._last_sampled_task

    def reset(self):
        self.episode_index = 0
        self.scene_index = 0
        self.max_tasks = self.reset_tasks

    def set_seed(self, seed: int):
        self.seed = seed
        if seed is not None:
            set_seed(seed)

    @property
    def length(self) -> Union[int, float]:
        """Length.

        # Returns

        Number of total tasks remaining that can be sampled.
        Can be float('inf').
        """
        return float("inf") if self.max_tasks is None else self.max_tasks

class PointNavObstaclesMissingActionDatasetTaskSampler(TaskSampler):
    def __init__(
            self,
            scenes: List[str],
            scene_directory: str,
            sensors: List[Sensor],
            max_steps: int,
            env_args: Dict[str, Any],
            action_space: gym.Space,
            rewards_config: Dict,
            seed: Optional[int] = None,
            deterministic_cudnn: bool = False,
            loop_dataset: bool = True,
            shuffle_dataset: bool = True,
            allow_flipping=False,
            env_class=IThorEnvironment,
            **kwargs,
    ) -> None:
        self.rewards_config = rewards_config
        self.env_args = env_args
        self.scenes = scenes
        self.shuffle_dataset: bool = shuffle_dataset
        self.episodes = {
            scene: ObjectNavDatasetTaskSampler.load_dataset(
                scene, scene_directory + "/episodes"
            )
            for scene in scenes
        }
        self.env_class = env_class
        self.env: Optional[IThorEnvironment] = None
        self.sensors = sensors
        self.max_steps = max_steps
        self._action_space = action_space
        self.allow_flipping = allow_flipping
        self.scene_counter: Optional[int] = None
        self.scene_order: Optional[List[str]] = None
        self.scene_id: Optional[int] = None
        # get the total number of tasks assigned to this process
        if loop_dataset:
            self.max_tasks = None
        else:
            self.max_tasks = sum(len(self.episodes[scene]) for scene in self.episodes)
        self.reset_tasks = self.max_tasks
        self.scene_index = 0
        self.episode_index = 0

        self._last_sampled_task: Optional[PointNavObstaclesMissingActionTask] = None

        self.seed: Optional[int] = None
        self.set_seed(seed)

        if deterministic_cudnn:
            set_deterministic_cudnn()

        self.reset()

    def _create_environment(self) -> IThorEnvironment:
        env = self.env_class(**self.env_args)
        return env

    @property
    def __len__(self) -> Union[int, float]:
        """Length.

        # Returns

        Number of total tasks remaining that can be sampled. Can be float('inf').
        """
        return float("inf") if self.max_tasks is None else self.max_tasks

    @property
    def total_unique(self) -> Optional[Union[int, float]]:
        return self.reset_tasks

    @property
    def last_sampled_task(self) -> Optional[PointNavObstaclesMissingActionTask]:
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

    def next_task(self, force_advance_scene: bool = False) -> Optional[PointNavObstaclesMissingActionTask]:
        if self.max_tasks is not None and self.max_tasks <= 0:
            return None

        if self.episode_index >= len(self.episodes[self.scenes[self.scene_index]]):
            self.scene_index = (self.scene_index + 1) % len(self.scenes)
            # shuffle the new list of episodes to train on
            if self.shuffle_dataset:
                random.shuffle(self.episodes[self.scenes[self.scene_index]])
            self.episode_index = 0

        scene = self.scenes[self.scene_index]
        episode = self.episodes[scene][self.episode_index]
        if self.env is not None:
            #if scene.replace("_physics", "") != self.env.scene_name.replace(
            #        "_physics", ""
            #):
            self.env.reset(scene_name=scene, filtered_objects=[])
        else:
            self.env = self._create_environment()
            self.env.reset(scene_name=scene, filtered_objects=[])

        def to_pos(s):
            if isinstance(s, (Dict, Tuple)):
                return s
            if isinstance(s, float):
                return {"x": 0, "y": s, "z": 0}
            return str_to_pos_for_cache(s)

        for k in ["initial_position", "initial_orientation", "target_position"]:
            episode[k] = to_pos(episode[k])

        missing_action = random.choice([0, 1, 2, 3, 6, 7, 8, 9])

        task_info = {
            "scene": scene,
            "initial_position": episode["initial_position"],
            "initial_orientation": episode["initial_orientation"],
            "target": episode["target_position"],
            "shortest_path": episode["shortest_path"],
            "distance_to_target": episode["shortest_path_length"],
            "id": episode["id"],
            "spawn_objects": episode["spawn_objects"],
            "obstacles_types": episode["obstacles_types"],
            "missing_action": missing_action,
        }

        if self.allow_flipping and random.random() > 0.5:
            task_info["mirrored"] = True
        else:
            task_info["mirrored"] = False

        self.episode_index += 1
        if self.max_tasks is not None:
            self.max_tasks -= 1

        if not self.env.teleport(
                pose=episode["initial_position"], rotation=episode["initial_orientation"]
        ):
            return self.next_task()

        for obj in episode["spawn_objects"]:
            if not self.env.spawn_obj(obj):
                return self.next_task()

        self.env.initialize(0.25, renderObjectImage=True, renderDepthImage=True)

        self._last_sampled_task = PointNavObstaclesMissingActionTask(
            env=self.env,
            sensors=self.sensors,
            task_info=task_info,
            max_steps=self.max_steps,
            action_space=self._action_space,
            reward_configs=self.rewards_config,
        )

        return self._last_sampled_task

    def reset(self):
        self.episode_index = 0
        self.scene_index = 0
        self.max_tasks = self.reset_tasks

    def set_seed(self, seed: int):
        self.seed = seed
        if seed is not None:
            set_seed(seed)

    @property
    def length(self) -> Union[int, float]:
        """Length.

        # Returns

        Number of total tasks remaining that can be sampled.
        Can be float('inf').
        """
        return float("inf") if self.max_tasks is None else self.max_tasks


class PointNavMissingActionDatasetTaskSampler(TaskSampler):
    def __init__(
            self,
            scenes: List[str],
            scene_directory: str,
            sensors: List[Sensor],
            max_steps: int,
            env_args: Dict[str, Any],
            action_space: gym.Space,
            rewards_config: Dict,
            seed: Optional[int] = None,
            deterministic_cudnn: bool = False,
            loop_dataset: bool = True,
            shuffle_dataset: bool = True,
            allow_flipping=False,
            env_class=IThorEnvironment,
            **kwargs,
    ) -> None:
        self.rewards_config = rewards_config
        self.env_args = env_args
        self.scenes = scenes
        self.shuffle_dataset: bool = shuffle_dataset
        self.episodes = {
            scene: ObjectNavDatasetTaskSampler.load_dataset(
                scene, scene_directory + "/episodes"
            )
            for scene in scenes
        }
        self.env_class = env_class
        self.env: Optional[IThorEnvironment] = None
        self.sensors = sensors
        self.max_steps = max_steps
        self._action_space = action_space
        self.allow_flipping = allow_flipping
        self.scene_counter: Optional[int] = None
        self.scene_order: Optional[List[str]] = None
        self.scene_id: Optional[int] = None
        # get the total number of tasks assigned to this process
        if loop_dataset:
            self.max_tasks = None
        else:
            self.max_tasks = sum(len(self.episodes[scene]) for scene in self.episodes)
        self.reset_tasks = self.max_tasks
        self.scene_index = 0
        self.episode_index = 0

        self._last_sampled_task: Optional[PointNavMissingActionTask] = None

        self.seed: Optional[int] = None
        self.set_seed(seed)

        if deterministic_cudnn:
            set_deterministic_cudnn()

        self.reset()

    def _create_environment(self) -> IThorEnvironment:
        env = self.env_class(**self.env_args)
        return env

    @property
    def __len__(self) -> Union[int, float]:
        """Length.

        # Returns

        Number of total tasks remaining that can be sampled. Can be float('inf').
        """
        return float("inf") if self.max_tasks is None else self.max_tasks

    @property
    def total_unique(self) -> Optional[Union[int, float]]:
        return self.reset_tasks

    @property
    def last_sampled_task(self) -> Optional[PointNavMissingActionTask]:
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

    def next_task(self, force_advance_scene: bool = False) -> Optional[PointNavMissingActionTask]:
        if self.max_tasks is not None and self.max_tasks <= 0:
            return None

        if self.episode_index >= len(self.episodes[self.scenes[self.scene_index]]):
            self.scene_index = (self.scene_index + 1) % len(self.scenes)
            # shuffle the new list of episodes to train on
            if self.shuffle_dataset:
                random.shuffle(self.episodes[self.scenes[self.scene_index]])
            self.episode_index = 0

        scene = self.scenes[self.scene_index]
        episode = self.episodes[scene][self.episode_index]
        if self.env is not None:
            #if scene.replace("_physics", "") != self.env.scene_name.replace(
            #        "_physics", ""
            #):
            self.env.reset(scene_name=scene, filtered_objects=[])
        else:
            self.env = self._create_environment()
            self.env.reset(scene_name=scene, filtered_objects=[])

        def to_pos(s):
            if isinstance(s, (Dict, Tuple)):
                return s
            if isinstance(s, float):
                return {"x": 0, "y": s, "z": 0}
            return str_to_pos_for_cache(s)

        for k in ["initial_position", "initial_orientation", "target_position"]:
            episode[k] = to_pos(episode[k])

        missing_action = [random.choice([0, 1, 2, 3])]

        task_info = {
            "scene": scene,
            "initial_position": episode["initial_position"],
            "initial_orientation": episode["initial_orientation"],
            "target": episode["target_position"],
            "shortest_path": episode["shortest_path"],
            "distance_to_target": episode["shortest_path_length"],
            "id": episode["id"],
            "obstacles_types": episode["obstacles_types"],
            "missing_action": missing_action,
        }

        if self.allow_flipping and random.random() > 0.5:
            task_info["mirrored"] = True
        else:
            task_info["mirrored"] = False

        self.episode_index += 1
        if self.max_tasks is not None:
            self.max_tasks -= 1

        if not self.env.teleport(
                pose=episode["initial_position"], rotation=episode["initial_orientation"]
        ):
            return self.next_task()

        self.env.initialize(0.25, renderObjectImage=True, renderDepthImage=True)

        self._last_sampled_task = PointNavMissingActionTask(
            env=self.env,
            sensors=self.sensors,
            task_info=task_info,
            max_steps=self.max_steps,
            action_space=self._action_space,
            reward_configs=self.rewards_config,
        )

        return self._last_sampled_task

    def reset(self):
        self.episode_index = 0
        self.scene_index = 0
        self.max_tasks = self.reset_tasks

    def set_seed(self, seed: int):
        self.seed = seed
        if seed is not None:
            set_seed(seed)

    @property
    def length(self) -> Union[int, float]:
        """Length.

        # Returns

        Number of total tasks remaining that can be sampled.
        Can be float('inf').
        """
        return float("inf") if self.max_tasks is None else self.max_tasks


class PointNavDynamicsCorruptionDatasetTaskSampler(TaskSampler):
    def __init__(
            self,
            scenes: List[str],
            scene_directory: str,
            sensors: List[Sensor],
            max_steps: int,
            env_args: Dict[str, Any],
            action_space: gym.Space,
            rewards_config: Dict,
            seed: Optional[int] = None,
            deterministic_cudnn: bool = False,
            loop_dataset: bool = True,
            shuffle_dataset: bool = True,
            allow_flipping=False,
            env_class=IThorEnvironment,
            **kwargs,
    ) -> None:
        self.rewards_config = rewards_config
        self.env_args = env_args
        self.scenes = scenes
        self.shuffle_dataset: bool = shuffle_dataset
        self.episodes = {
            scene: ObjectNavDatasetTaskSampler.load_dataset(
                scene, scene_directory + "/episodes"
            )
            for scene in scenes
        }
        self.env_class = env_class
        self.env: Optional[IThorEnvironment] = None
        self.sensors = sensors
        self.max_steps = max_steps
        self._action_space = action_space
        self.allow_flipping = allow_flipping
        self.scene_counter: Optional[int] = None
        self.scene_order: Optional[List[str]] = None
        self.scene_id: Optional[int] = None
        # get the total number of tasks assigned to this process
        if loop_dataset:
            self.max_tasks = None
        else:
            self.max_tasks = sum(len(self.episodes[scene]) for scene in self.episodes)
        self.reset_tasks = self.max_tasks
        self.scene_index = 0
        self.episode_index = 0

        self._last_sampled_task: Optional[PointNavDynamicsCorruptionTask] = None

        self.seed: Optional[int] = None
        self.set_seed(seed)

        if deterministic_cudnn:
            set_deterministic_cudnn()

        self.reset()

    def _create_environment(self) -> IThorEnvironment:
        env = self.env_class(**self.env_args)
        return env

    @property
    def __len__(self) -> Union[int, float]:
        """Length.

        # Returns

        Number of total tasks remaining that can be sampled. Can be float('inf').
        """
        return float("inf") if self.max_tasks is None else self.max_tasks

    @property
    def total_unique(self) -> Optional[Union[int, float]]:
        return self.reset_tasks

    @property
    def last_sampled_task(self) -> Optional[PointNavDynamicsCorruptionTask]:
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

    def next_task(self, force_advance_scene: bool = False) -> Optional[PointNavDynamicsCorruptionTask]:
        if self.max_tasks is not None and self.max_tasks <= 0:
            return None

        if self.episode_index >= len(self.episodes[self.scenes[self.scene_index]]):
            self.scene_index = (self.scene_index + 1) % len(self.scenes)
            # shuffle the new list of episodes to train on
            if self.shuffle_dataset:
                random.shuffle(self.episodes[self.scenes[self.scene_index]])
            self.episode_index = 0

        scene = self.scenes[self.scene_index]
        episode = self.episodes[scene][self.episode_index]
        if self.env is not None:
            #if scene.replace("_physics", "") != self.env.scene_name.replace(
            #        "_physics", ""
            #):
            self.env.reset(scene_name=scene, move_mag=self.env._move_mag, filtered_objects=[])
        else:
            self.env = self._create_environment()
            self.env.reset(scene_name=scene, move_mag=self.env._move_mag, filtered_objects=[])

        def to_pos(s):
            if isinstance(s, (Dict, Tuple)):
                return s
            if isinstance(s, float):
                return {"x": 0, "y": s, "z": 0}
            return str_to_pos_for_cache(s)

        for k in ["initial_position", "initial_orientation", "target_position"]:
            episode[k] = to_pos(episode[k])

        missing_action = [random.choice([0, 1, 2, 3, 7])]
        if missing_action[0] == 0 or missing_action[0] == 1:
            move_mag = random.choice([0.25, 0.5, 0.75])
            if move_mag == 0.25:
                move_rot_deg = random.choice([-1.15, 1.15])
            else:
                move_rot_deg = random.choice([-1.15, 0, 1.15])
        else:
            move_mag = 0.25
            move_rot_deg = 0

        if missing_action[0] == 2 or missing_action[0] == 3:
            rot_deg = random.choice([0, 20, 40])
        else:
            rot_deg = 30

        task_info = {
            "scene": scene,
            "initial_position": episode["initial_position"],
            "initial_orientation": episode["initial_orientation"],
            "target": episode["target_position"],
            "shortest_path": episode["shortest_path"],
            "distance_to_target": episode["shortest_path_length"],
            "id": episode["id"],
            "obstacles_types": episode["obstacles_types"],
            "missing_action": missing_action,
            "move_mag": move_mag,
            "move_rot_deg": move_rot_deg,
            "rot_deg": rot_deg,
        }

        if self.allow_flipping and random.random() > 0.5:
            task_info["mirrored"] = True
        else:
            task_info["mirrored"] = False

        self.episode_index += 1
        if self.max_tasks is not None:
            self.max_tasks -= 1

        if not self.env.teleport(
                pose=episode["initial_position"], rotation=episode["initial_orientation"]
        ):
            if int(scene.split("_")[0][-2:]) < 21:
                return self.next_task()

        self.env.initialize(0.25, renderObjectImage=True, renderDepthImage=True)

        self._last_sampled_task = PointNavDynamicsCorruptionTask(
            env=self.env,
            sensors=self.sensors,
            task_info=task_info,
            max_steps=self.max_steps,
            action_space=self._action_space,
            reward_configs=self.rewards_config,
        )

        return self._last_sampled_task

    def reset(self):
        self.episode_index = 0
        self.scene_index = 0
        self.max_tasks = self.reset_tasks

    def set_seed(self, seed: int):
        self.seed = seed
        if seed is not None:
            set_seed(seed)

    @property
    def length(self) -> Union[int, float]:
        """Length.

        # Returns

        Number of total tasks remaining that can be sampled.
        Can be float('inf').
        """
        return float("inf") if self.max_tasks is None else self.max_tasks

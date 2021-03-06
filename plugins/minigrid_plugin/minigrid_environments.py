import copy
from typing import Optional, Set

import numpy as np
from gym import register
from gym_minigrid.envs import CrossingEnv, DynamicsCorruptionEmptyEnv
from gym_minigrid.minigrid import Lava, Wall


class FastCrossing(CrossingEnv):
    """Similar to `CrossingEnv`, but to support faster task sampling as per
    `repeat_failed_task_for_min_steps` flag in MiniGridTaskSampler."""

    def __init__(self, size=9, num_crossings=1, obstacle_type=Lava, seed=None):
        self.init_agent_pos: Optional[np.ndarray] = None
        self.init_agent_dir: Optional[int] = None
        self.step_count: Optional[int] = None
        super(FastCrossing, self).__init__(
            size=size,
            num_crossings=num_crossings,
            obstacle_type=obstacle_type,
            seed=seed,
        )

    def same_seed_reset(self):
        assert self.init_agent_pos is not None

        # Current position and direction of the agent
        self.agent_pos = self.init_agent_pos
        self.agent_dir = self.init_agent_dir

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

        assert self.carrying is None

        # Step count since episode start
        self.step_count = 0

        # Return first observation
        obs = self.gen_obs()
        return obs

    def reset(self, partial_reset: bool = False):
        super(FastCrossing, self).reset()
        self.init_agent_pos = copy.deepcopy(self.agent_pos)
        self.init_agent_dir = self.agent_dir


class AskForHelpSimpleCrossing(CrossingEnv):
    """Corresponds to WC FAULTY SWITCH environment."""

    def __init__(
        self,
        size=9,
        num_crossings=1,
        obstacle_type=Wall,
        seed=None,
        exploration_reward: Optional[float] = None,
        death_penalty: Optional[float] = None,
        toggle_is_permenant: bool = False,
    ):
        self.init_agent_pos: Optional[np.ndarray] = None
        self.init_agent_dir: Optional[int] = None
        self.should_reveal_image: bool = False
        self.exploration_reward = exploration_reward
        self.death_penalty = death_penalty

        self.explored_points: Set = set()
        self._was_successful = False
        self.toggle_is_permanent = toggle_is_permenant

        self.step_count: Optional[int] = None

        super(AskForHelpSimpleCrossing, self).__init__(
            size=size,
            num_crossings=num_crossings,
            obstacle_type=obstacle_type,
            seed=seed,
        )

    @property
    def was_successful(self) -> bool:
        return self._was_successful

    def gen_obs(self):
        obs = super(AskForHelpSimpleCrossing, self).gen_obs()
        if not self.should_reveal_image:
            obs["image"] *= 0
        return obs

    def metrics(self):
        return {
            "explored_count": len(self.explored_points),
            "final_distance": float(
                min(
                    abs(x - (self.width - 2)) + abs(y - (self.height - 2))
                    for x, y in self.explored_points
                )
            ),
        }

    def step(self, action: int):
        """Reveal the observation only if the `toggle` action is executed."""
        if action == self.actions.toggle:
            self.should_reveal_image = True
        else:
            self.should_reveal_image = (
                self.should_reveal_image and self.toggle_is_permanent
            )

        minigrid_obs, reward, done, info = super(AskForHelpSimpleCrossing, self).step(
            action=action
        )

        assert not self._was_successful, "Called step after done."
        self._was_successful = self._was_successful or (reward > 0)

        if (
            done
            and self.steps_remaining != 0
            and (not self._was_successful)
            and self.death_penalty is not None
        ):
            reward += self.death_penalty

        t = tuple(self.agent_pos)
        if self.exploration_reward is not None:
            if t not in self.explored_points:
                reward += self.exploration_reward
        self.explored_points.add(t)

        return minigrid_obs, reward, done, info

    def same_seed_reset(self):
        assert self.init_agent_pos is not None
        self._was_successful = False

        # Current position and direction of the agent
        self.agent_pos = self.init_agent_pos
        self.agent_dir = self.init_agent_dir

        self.explored_points.clear()
        self.explored_points.add(tuple(self.agent_pos))
        self.should_reveal_image = False

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

        assert self.carrying is None

        # Step count since episode start
        self.step_count = 0

        # Return first observation
        obs = self.gen_obs()
        return obs

    def reset(self, partial_reset: bool = False):
        super(AskForHelpSimpleCrossing, self).reset()
        self.explored_points.clear()
        self.explored_points.add(tuple(self.agent_pos))
        self.init_agent_pos = copy.deepcopy(self.agent_pos)
        self.init_agent_dir = self.agent_dir
        self._was_successful = False
        self.should_reveal_image = False


class DynamicsCorruptionEmpty(DynamicsCorruptionEmptyEnv):
    """Corresponds to WC FAULTY SWITCH environment."""

    def __init__(
            self,
            size=9,
            corruption_actions=[0,1,2,8],
            num_corruption_changes=3,
            num_corruption_actions=1,
            n_obstacles=0,
            is_obstacles_movable=False,
            max_steps=100,
            agent_view_size=5,
            seed=None,
            exploration_reward: Optional[float] = None,
            death_penalty: Optional[float] = None,
    ):
        self.size = size
        self.init_agent_pos: Optional[np.ndarray] = None
        self.init_agent_dir: Optional[int] = None
        self.exploration_reward = exploration_reward
        self.death_penalty = death_penalty
        self.num_action_made = 0
        self.num_missing_action_made = 0

        self.explored_points: Set = set()
        self._was_successful = False

        self.step_count: Optional[int] = None

        super(DynamicsCorruptionEmpty, self).__init__(
            size=size,
            corruption_actions=corruption_actions,
            num_corruption_changes=num_corruption_changes,
            num_corruption_actions=num_corruption_actions,
            n_obstacles=n_obstacles,
            is_obstacles_movable=is_obstacles_movable,
            max_steps=max_steps,
            agent_view_size=agent_view_size,
            seed=seed,
            agent_start_pos=None,
        )

        self.prev_obs = self.gen_obs()

    @property
    def was_successful(self) -> bool:
        return self._was_successful

    def metrics(self):
        return {
            "explored_count": len(self.explored_points),
            "final_distance": self.distance_to_goal,
        }

    def gen_prev_obs(self):
        return self.prev_obs

    def step(self, action: int):
        self.prev_obs = self.gen_obs()
        minigrid_obs, reward, done, info = super(DynamicsCorruptionEmpty, self).step(
            action=action
        )

        self.num_action_made += 1
        if action in self.current_corruption_actions:
            self.num_missing_action_made += 1

        assert not self._was_successful, "Called step after done."
        self._was_successful = self._was_successful or (reward > 0)

        if (
                done
                and self.steps_remaining != 0
                and (not self._was_successful)
                and self.death_penalty is not None
        ):
            reward += self.death_penalty

        t = tuple(self.agent_pos)
        if self.exploration_reward is not None:
            if t not in self.explored_points:
                reward += self.exploration_reward
        self.explored_points.add(t)

        return minigrid_obs, reward, done, info

    def same_seed_reset(self):
        assert self.init_agent_pos is not None
        self._was_successful = False

        # Current position and direction of the agent
        self.agent_pos = self.init_agent_pos
        self.agent_dir = self.init_agent_dir

        self.explored_points.clear()
        self.explored_points.add(tuple(self.agent_pos))

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

        assert self.carrying is None

        # Step count since episode start
        self.step_count = 0

        # Return first observation
        obs = self.gen_obs()
        return obs

    def reset(self, partial_reset: bool = False):
        super(DynamicsCorruptionEmpty, self).reset()
        self.explored_points.clear()
        self.explored_points.add(tuple(self.agent_pos))
        self.init_agent_pos = copy.deepcopy(self.agent_pos)
        self.init_agent_dir = self.agent_dir
        self._was_successful = False
        self.num_action_made = 0
        self.num_missing_action_made = 0


class LavaCrossingS25N10(CrossingEnv):
    def __init__(self):
        super(LavaCrossingS25N10, self).__init__(size=25, num_crossings=10)


class LavaCrossingS15N7(CrossingEnv):
    def __init__(self):
        super(LavaCrossingS15N7, self).__init__(size=15, num_crossings=7)


class LavaCrossingS11N7(CrossingEnv):
    def __init__(self):
        super(LavaCrossingS11N7, self).__init__(size=9, num_crossings=4)


register(
    id="MiniGrid-LavaCrossingS25N10-v0",
    entry_point="plugins.minigrid_plugin.minigrid_environments:LavaCrossingS25N10",
)

register(
    id="MiniGrid-LavaCrossingS15N7-v0",
    entry_point="plugins.minigrid_plugin.minigrid_environments:LavaCrossingS15N7",
)

register(
    id="MiniGrid-LavaCrossingS11N7-v0",
    entry_point="plugins.minigrid_plugin.minigrid_environments:LavaCrossingS11N7",
)

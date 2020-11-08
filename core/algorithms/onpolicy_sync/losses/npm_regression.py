"""Defining imitation losses for actor critic type models."""

from typing import Dict, Union

import torch
import typing

from core.algorithms.onpolicy_sync.losses.abstract_loss import AbstractActorCriticLoss
from core.base_abstractions.misc import ActorCriticOutput
from core.base_abstractions.distributions import CategoricalDistr

from utils.utils_3d_torch import get_gt_affine_matrix_by_pose, get_rotation_matrix_batch, project_to_agent_coordinate


class NPM_Reg(AbstractActorCriticLoss):
    """Expert imitation loss."""
    def __init__(self,
                 agent_pose_uuid: str = None,
                 pose_uuid: str = None,
                 local_keypoints_uuid: str = None,
                 global_keypoints_uuid: str = None,
                 obj_update_mask_uuid: str = None,
                 obj_action_mask_uuid: str = None):
        self.agent_pose_uuid = agent_pose_uuid
        self.pose_uuid = pose_uuid
        self.local_keypoints_uuid = local_keypoints_uuid
        self.global_keypoints_uuid = global_keypoints_uuid
        self.obj_update_mask_uuid = obj_update_mask_uuid
        self.obj_action_mask_uuid = obj_action_mask_uuid

    def loss(  # type: ignore
        self,
        step_count: int,
        batch: Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]],
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
        *args,
        **kwargs
    ):
        """Computes the neural physics machine loss.

        # Parameters

        batch : A batch of data corresponding to the information collected when rolling out (possibly many) agents
            over a fixed number of steps. In particular this batch should have the same format as that returned by
            `RolloutStorage.recurrent_generator`.
            Here `batch["observations"]` must contain `"expert_action"` observations
            or `"expert_policy"` observations. See `ExpertActionSensor` (or `ExpertPolicySensor`) for an example of
            a sensor producing such observations.
        actor_critic_output : The output of calling an ActorCriticModel on the observations in `batch`.
        args : Extra args. Ignored.
        kwargs : Extra kwargs. Ignored.

        # Returns

        A (0-dimensional) torch.FloatTensor corresponding to the computed loss. `.backward()` will be called on this
        tensor in order to compute a gradient update to the ActorCriticModel's parameters.
        """
        observations = typing.cast(Dict[str, torch.Tensor], batch["observations"])
        pose = observations[self.pose_uuid]
        global_keypoints = observations[self.global_keypoints_uuid]
        update_mask = observations[self.obj_update_mask_uuid]
        action_mask = observations[self.obj_action_mask_uuid]

        next_observations = typing.cast(Dict[str, torch.Tensor], batch["next_observations"])
        next_agent_pose = next_observations[self.agent_pose_uuid]
        next_pose = next_observations[self.pose_uuid]

        nb, ng, no, _ = pose.shape
        gt_affine_matrix = get_gt_affine_matrix_by_pose(pose.view(nb * ng * no, 6), next_pose.view(nb * ng * no, 6))

        global_keypoints = global_keypoints.view(nb * ng * no, 8, 3)
        global_keypoints_homo = torch.cat((global_keypoints,
                                           torch.ones((nb * ng * no, 8, 1)).to(global_keypoints.device)), 2)
        global_keypoints_homo = global_keypoints_homo.transpose(1, 2)
        global_keypoints_homo[:, 2, :] *= -1
        gt_next_global_keypoints_homo = torch.matmul(gt_affine_matrix, global_keypoints_homo)
        gt_next_global_keypoints_homo[:, 2, :] *= -1
        gt_next_global_keypoints = gt_next_global_keypoints_homo.transpose(1, 2)[:, :, :3]

        next_agent_pose = next_agent_pose.view(nb * ng, 6)
        r = get_rotation_matrix_batch(next_agent_pose[:, 3:])
        next_agent_pose = next_agent_pose.view(nb * ng, 1, 6).repeat(1, no, 1).view(nb * ng * no, 6)
        r = r.view(nb * ng, 1, 3, 3).repeat(1, no, 1, 1).view(nb * ng * no, 3, 3)
        gt_next_local_keypoints = project_to_agent_coordinate(gt_next_global_keypoints,
                                                              next_agent_pose[:, :3],
                                                              r)
        gt_next_local_keypoints = gt_next_local_keypoints.view(nb, ng, no, 8, 3)

        gt_next_local_keypoints = gt_next_local_keypoints.reshape(nb * ng, no, 24)
        pred_next_local_keypoints = actor_critic_output["npm_output"].reshape(nb * ng, no, 24)
        update_mask = update_mask.view(nb * ng, no, 1).repeat(1, 1, 24)
        action_mask = action_mask.view(nb * ng, no, 1).repeat(1, 1, 24)

        action = batch["actions"].view(nb * ng)
        masks = batch["masks"].view(nb * ng)
        loss = 0.
        for b_idx, a in enumerate(action):
            l = torch.nn.functional.l1_loss(pred_next_local_keypoints[b_idx],
                                            gt_next_local_keypoints[b_idx],
                                            reduce=False) * update_mask[b_idx]
            if a in [5, 6, 7, 8]:
                loss += (((l * action_mask[b_idx]).sum() / max(1, action_mask[b_idx].sum())) * masks[b_idx])
            else:
                loss += ((l.sum() / max(1, update_mask[b_idx].sum())) * masks[b_idx])
        loss /= (nb * ng)
        #loss *= 3

        return (
            loss,
            {"npm_regression": loss.item(),},
        )

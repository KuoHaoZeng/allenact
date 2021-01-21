"""Defining imitation losses for actor critic type models."""

from typing import Dict, Union

import torch
import typing

from core.algorithms.onpolicy_sync.losses.abstract_loss import AbstractActorCriticLoss
from core.base_abstractions.misc import ActorCriticOutput
from core.base_abstractions.distributions import CategoricalDistr

from utils.utils_3d_torch import get_gt_affine_matrix_by_pose, get_rotation_matrix_batch, project_to_agent_coordinate


class PBL_loss(AbstractActorCriticLoss):
    """Expert imitation loss."""
    def __init__(self,
                 obs_uuid: str = None,
                 forward_uuid: str = None,
                 reverse_uuid: str = None,
                 b_uuid: str = None):
        self.obs_uuid = obs_uuid
        self.forward_uuid = forward_uuid
        self.reverse_uuid = reverse_uuid
        self.b_uuid = b_uuid

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

        obs = actor_critic_output[self.obs_uuid]
        forward = actor_critic_output[self.forward_uuid]
        reverse = actor_critic_output[self.reverse_uuid]
        b = actor_critic_output[self.b_uuid]

        nt, ng, nf = obs.shape[0], obs.shape[1], obs.shape[2]
        obs = obs[1:]
        forward = forward[:-1]
        obs = obs.view((nt - 1) * ng, nf)
        forward = forward.view((nt - 1) * ng, nf)

        reverse = reverse[1:]
        b = b[:-1]
        reverse = reverse.view((nt - 1) * ng, nf)
        b = b.view((nt - 1) * ng, nf)

        masks_forward = batch["masks"][:-1].view((nt - 1) * ng)
        masks_reverse = batch["masks"][:-1].view((nt - 1) * ng)
        loss_forward = torch.nn.functional.mse_loss(forward, obs.detach(), reduce=False).mean(1) * masks_forward
        loss_forward = loss_forward.sum() / masks_forward.sum()
        loss_reverse = torch.nn.functional.mse_loss(reverse, b.detach(), reduce=False).mean(1) * masks_reverse
        loss_reverse = loss_reverse.sum() / masks_reverse.sum()
        loss = (1 * loss_forward + 1 * loss_reverse)

        return (
            loss,
            {"PBL_loss": loss.item(),},
        )

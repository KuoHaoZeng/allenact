"""Defining imitation losses for actor critic type models."""

from typing import Dict, Union

import torch
import typing

from core.algorithms.onpolicy_sync.losses.abstract_loss import AbstractActorCriticLoss
from core.base_abstractions.misc import ActorCriticOutput
from core.base_abstractions.distributions import CategoricalDistr

from utils.utils_3d_torch import get_gt_affine_matrix_by_pose, get_rotation_matrix_batch, project_to_agent_coordinate


class CPC_loss(AbstractActorCriticLoss):
    """Expert imitation loss."""
    def __init__(self,
                 positive_uuid: str = None,
                 negative_uuid: str = None):
        self.positive_uuid = positive_uuid
        self.negative_uuid = negative_uuid

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

        positive = actor_critic_output[self.positive_uuid]
        negative = actor_critic_output[self.negative_uuid]

        nt, ng = positive.shape[0], positive.shape[1]
        positive = positive.view(nt * ng)
        negative = negative.view(nt * ng)
        masks = batch["masks"][:-1].view(nt * ng)

        pos_label = torch.ones(nt * ng).to(positive.device)
        neg_label = torch.zeros(nt * ng).to(negative.device)
        loss_pos = torch.nn.functional.binary_cross_entropy_with_logits(positive, pos_label, reduce=False)
        loss_neg = torch.nn.functional.binary_cross_entropy_with_logits(negative, neg_label, reduce=False)
        loss = (1 * loss_pos + 1 * loss_neg) * masks
        loss = loss.sum() / masks.sum()

        return (
            loss,
            {"CPC_loss": loss.item(),},
        )

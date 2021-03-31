"""Defining imitation losses for actor critic type models."""

from typing import Dict, Union

import torch
import typing

from core.algorithms.onpolicy_sync.losses.abstract_loss import AbstractActorCriticLoss
from core.base_abstractions.misc import ActorCriticOutput
from core.base_abstractions.distributions import CategoricalDistr

from utils.hessian_penality import hessian_penalty

class HP_loss(AbstractActorCriticLoss):
    """Expert imitation loss."""
    def __init__(self,
                 internal_uuid: str = None):
        self.internal_uuid = internal_uuid

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

        internal_out = actor_critic_output[self.internal_uuid]
        internal_gt = batch["observations"]["missing_action"]

        nt, ng = internal_out.shape[0], internal_out.shape[1]
        internal_out = internal_out.view(nt * ng, -1)[torch.arange(nt * ng), prev_actions]

        loss = torch.nn.functional.binary_cross_entropy_with_logits(internal_out, internal_gt, reduce=False)
        loss = (loss * masks).sum() / masks.sum()
        #loss = (loss * masks * internal_gt_mask).sum() / (masks * internal_gt_mask).sum()
        loss = loss

        return (
            loss,
            {"MA_loss": loss.item(),},
        )

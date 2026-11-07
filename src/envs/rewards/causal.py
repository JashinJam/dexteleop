"""Explicit human-command intent reward; keep original task rewards unchanged."""
import torch
from envs._common import action_term
from envs.rewards.motion_tracking import MotionTrackingRewards


class CausalRewards(MotionTrackingRewards):
    @staticmethod
    def tracking_causal_finger_intent_exp(env,command_name,scale,side=None):
        action=action_term(env);ids=getattr(action, '_finger_ids', action._all_joint_ids)
        if side is not None:
            names=action._entity.joint_names
            ids=ids[torch.tensor([names[int(i)].startswith(side+'_') for i in ids],device=ids.device)]
        target=action._entity.data.joint_pos_target[:,ids]
        reference=action.ref_joint_pos_used[:,ids]
        limits=action.entity_joint_limits[ids]
        error=.5*(target-reference).square()/(limits[:,1]-limits[:,0])
        return torch.exp(-scale*error).mean(-1)

"""S2b actor inputs: human/proprio history plus the explicit current wrist reference."""
import torch
from envs._common import action_term
from .causal import CausalObs


class HybridSharpaObs(CausalObs):
    @staticmethod
    def hybrid_wrist_reference(env, command_name):
        action = action_term(env)
        action.ensure_command()
        command = env.command_manager.get_term(command_name)
        ids = action._wrist_ids
        reference = command.ref_joint_pos[:, ids]
        actual = action._entity.data.joint_pos[:, ids]
        return torch.cat([
            reference, (reference - actual) / action._wrist_scale,
            action._hybrid_filtered[:, :12],
        ], dim=-1)

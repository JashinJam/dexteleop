"""Flash reference-wrist residuals and full-range, zero-centred finger residuals.

The offline wrist reference is deliberate: this is the S2b comparison interface,
not a claim of causal live-wrist deployment. Geometry comes from the named cache.
"""
from dataclasses import dataclass

import torch
from mjlab.managers.action_manager import ActionTermCfg

from .causal import CausalAction
from .lpf import ema_alpha


def full_range_target(base, action, lower, upper):
    """Zero keeps base; -1/+1 reach the model limits without a small angle cap."""
    base = base.clamp(lower, upper)
    action = action.clamp(-1, 1)
    room = torch.where(action >= 0, upper - base, base - lower)
    return (base + action * room).clamp(lower, upper)


@dataclass(kw_only=True)
class HybridSharpaActionCfg(ActionTermCfg):
    entity_name: str
    command_name: str
    command_cache: str
    wrist_translation_scale: float = 0.1
    wrist_rotation_scale: float = 1.0
    wrist_cutoff_hz: float = 5.0
    zero_residual: bool = False

    def build(self, env):
        return HybridSharpaAction(self, env)


class HybridSharpaAction(CausalAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._finger_ids = self._all_joint_ids.clone()
        self._all_joint_ids = torch.cat([self._wrist_ids, self._finger_ids])
        self._hybrid_raw = torch.zeros(self.num_envs, 56, device=self.device)
        self._hybrid_filtered = self._hybrid_raw.clone()
        self._wrist_pending = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._wrist_scale = torch.tensor(
            [cfg.wrist_translation_scale] * 3 + [cfg.wrist_rotation_scale] * 3,
            device=self.device,
        ).repeat(2)
        self._wrist_alpha = ema_alpha(cfg.wrist_cutoff_hz, env.step_dt)

    @property
    def action_dim(self):
        return 56

    @property
    def n_dofs(self):
        return 56

    @property
    def raw_action(self):
        return self._hybrid_raw

    def process_actions(self, actions):
        self.ensure_command()
        if actions.shape != self._hybrid_raw.shape or not torch.isfinite(actions).all():
            raise ValueError("Expected finite (num_envs, 56) actions")
        self._hybrid_raw.copy_(actions.clamp(-1, 1))
        if self.cfg.zero_residual:
            self._hybrid_raw.zero_()
        wrist = self._hybrid_raw[:, :12]
        previous = self._hybrid_filtered[:, :12]
        filtered = torch.where(
            self._wrist_pending[:, None], wrist,
            self._wrist_alpha * wrist + (1 - self._wrist_alpha) * previous,
        )
        self._hybrid_filtered[:, :12] = filtered
        self._hybrid_filtered[:, 12:] = self._hybrid_raw[:, 12:]
        self._wrist_pending.fill_(False)
        # The original human/proprio history still exposes the 44 finger commands.
        self.state.raw.copy_(self._hybrid_raw[:, 12:])
        self.state.filtered.copy_(self._hybrid_filtered[:, 12:])
        command = self._env.command_manager.get_term(self.cfg.command_name)
        self.ref_joint_pos_used.copy_(command.ref_joint_pos)
        self.ref_joint_pos_used[:, self._finger_ids] = self.state.finger_q
        self._target.copy_(self.ref_joint_pos_used)
        limits = self.entity_joint_limits
        self._target[:, self._wrist_ids] = (
            command.ref_joint_pos[:, self._wrist_ids] + filtered * self._wrist_scale
        ).clamp(limits[self._wrist_ids, 0], limits[self._wrist_ids, 1])
        self._target[:, self._finger_ids] = full_range_target(
            self.state.finger_q, self._hybrid_filtered[:, 12:],
            limits[self._finger_ids, 0], limits[self._finger_ids, 1],
        )

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._hybrid_raw[ids] = 0
        self._hybrid_filtered[ids] = 0
        self._wrist_pending[ids] = True

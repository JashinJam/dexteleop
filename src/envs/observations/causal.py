"""Actor-visible terms read only human commands and robot proprioception."""
import torch
from mjlab.utils.lab_api.math import matrix_from_quat
from envs._common import action_term
from envs.commands.causal_hand import rot6d
from envs.observations.motion_tracking import MotionTrackingObs


class CausalObs(MotionTrackingObs):
    @staticmethod
    def causal_command_history(env):
        action=action_term(env);action.ensure_command()
        return action.state.history.flatten(1)

    @staticmethod
    def causal_proprioception(env,command_name):
        action=action_term(env);command=env.command_manager.get_term(command_name)
        ids=getattr(action, '_finger_ids', action._all_joint_ids);robot=action._entity
        pose=torch.cat([command.robot_wrist_trans_w-env.scene.env_origins[:,None],rot6d(matrix_from_quat(command.robot_wrist_quat_w))],-1).flatten(1)
        result=torch.cat([pose,command.robot_wrist_lin_vel_w.flatten(1)*2,
             command.robot_wrist_ang_vel_w.flatten(1)*.25,
             action.state.normalize_q(robot.data.joint_pos[:,ids]),robot.data.joint_vel[:,ids]*.1,
             MotionTrackingObs.robot_tip_trans_wrist(env,command_name)/.1,
             MotionTrackingObs.robot_tip_lin_vel_wrist(env,command_name)*2,action.state.filtered],-1)
        assert result.shape[1]==222
        return result

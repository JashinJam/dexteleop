"""44 finger residuals; human-only calibrated wrist targets outside the policy."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.utils.lab_api.math import matrix_from_quat
from envs.commands.causal_hand import CausalHandState


@dataclass(kw_only=True)
class CausalActionCfg(ActionTermCfg):
    entity_name: str
    command_name: str
    command_cache: str
    residual_limit: float = .15
    zero_residual: bool = False

    def build(self,env):return CausalAction(self,env)


class CausalAction(ActionTerm):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        names=self._entity.joint_names
        self._all_joint_ids=torch.tensor([i for side in ('right','left') for i,n in enumerate(names) if n.startswith(side+'_')],device=self.device)
        assert len(self._all_joint_ids)==44
        self._wrist_by_side=[]
        for prefix in ('R','L'):
            self._wrist_by_side.append([names.index(f'{prefix}_forearm_{part}_joint') for part in ('pos_x','pos_y','pos_z','rot_z','rot_x','rot_y')])
        self._wrist_ids=torch.tensor(self._wrist_by_side,device=self.device).flatten()
        self._cache=torch.load(cfg.command_cache,map_location='cpu',weights_only=False)
        self.state=CausalHandState(self.num_envs,self.device,self.entity_joint_limits[self._all_joint_ids],self._cache['wrist_parameters'],env.step_dt)
        self._flat=None
        self._last_tick=-1
        self._last_motion=torch.full((self.num_envs,),-1,device=self.device,dtype=torch.long)
        self._last_frame=self._last_motion.clone()
        self._target=self._entity.data.joint_pos.clone()
        self.ref_joint_pos_used=self._target.clone()
        self._now=torch.zeros(self.num_envs,2,device=self.device)

    @property
    def action_dim(self):return 44
    @property
    def n_dofs(self):return 44
    @property
    def raw_action(self):return self.state.raw
    @property
    def entity_joint_limits(self):return self._entity.data.soft_joint_pos_limits[0]

    def _load_flat(self,command):
        lib=command.motion_lib
        columns={key:[] for key in ('wrist_pos','wrist_rot','joints_local','finger_q')}
        for i,name in enumerate(lib.motion_files):
            name=name.rsplit('#',1)[-1]
            record=self._cache['clips'][name]
            count=int(lib._motion_num_frames[i]);crop=len(record['frame_id'])-count
            assert crop>=0,(name,crop)
            for side in ('right','left'):
                ids=[int(x) for x in self._all_joint_ids if self._entity.joint_names[int(x)].startswith(side+'_')]
                assert record[side]['joint_names']==[self._entity.joint_names[x] for x in ids]
                start=int(lib.length_starts[i])
                # Alignment is checked using orientation, which is unaffected
                # by the offline object-based floor translation.
                reference=lib.mano_wrist_rot[side][start:start+count].cpu().numpy()
                assert np.max(abs(record[side]['wrist_rot'][crop:]-reference))<2e-5,(name,side,'time alignment')
            for key in columns:
                arr=np.stack([record[s][key][crop:] for s in ('right','left')],axis=1)
                if key=='finger_q':arr=arr.reshape(count,44)
                columns[key].append(arr)
        self._flat={key:torch.as_tensor(np.concatenate(value),dtype=torch.float32,device=self.device) for key,value in columns.items()}
        del self._cache

    def ensure_command(self):
        command=self._env.command_manager.get_term(self.cfg.command_name)
        if self._flat is None:self._load_flat(command)
        tick=int(self._env.common_step_counter)
        changed=(command.motion_ids!=self._last_motion)|(command.motion_steps!=self._last_frame)
        ids=torch.where(changed|~self.state.initialized|(tick!=self._last_tick))[0]
        if not len(ids):return
        discontinuity=(command.motion_ids[ids]!=self._last_motion[ids])|((command.motion_steps[ids]!=self._last_frame[ids]+1)&(command.motion_steps[ids]!=self._last_frame[ids]))
        self.state.reset(ids[discontinuity])
        index=command.motion_lib.length_starts[command.motion_ids[ids]]+command.motion_steps[ids]
        values={key:value[index] for key,value in self._flat.items()}
        now=torch.full((len(ids),2),tick*self._env.step_dt,device=self.device)
        self._now[ids]=now
        robot_pos=command.robot_wrist_trans_w[ids]-self._env.scene.env_origins[ids,None]
        robot_rot=matrix_from_quat(command.robot_wrist_quat_w[ids])
        self.state.push(ids,local_joints=values['joints_local'],finger_q=values['finger_q'],
                        human_pos=values['wrist_pos'],human_rot=values['wrist_rot'],sample_time=now,now=now,
                        robot_pos=robot_pos,robot_rot=robot_rot)
        self._last_motion[ids]=command.motion_ids[ids];self._last_frame[ids]=command.motion_steps[ids];self._last_tick=tick

    def process_actions(self,actions):
        self.ensure_command()
        fingers=self.state.action(actions,self.cfg.residual_limit,self._now,self.cfg.zero_residual)
        self._target[:,self._all_joint_ids]=fingers
        wrist=self.state.wrist_q.flatten(1)
        limits=self.entity_joint_limits[self._wrist_ids]
        self._target[:,self._wrist_ids]=wrist.clamp(limits[:,0],limits[:,1])
        self.ref_joint_pos_used.copy_(self._target)
        self.ref_joint_pos_used[:,self._all_joint_ids]=self.state.finger_q

    def apply_actions(self):
        self._entity.set_joint_position_target(self._target)

    def reset(self,env_ids=None):
        if env_ids is None or isinstance(env_ids,slice):env_ids=torch.arange(self.num_envs,device=self.device)[env_ids or slice(None)]
        self.state.reset(env_ids)
        self._last_motion[env_ids]=-1;self._last_frame[env_ids]=-1

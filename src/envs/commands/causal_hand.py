"""Explicit causal command state, usable without a simulator or object inputs."""
from __future__ import annotations
import math
import torch


def rot6d(matrix):
    # Columns, not the first six entries of a row-major flattened matrix.
    return matrix[..., :, :2].transpose(-1,-2).flatten(-2)


class CausalHandState:
    frame_dim = 228
    history_length = 4

    def __init__(self, count, device, limits, wrist_parameters, dt=1/60):
        self.device, self.count, self.dt = device, count, dt
        self.limits = torch.as_tensor(limits,device=device,dtype=torch.float32)
        self.lower,self.upper = self.limits.unbind(-1)
        self.A = torch.stack([torch.as_tensor(wrist_parameters[s]['A'],device=device,dtype=torch.float32) for s in ('right','left')])
        self.D = torch.stack([torch.as_tensor(wrist_parameters[s]['D'],device=device,dtype=torch.float32) for s in ('right','left')])
        self.origin = torch.stack([torch.as_tensor(wrist_parameters[s]['origin'],device=device,dtype=torch.float32) for s in ('right','left')])
        zeros=lambda *shape:torch.zeros(count,*shape,device=device)
        self.history=zeros(4,228); self.finger_q=zeros(44)
        self.wrist_pos=zeros(2,3);self.wrist_rot=torch.eye(3,device=device).expand(count,2,3,3).clone()
        self.wrist_q=zeros(2,6);self.local_joints=zeros(2,20,3)
        self.human_anchor=zeros(2,3);self.robot_anchor=zeros(2,3)
        self.calibration_rot=self.wrist_rot.clone();self.sample_time=zeros(2)
        self.hand_valid=zeros(2);self.joint_valid=zeros(2,20)
        self.filtered=zeros(44);self.raw=zeros(44)
        self.initialized=torch.zeros(count,dtype=torch.bool,device=device)
        self.alpha=2*math.pi*8*dt/(1+2*math.pi*8*dt)

    def reset(self, ids):
        self.initialized[ids]=False
        self.history[ids]=0;self.filtered[ids]=0;self.raw[ids]=0
        self.hand_valid[ids]=0;self.joint_valid[ids]=0

    def normalize_q(self,q):
        return (2*q-self.lower-self.upper)/(self.upper-self.lower)

    def push(self, ids, human_pos, human_rot, local_joints, finger_q,
             sample_time, now, robot_pos, robot_rot, valid=None):
        """One control tick. Timestamps refer to the same monotonic clock.

        A caller must never call twice for one tick. Validation/reception age
        cannot be replaced by the recorded motion's remaining duration.
        """
        if valid is None:valid=torch.ones(len(ids),2,dtype=torch.bool,device=self.device)
        if bool((sample_time>now+1e-6).any()):raise ValueError('Future human sample')
        valid=valid & torch.isfinite(human_pos).all(-1) & torch.isfinite(human_rot).all((-1,-2)) & torch.isfinite(local_joints).all((-1,-2))
        valid=valid & torch.isfinite(finger_q.reshape(-1,2,22)).all(-1)
        first=~self.initialized[ids]
        if bool((first & ~valid.all(-1)).any()):raise ValueError('Initialization needs two valid hands')
        first_ids=ids[first]
        self.human_anchor[first_ids]=human_pos[first]
        self.robot_anchor[first_ids]=robot_pos[first]
        self.calibration_rot[first_ids]=robot_rot[first]@human_rot[first].transpose(-1,-2)
        # Translation axes are fixed in the common world; orientation is
        # calibrated once from the currently observable human/robot wrists.
        p=self.robot_anchor[ids]+human_pos-self.human_anchor[ids]
        r=self.calibration_rot[ids]@human_rot
        fresh=valid & ((now-sample_time)<=.1)
        self.wrist_pos[ids]=torch.where(fresh[...,None],p,self.wrist_pos[ids])
        self.wrist_rot[ids]=torch.where(fresh[...,None,None],r,self.wrist_rot[ids])
        fq=finger_q.reshape(-1,2,22)
        self.finger_q[ids]=torch.where(fresh[...,None],fq,self.finger_q[ids].reshape(-1,2,22)).reshape(-1,44).clamp(self.lower,self.upper)
        self.local_joints[ids]=torch.where(fresh[...,None,None],local_joints,self.local_joints[ids])
        self.sample_time[ids]=torch.where(valid,sample_time,self.sample_time[ids])
        age=(now-self.sample_time[ids]).clamp(min=0)
        self.hand_valid[ids]=fresh.float();self.joint_valid[ids]=fresh[...,None].expand(-1,-1,20).float()
        self._inverse_wrist(ids,first)
        frame=torch.cat([torch.cat([self.wrist_pos[ids],rot6d(self.wrist_rot[ids])],-1).flatten(1),
                         (self.local_joints[ids]/.1).flatten(1),self.normalize_q(self.finger_q[ids]),
                         (age/.1).clamp(0,1),fresh.float(),torch.ones_like(age),self.joint_valid[ids].flatten(1)],-1)
        assert frame.shape[-1]==228
        self.history[ids,1:]=self.history[ids,:-1].clone();self.history[ids,0]=frame
        if len(first_ids):
            self.history[first_ids]=frame[first,None].expand(-1,4,-1).clone()
            self.history[first_ids,1:,186:188]=0
        self.initialized[ids]=True

    def _inverse_wrist(self,ids,first):
        p=self.wrist_pos[ids];r=self.wrist_rot[ids]
        xyz=(self.A.transpose(-1,-2)@(p-self.origin)[...,None]).squeeze(-1)
        m=self.A.transpose(-1,-2)@r@self.D.transpose(-1,-2)
        x=torch.asin(m[...,2,1].clamp(-1,1))
        z=torch.atan2(-m[...,0,1],m[...,1,1]);y=torch.atan2(m[...,2,0],m[...,2,2])
        angles=torch.stack([z,x,y],-1)
        previous=self.wrist_q[ids,:,3:]
        angles=angles+2*math.pi*torch.round((previous-angles)/(2*math.pi))
        alternate=torch.stack([z+math.pi,math.pi-x,y-math.pi],-1)
        alternate=alternate+2*math.pi*torch.round((previous-alternate)/(2*math.pi))
        use_other=(alternate-previous).square().sum(-1)<(angles-previous).square().sum(-1)
        angles=torch.where(use_other[...,None] & ~first[:,None,None],alternate,angles)
        self.wrist_q[ids]=torch.cat([xyz,angles],-1)

    def action(self,raw,epsilon,now,zero_residual=False):
        self.raw.copy_(raw.clamp(-1,1))
        self.filtered.lerp_(self.raw,self.alpha)
        age=(now-self.sample_time).clamp(min=0)
        # Fade the allowed residual amplitude to exactly zero by 200 ms.
        fade=((.2-age)/.1).clamp(0,1)[...,None].expand(-1,-1,22).reshape(-1,44)
        residual=self.filtered*fade
        if zero_residual:
            self.filtered.zero_();self.raw.zero_();residual=torch.zeros_like(residual)
        return (self.finger_q+float(epsilon)*residual).clamp(self.lower,self.upper)

"""Object-free, frame-causal human-to-Sharpa geometry.

Only kinematics/Jacobians are evaluated: no contact solver, trajectory fitting,
object state, central differences or look-ahead. All state is explicit per stream.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from retarget.human_demo.clips import FINGERS, MANO_JOINT_NAMES


@dataclass(frozen=True)
class GeometryConfig:
    direction_weight: float = .2
    previous_weight: float = .02
    neutral_weight: float = .002
    damping: float = .001
    max_increment: float = .1
    first_iterations: int = 20
    iterations: int = 5


class CausalHandMapper:
    def __init__(self, side: str, config: GeometryConfig = GeometryConfig()):
        if side not in ('right', 'left'):
            raise ValueError(side)
        self.side, self.config = side, config
        xml = Path(__file__).resolve().parents[2]/'assets/robot/sharpa'/f'{side}.xml'
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        self.names = [self.model.joint(i).name for i in range(self.model.njnt)]
        self.finger_ids = np.array([i for i,n in enumerate(self.names) if n.startswith(side+'_')])
        self.wrist_ids = np.array([i for i,n in enumerate(self.names) if 'forearm' in n])
        assert len(self.finger_ids) == 22 and len(self.wrist_ids) == 6
        self.lower, self.upper = self.model.jnt_range[self.finger_ids].T.copy()
        self.ranges = self.upper-self.lower
        self.neutral = np.clip(np.zeros(22), self.lower, self.upper)
        self.palm = self.model.site(f'{side}_palm').id
        self.points = []
        for finger in FINGERS:
            bones = ['thumb_MC','thumb_PP','thumb_DP'] if finger == 'thumb' else [f'{finger}_MCP_VL',f'{finger}_MP',f'{finger}_DP']
            self.points += [('body', self.model.body(f'{side}_{b}').id) for b in bones]
            self.points += [('site', self.model.site(f'track_hand_{side}_{finger}_tip').id)]
        self.human_order = [MANO_JOINT_NAMES.index(f'{f}_{p}') for f in FINGERS for p in ('proximal','intermediate','distal','tip')]
        self._jp = np.zeros((3,self.model.nv)); self._jr = np.zeros_like(self._jp)
        points, _ = self._kinematics(self.neutral, jacobian=False)
        self.robot_neutral_points = points.reshape(5,4,3)
        self.robot_lengths = np.linalg.norm(np.diff(self.robot_neutral_points, axis=1), axis=-1)
        self.reset()

    def reset(self):
        self.previous = self.neutral.copy()
        self.started = False
        self.human_lengths = None
        self.fallback_count = 0

    def _kinematics(self, q, jacobian=True):
        self.data.qpos[self.finger_ids] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        origin = self.data.site_xpos[self.palm]
        rotation = self.data.site_xmat[self.palm].reshape(3,3)
        points, jac = [], []
        for kind, idx in self.points:
            if kind == 'body':
                p = self.data.xpos[idx]
                if jacobian: mujoco.mj_jac(self.model,self.data,self._jp,self._jr,p,idx)
            else:
                p = self.data.site_xpos[idx]
                if jacobian: mujoco.mj_jacSite(self.model,self.data,self._jp,self._jr,idx)
            points.append((p-origin) @ rotation)
            if jacobian: jac.append(rotation.T @ self._jp[:,self.finger_ids])
        return np.stack(points), np.stack(jac) if jacobian else None

    def step(self, local_joints):
        """20 current joint coordinates in the fixed MANO palm frame (metres).

        Missing data is rejected as a frame, preserving the last valid solution.
        Sparse/SPIDER input requires its own explicit mask-aware extension.
        """
        start = time.perf_counter()
        human = np.asarray(local_joints, dtype=np.float64)[self.human_order].reshape(5,4,3)
        bones = np.diff(human,axis=1); lengths = np.linalg.norm(bones,axis=-1)
        if not np.isfinite(human).all() or np.any(lengths < 1e-5):
            self.fallback_count += 1
            return self.previous.copy(), dict(valid=False, seconds=time.perf_counter()-start)
        if self.human_lengths is None:
            self.human_lengths = lengths.copy()
        # Frozen per-bone size calibration. Keep current directions and length
        # changes; no whole-record median or future extrema are used.
        scaled_bones = bones * (self.robot_lengths/self.human_lengths)[...,None]
        targets = self.robot_neutral_points[:,0] + scaled_bones.sum(axis=1)
        desired_direction = bones/lengths[...,None]
        q = self.previous.copy(); cfg = self.config
        n = cfg.iterations if self.started else cfg.first_iterations
        for _ in range(n):
            points,jac = self._kinematics(q)
            points = points.reshape(5,4,3); jac = jac.reshape(5,4,3,22)
            tip_r = (points[:,-1]-targets)/(.1*np.sqrt(5))
            tip_j = jac[:,-1]/(.1*np.sqrt(5))
            v = np.diff(points,axis=1); vj = np.diff(jac,axis=1)
            length = np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-8)
            unit = v/length
            direction_j = np.einsum('...ij,...jk->...ik',np.eye(3)-unit[..., :, None]*unit[..., None, :],vj)/length[...,None]
            w = np.sqrt(cfg.direction_weight/15)
            r = np.concatenate([tip_r.ravel(), ((unit-desired_direction)*w).ravel(),
                                (q-self.previous)/self.ranges*np.sqrt(cfg.previous_weight/22),
                                (q-self.neutral)/self.ranges*np.sqrt(cfg.neutral_weight/22)])
            J = np.concatenate([tip_j.reshape(-1,22), (direction_j*w).reshape(-1,22),
                                np.diag(np.sqrt(cfg.previous_weight/22)/self.ranges),
                                np.diag(np.sqrt(cfg.neutral_weight/22)/self.ranges)])
            dq = np.linalg.solve(J.T@J+cfg.damping*np.eye(22), -J.T@r)
            q = np.clip(q+np.clip(dq,-cfg.max_increment,cfg.max_increment), self.lower,self.upper)
        valid = bool(np.isfinite(q).all())
        if valid: self.previous = q; self.started = True
        else: self.fallback_count += 1
        points,_ = self._kinematics(self.previous,jacobian=False)
        err = np.linalg.norm(points.reshape(5,4,3)[:,-1]-targets,axis=-1)
        return self.previous.copy(), dict(valid=valid, tip_error_mean_m=float(err.mean()),
                                          tip_error_max_m=float(err.max()),seconds=time.perf_counter()-start)

    def wrist_parameters(self):
        """Fixed MJCF transforms for p=p0+A*t, R=A*Rz*Rx*Ry(-q5)*D."""
        self.data.qpos[:] = 0
        mujoco.mj_kinematics(self.model,self.data)
        root = self.model.jnt_bodyid[self.wrist_ids[0]]
        A = self.data.xmat[root].reshape(3,3).copy()
        return dict(origin=self.data.site_xpos[self.palm].copy(), A=A,
                    D=A.T@self.data.site_xmat[self.palm].reshape(3,3))

    def wrist_inverse(self, position, rotation, previous=None):
        params = self.wrist_parameters(); A,D = params['A'],params['D']
        xyz = A.T@(np.asarray(position)-params['origin'])
        angles = Rotation.from_matrix(A.T@rotation@D.T).as_euler('ZXY')
        angles[2] *= -1
        if previous is not None:
            angles += 2*np.pi*np.round((np.asarray(previous)[3:]-angles)/(2*np.pi))
        return np.concatenate([xyz,angles])

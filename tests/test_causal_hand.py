"""Physics-independent tests of the new deployment boundary and real MJCF geometry."""
import unittest
import numpy as np
import torch
import mujoco
from retarget.causal_hand import CausalHandMapper
from envs.commands.causal_hand import CausalHandState


class GeometryTest(unittest.TestCase):
    def test_wrist_roundtrip_real_model(self):
        rng=np.random.default_rng(4)
        for side in ('right','left'):
            mapper=CausalHandMapper(side)
            for _ in range(30):
                expected=np.r_[rng.uniform(-.4,.4,3),rng.uniform(-1.4,1.4,3)]
                mapper.data.qpos[:]=0;mapper.data.qpos[:6]=expected
                mujoco.mj_kinematics(mapper.model,mapper.data)
                pos=mapper.data.site_xpos[mapper.palm].copy()
                rot=mapper.data.site_xmat[mapper.palm].reshape(3,3).copy()
                np.testing.assert_allclose(mapper.wrist_inverse(pos,rot,expected),expected,atol=1e-7)

    def test_geometry_jacobian_and_prefix(self):
        mapper=CausalHandMapper('right');q=mapper.neutral.copy()+.04
        p,j=mapper._kinematics(q)
        for i in (0,7,17,21):
            perturb=q.copy();perturb[i]+=1e-6
            pp,_=mapper._kinematics(perturb,jacobian=False)
            np.testing.assert_allclose((pp-p)/1e-6,j[:,:,i],atol=1e-6)
        sequence=[]
        for t in range(30):
            q=mapper.neutral.copy()+.1*np.sin(t/9)
            p,_=mapper._kinematics(q,jacobian=False)
            sequence.append(p[np.argsort(mapper.human_order)])
        def replay(frames):
            m=CausalHandMapper('right');return np.stack([m.step(x)[0] for x in frames])
        np.testing.assert_array_equal(replay(sequence[:15]),replay(sequence)[:15])
        m=CausalHandMapper('right');before=m.step(sequence[0])[0]
        after,meta=m.step(np.full((20,3),np.nan))
        self.assertFalse(meta['valid']);np.testing.assert_array_equal(before,after)


class StateTest(unittest.TestCase):
    def setUp(self):
        maps=[CausalHandMapper(s) for s in ('right','left')]
        limits=np.concatenate([np.stack([m.lower,m.upper],-1) for m in maps])
        params={s:m.wrist_parameters() for s,m in zip(('right','left'),maps)}
        self.state=CausalHandState(2,'cpu',limits,params)
        self.ids=torch.arange(2);self.eye=torch.eye(3).expand(2,2,3,3).clone()
        self.pos=torch.zeros(2,2,3);self.joints=torch.zeros(2,2,20,3);self.q=torch.zeros(2,44)
    def push(self,t,pos=None,stamp=None):
        self.state.push(self.ids,self.pos if pos is None else pos,self.eye,self.joints,self.q,
                        torch.full((2,2),t if stamp is None else stamp),torch.full((2,2),t),self.pos,self.eye)
    def test_reset_history_stop_reverse_and_stale(self):
        self.push(0.)
        self.assertEqual(tuple(self.state.history.shape),(2,4,228))
        self.assertTrue(bool((self.state.history[:,1:,186:188]==0).all()))
        self.push(1/60,self.pos+.01)
        torch.testing.assert_close(self.state.wrist_pos,self.pos+.01)
        self.push(2/60,self.pos+.01) # stop means a fresh, stationary sample
        torch.testing.assert_close(self.state.wrist_pos,self.pos+.01)
        self.push(3/60,self.pos) # immediately reverse
        torch.testing.assert_close(self.state.wrist_pos,self.pos)
        target=self.state.action(torch.ones(2,44),.15,torch.full((2,2),3/60))
        self.assertLessEqual(float((target-self.q).abs().max()),.15)
        self.push(.3,self.pos+3,stamp=3/60) # old packet must not move wrist
        torch.testing.assert_close(self.state.wrist_pos,self.pos)
        target=self.state.action(torch.ones(2,44),.15,torch.full((2,2),.3))
        torch.testing.assert_close(target,self.q)
        self.state.reset(torch.tensor([0]))
        self.assertEqual(float(self.state.history[0].abs().sum()),0)
        self.assertEqual(float(self.state.filtered[0].abs().sum()),0)
    def test_reject_future_sample_and_clear_zero_baseline(self):
        with self.assertRaises(ValueError):self.push(0.,stamp=.01)
        self.push(0.)
        self.state.action(torch.ones(2,44),.25,torch.zeros(2,2))
        target=self.state.action(torch.zeros(2,44),.25,torch.zeros(2,2),True)
        torch.testing.assert_close(target,self.q)
        self.assertEqual(float(self.state.filtered.abs().sum()),0.)


if __name__=='__main__':unittest.main()

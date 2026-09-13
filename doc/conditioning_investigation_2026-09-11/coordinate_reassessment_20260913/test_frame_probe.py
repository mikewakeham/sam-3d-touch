"""CPU regressions for pairing, negative controls and coordinate arithmetic."""
import copy
import json
from pathlib import Path
import unittest

import numpy as np
from frame_probe_core import check_observation, check_report_pair, check_loss_pair, normalization_check
from analyze_frame_probe import summarize

ROOT=Path(__file__).resolve().parent.parent/'oracle_upper_bound'


def fixture():
    rollouts={a:json.loads((ROOT/'rollouts_returned_20260913_143515'/a/'results.json').read_text()) for a in ('oracle','constant')}
    losses={a:json.loads((ROOT/'returned_20260913_133239'/a/'results.json').read_text()) for a in ('oracle','constant')}
    # Deliberately synthetic camera report based on saved schemas, not GPU evidence.
    camera=copy.deepcopy(rollouts['oracle']);camera['settings']['arm']='camera'
    camera['checkpoint_metadata']['conditioning_config']['oracle_point_frame']=False
    camera['coordinate_contract_passed']=True
    camera['rows']=copy.deepcopy(losses['oracle']['rows']);camera['loss_banks']=copy.deepcopy(losses['oracle']['banks'])
    for b in camera['inputs']:
        b['oracle_points_sha256']=b['points_sha256'];b['points_sha256']='synthetic_camera';b['tokens_sha256']='synthetic_camera'
    return camera,rollouts,losses


class FrameTests(unittest.TestCase):
    def test_normalization_and_wrong_coordinates(self):
        p=np.random.default_rng(29).normal(size=(8192,3))*[.3,.1,.2]+[.2,-.1,.4]
        c=(p.min(0)+p.max(0))/2;r=np.linalg.norm(p-c,axis=1).max();q=(p-c)/r
        result=normalization_check(p,q,c,1/r)
        self.assertLess(result['inverse_max_error'],1e-12)
        for changed in (q*1.1,q+[.1,0,0],q[:,[1,0,2]],q[:-1]):
            with self.assertRaises(AssertionError):normalization_check(p,changed,c,1/r)

    def test_intended_frame_difference_allowed(self):
        camera,rollouts,losses=fixture()
        check_report_pair(camera,rollouts['oracle']);check_loss_pair(camera,losses['oracle'])
        result=summarize(camera,rollouts,losses)
        for split in ('train','val'):
            self.assertEqual(result['summary'][split]['present']['shape']['oracle_minus_camera']['mean_difference'],0)
            self.assertEqual(result['summary'][split]['zero']['loss']['native']['mean_camera_minus_oracle'],0)

    def test_visual_target_oracle_replay_mismatch_rejected(self):
        for field in ('visual_sha256','target_sha256','oracle_points_sha256','sample_ids'):
            camera,rollouts,_=fixture();camera['inputs'][0][field]='wrong'
            with self.assertRaises(AssertionError):check_report_pair(camera,rollouts['oracle'])

    def test_rollout_missing_duplicate_noise_wrong_identity_rejected(self):
        for mutation in ('missing','duplicate','noise','identity','metric','failed_coordinates'):
            camera,rollouts,_=fixture()
            if mutation=='missing':camera['samples'].pop()
            elif mutation=='duplicate':camera['samples'].append(camera['samples'][0])
            elif mutation=='noise':camera['banks'][0]['seed']+=1
            elif mutation=='identity':camera['samples'][0]['object_id']='wrong'
            elif mutation=='metric':camera['samples'][0]['iou']=float('nan')
            else:camera['coordinate_contract_passed']=False
            with self.assertRaises(AssertionError):check_report_pair(camera,rollouts['oracle'])

    def test_loss_pairing_and_completeness(self):
        for mutation in ('missing','duplicate','noise','negative'):
            camera,_,losses=fixture()
            if mutation=='missing':camera['rows'].pop()
            elif mutation=='duplicate':camera['rows'].append(camera['rows'][0])
            elif mutation=='noise':camera['loss_banks'][0]['time_sha256']='wrong'
            else:camera['rows'][0]['losses'][0]=-1
            with self.assertRaises(AssertionError):check_loss_pair(camera,losses['oracle'])


if __name__=='__main__':unittest.main()

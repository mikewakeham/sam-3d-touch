"""CPU checks of the actual geometric contract, independent of model execution."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from orientation_coverage_protocol import rotations,rotate_points,rotate_grid,inverse_frame,schedule
from shared_orientation_protocol import transform_points


def check_launcher():
    import run_orientation_coverage_pair as launcher
    for count,prep_code in ((1,0),(2,0),(2,1)):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fit=root/'fit';probe=root/'probe';fit.mkdir();probe.mkdir()
            for p in (fit/'checkpoint_2000.pt',fit/'scope_shape_path_bundle.zip',probe/'results.json'):p.touch()
            events=[]
            class Process:
                def __init__(self,cmd,env):
                    phase=cmd[cmd.index('--phase')+1]
                    self.name=phase if phase=='prepare' else cmd[cmd.index('--arm')+1]
                    events.append(('start',self.name,env['CUDA_VISIBLE_DEVICES']))
                def wait(self):
                    events.append(('wait',self.name))
                    return prep_code if self.name=='prepare' else 0
            fake_torch=SimpleNamespace(cuda=SimpleNamespace(device_count=lambda:count))
            argv=['launcher','--fit-dir',str(fit),'--visual-probe-dir',str(probe),'--output-dir',str(root/'out')]
            with patch.dict(sys.modules,{'torch':fake_torch}),patch.object(sys,'argv',argv),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':','.join(['GPU-a','GPU-b'][:count])}),patch.object(launcher.subprocess,'Popen',Process):
                try:launcher.main()
                except SystemExit:
                    assert prep_code
                else:assert not prep_code
            names=[e[:2] for e in events]
            if prep_code:assert names==[('start','prepare'),('wait','prepare')]
            elif count==1:assert names==[('start','prepare'),('wait','prepare'),('start','control'),('wait','control'),('start','augmented'),('wait','augmented')]
            else:assert names==[('start','prepare'),('wait','prepare'),('start','control'),('start','augmented'),('wait','control'),('wait','augmented')]
            starts={e[1]:e[2] for e in events if e[0]=='start'}
            assert starts['prepare']=='GPU-a'
            if not prep_code:assert starts['control']=='GPU-a' and starts['augmented']==('GPU-b' if count==2 else 'GPU-a')
            status=json.loads((root/'out/pair_status.json').read_text())
            assert len(status)==(1 if prep_code else 3)


def main():
    rr=rotations();assert len(rr)==24 and len({r.tobytes() for r in rr})==24
    grid=np.zeros((64,64,64),dtype=bool)
    coords=np.array([[0,0,0],[63,7,18],[4,21,40],[32,31,16],[62,63,63]])
    grid[tuple(coords.T)]=True;centers=(coords+.5)/64-.5
    inv=np.eye(4);inv[:3,:3]=np.array([[0,-.83,0],[.83,0,0],[0,0,.83]]);inv[:3,3]=[.1,-.2,.03]
    for r in rr:
        out=rotate_grid(grid,r)
        expected=np.rint((rotate_points(centers,r)+.5)*64-.5).astype(int)
        assert out.sum()==grid.sum() and out[tuple(expected.T)].all()
        np.testing.assert_array_equal(rotate_grid(out,r.T),grid)
        np.testing.assert_allclose(transform_points(rotate_points(centers,r),inverse_frame(inv,r)),transform_points(centers,inv),atol=1e-15)
        assert any(np.array_equal(r.T,q) for q in rr)
        for q in rr:assert any(np.array_equal(r@q,k) for k in rr)
        # Signed axis changes commute with the existing bbox-center/radius
        # normalization; this is not asserted for arbitrary continuous rotations.
        def norm(x):
            y=x-(x.max(0)+x.min(0))/2;return y/np.linalg.norm(y,axis=1).max()
        np.testing.assert_allclose(norm(rotate_points(centers,r)),rotate_points(norm(centers),r),atol=1e-15)
    for bad in [np.diag([-1,1,1]),np.eye(3)*2,np.ones((3,3))]:
        try:rotate_grid(grid,bad)
        except ValueError:pass
        else:raise AssertionError('Accepted an invalid rotation')
    ss=schedule();assert ss==schedule() and len(ss)==1000
    for g in range(4):
        group=[x for x in ss if x['group']==g]
        assert len(group)==250 and sum(x['visual_dropped'] for x in group)==125
        counts=[sum(x['rotation_index']==ri for x in group) for ri in range(1,24)]
        assert min(counts)==5 and max(counts)==6
    assert all((x['rotation_index']!=0)==x['visual_dropped'] for x in ss)
    assert len({x['noise_seed'] for x in ss})==1000
    # Replay on all16 physical training supports, not just the synthetic cloud.
    here=Path(__file__).resolve().parent
    root=here/'shared_orientation_visuals_analysis/bundle'
    d=json.loads((root/'visual_probe/results.json').read_text());checks=0
    for f in d['target_frames']:
        if f['group']>=4:continue
        a=np.load(root/f'physical/{f["sample_id"]}.npy',allow_pickle=False)
        for r in rr:
            b=rotate_grid(a,r);np.testing.assert_array_equal(rotate_grid(b,r.T),a)
            assert b.sum()==a.sum();checks+=1
    check_launcher()
    print('PASS:24 proper rotations; grid/point/frame agreement; normalization; inverse/group closure; negative controls; balanced1000-update schedule;',checks,'actual-support rotations; one/two-GPU launcher order and preparation-failure stop (mocked processes).')


if __name__=='__main__':main()

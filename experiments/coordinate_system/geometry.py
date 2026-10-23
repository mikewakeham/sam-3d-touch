"""Estimate orientation from posed stage-1 geometry; cameras never initialize fits.

The residual/alternative-fit screen is a diagnostic, not a calibrated confidence.
Deformed or ambiguous shapes need not have an identifiable reference orientation.
"""
import itertools
import time
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

OUTPUT = Path(__file__).resolve().parent / 'outputs'
REFIT = True  # Set False to redraw plots from geometry_fits.jsonl.
WORKERS = 3


def angle(a, b):
    relative = np.asarray(a) @ np.swapaxes(np.asarray(b), -1, -2)
    return np.degrees(np.arccos(np.clip((np.trace(relative, axis1=-2, axis2=-1)-1)/2, -1, 1)))


def normalize(x):
    x = np.asarray(x, dtype=float) - np.mean(x, axis=0)
    return x / np.sqrt(np.mean(np.sum(x*x, axis=1)))


def basis(x):
    _, v = np.linalg.eigh(x.T @ x)
    v = v[:, ::-1]
    v[:, -1] *= np.linalg.det(v)
    return v


AXES = []
for perm in itertools.permutations(range(3)):
    for signs in itertools.product([-1, 1], repeat=3):
        r = np.zeros((3, 3)); r[np.arange(3), perm] = signs
        if np.linalg.det(r) > 0:
            AXES.append(r)


def fit(a, b):
    ac, bc = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((a-ac).T @ (b-bc))
    r = vt.T @ np.diag([1, 1, np.linalg.det(vt.T @ u.T)]) @ u.T
    return r, bc-r@ac


def icp(a, b, r, t, iterations):
    # Symmetric correspondences. Source and target have equal weight.
    at, bt = cKDTree(a), cKDTree(b)
    for _ in range(iterations):
        _, j = bt.query(a@r.T+t)
        _, i = at.query((b-t)@r)
        aa, bb = np.concatenate([a, a[i]]), np.concatenate([b[j], b])
        rn, tn = fit(aa, bb)
        change = np.linalg.norm(rn-r)+np.linalg.norm(tn-t)
        r, t = rn, tn
        if change < 1e-5:
            break
    return r, t


def score(a, b, r, t, at=None, bt=None):
    at = cKDTree(a) if at is None else at
    bt = cKDTree(b) if bt is None else bt
    # RMS symmetric Chamfer in units of RMS radius, not a pose confidence.
    return np.sqrt((np.mean(bt.query(a@r.T+t)[0]**2)
                    + np.mean(at.query((b-t)@r)[0]**2))/2)


def register(source, target, seed=0, random_starts=24, refine=6):
    start = time.time()
    rng = np.random.default_rng(seed)
    fulla, fullb = normalize(source), normalize(target)
    def sample(x, n):
        return x[rng.choice(len(x), min(n, len(x)), replace=False)]
    a, b = sample(fulla, 512), sample(fullb, 512)
    starts = [basis(fullb)@r@basis(fulla).T for r in AXES]
    starts += list(Rotation.random(random_starts, random_state=rng).as_matrix())
    candidates = []
    at, bt = cKDTree(a), cKDTree(b)
    for r in starts:
        r, t = icp(a, b, r, np.zeros(3), 40)
        candidates.append((float(score(a,b,r,t,at,bt)),r,t))
    # Refine distinct basins; full-cloud scoring avoids selecting sample noise.
    selected = []
    for s, r, t in sorted(candidates, key=lambda c:c[0]):
        if not selected or min(angle(r, x[1]) for x in selected) > 12:
            selected.append((s,r,t))
        if len(selected) == refine:
            break
    a, b = sample(fulla, 1600), sample(fullb, 1600)
    at, bt = cKDTree(fulla), cKDTree(fullb)
    results = []
    for _, r, t in selected:
        r, t = icp(a,b,r,t,70)
        results.append((float(score(fulla,fullb,r,t,at,bt)),r,t))
    results.sort(key=lambda c:c[0])
    s, r, t = results[0]
    alt = [x for x in results if angle(x[1],r)>20]
    return dict(rotation=r.tolist(), translation=t.tolist(), score=s,
                gap=alt[0][0]-s if alt else None,
                alternative_angle=float(angle(alt[0][1],r)) if alt else None,
                candidates=[dict(score=s,rotation=r.tolist(),translation=t.tolist()) for s,r,t in results],
                seconds=time.time()-start)


def fit_prediction(row, output):
    with np.load(output/row['stage1_path']) as shape, np.load(output/row['pose_path']) as pose:
        source = np.argwhere(shape['target']).astype(float)/64-.5
        prediction = np.argwhere(shape['prediction']).astype(float)/64-.5
        transform = pose['prediction_transform'].astype(float)
        placed = prediction@transform[:3,:3].T+transform[:3,3]
        result = register(source, placed)
        # Camera references enter only after geometry registration is finished.
        result['pose_rotation'] = (transform[:3,:3]/np.linalg.norm(transform[:3,:3],axis=0)).tolist()
        result['gt_rotation'] = pose['target_transform'][:3,:3].tolist()
        result['prediction_transform'] = transform.tolist()
        result['target_transform'] = pose['target_transform'].tolist()
    result.update({k:row[k] for k in ['model','camera_set','object_id','sample_id']})
    return result


def analyze(results, references, folder):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    rows=[]
    for r in results:
        distances=angle(r['rotation'],references)
        rows.append({k:r[k] for k in ['model','camera_set','object_id','sample_id','score','gap']} | dict(
            geometry_error_deg=float(angle(r['rotation'],r['gt_rotation'])),
            unposed_to_camera_deg=float(angle(np.asarray(r['pose_rotation']).T@np.asarray(r['rotation']),r['gt_rotation'])),
            geometry_nearest_old_deg=float(distances.min()),geometry_nearest_old_index=int(distances.argmin()),
            gt_nearest_old_deg=float(angle(r['gt_rotation'],references).min()),
            pose_nearest_old_deg=float(angle(r['pose_rotation'],references).min())))
    data=pd.DataFrame(rows)
    data['fit_screen']=(data.score<.1)&(data.gap>.01)
    data.to_csv(folder/'geometry_metrics.csv',index=False)
    summaries=[]
    for (model,cameras),group in data.groupby(['model','camera_set']):
        selected=group[group.fit_screen]
        summaries.append(dict(model=model,camera_set=cameras,total=len(group),screened=len(selected),
                              screened_percent=100*len(selected)/len(group),
                              median_geometry_error_deg=selected.geometry_error_deg.median(),
                              median_nearest_old_deg=selected.geometry_nearest_old_deg.median(),
                              median_gt_nearest_old_deg=selected.gt_nearest_old_deg.median()))
    summary=pd.DataFrame(summaries)
    summary.to_csv(folder/'geometry_summary.csv',index=False)
    print(summary.to_string(index=False))
    sns.set_theme()

    def save(name):
        plt.tight_layout()
        plt.savefig(folder/f'{name}.png',dpi=200)
        plt.savefig(folder/f'{name}.pdf')
        plt.close()

    fig,axes=plt.subplots(1,3,figsize=(15,4),sharex=True,sharey=True)
    for ax,model in zip(axes,['base','old','new']):
        group=data[(data.model==model)&(data.camera_set=='sampled')]
        selected=group[group.fit_screen]
        curves=pd.concat([pd.DataFrame({'degrees':selected.geometry_nearest_old_deg,'orientation':'Posed geometry'}),
                          pd.DataFrame({'degrees':selected.gt_nearest_old_deg,'orientation':'GT camera'})])
        sns.ecdfplot(data=curves,x='degrees',hue='orientation',ax=ax)
        ax.set(title=f'{model}, sampled ({len(selected)}/{len(group)} fits)',xlabel='Distance to nearest old rotation (degrees)')
    save('geometry_distance')

    selected=data[(data.model=='old')&(data.camera_set=='sampled')&data.fit_screen]
    curves=pd.concat([pd.DataFrame({'degrees':selected[column],'orientation':name}) for column,name in
                      [('pose_nearest_old_deg','Pose branch'),('geometry_nearest_old_deg','Posed geometry'),
                       ('gt_nearest_old_deg','GT camera')]])
    plt.figure(figsize=(6,4))
    sns.ecdfplot(data=curves,x='degrees',hue='orientation')
    plt.xlabel('Distance to nearest old rotation (degrees)')
    plt.title(f'old, sampled cameras ({len(selected)} fits)')
    save('geometry_old_comparison')

    plt.figure(figsize=(6,4))
    sns.barplot(data=summary,x='model',y='screened_percent',hue='camera_set',order=['base','old','new'])
    plt.xlabel('');plt.ylabel('Fits passing screen (%)');plt.ylim(0,100)
    save('geometry_coverage')

    selected=data[(data.model=='new')&(data.camera_set=='sampled')&data.fit_screen]
    curves=pd.concat([pd.DataFrame({'degrees':selected[column],'geometry':name}) for column,name in
                      [('unposed_to_camera_deg','Raw camera-frame shape'),('geometry_error_deg','After predicted pose')]])
    plt.figure(figsize=(6,4))
    sns.ecdfplot(data=curves,x='degrees',hue='geometry')
    plt.xlabel('Rotation error (degrees)');plt.title(f'new, sampled cameras ({len(selected)} fits)')
    save('geometry_new_pose')

    # This direction projection omits roll; the distance plots above use full SO(3).
    selected=data[(data.model=='old')&(data.camera_set=='sampled')&data.fit_screen]
    keys=set(selected.sample_id)
    records=[r for r in results if r['model']=='old' and r['camera_set']=='sampled' and r['sample_id'] in keys]
    reference_directions=-references[:,2,:]
    az=np.degrees(np.arctan2(reference_directions[:,1],reference_directions[:,0]))
    el=np.degrees(np.arcsin(np.clip(reference_directions[:,2],-1,1)))
    fig,axes=plt.subplots(1,3,figsize=(15,4),sharex=True,sharey=True)
    titles=['old, pose branch',f'old, posed geometry ({len(records)} fits)','GT sampled cameras']
    for ax,field,title in zip(axes,['pose_rotation','rotation','gt_rotation'],titles):
        directions=-np.asarray([r[field] for r in records]).reshape(-1,3,3)[:,2,:]
        sns.scatterplot(x=np.degrees(np.arctan2(directions[:,1],directions[:,0])),
                        y=np.degrees(np.arcsin(np.clip(directions[:,2],-1,1))),alpha=.5,ax=ax)
        sns.scatterplot(x=az,y=el,marker='X',s=130,color=sns.color_palette()[1],ax=ax,label='Old cameras')
        ax.set(title=title,xlabel='Azimuth (degrees)',ylabel='Elevation (degrees)',xlim=(-180,180),ylim=(-90,90))
    save('geometry_old_directions')


def main():
    import pandas as pd
    records=pd.read_csv(OUTPUT/'metrics.csv').to_dict('records')
    folder=OUTPUT/'analysis'
    folder.mkdir(exist_ok=True)
    path=folder/'geometry_fits.jsonl'
    if REFIT:
        with ProcessPoolExecutor(max_workers=WORKERS) as pool, path.open('w') as file:
            for i,result in enumerate(pool.map(partial(fit_prediction,output=OUTPUT),records)):
                file.write(json.dumps(result)+'\n');file.flush()
                print(i+1,'/',len(records),flush=True)
    results=[json.loads(line) for line in path.read_text().splitlines()]
    keys=lambda rows:{(r['model'],r['camera_set'],r['sample_id']) for r in rows}
    assert len(results)==len(records) and keys(results)==keys(records)
    references=np.load(OUTPUT/'reference_cameras.npz')['rotations']
    analyze(results,references,folder)


if __name__=='__main__':
    main()

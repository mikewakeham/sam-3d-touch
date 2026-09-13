"""Proper-rigid diagnostic registration. No scaling, reflection, trimming or warps."""
import itertools
import numpy as np
from scipy.spatial import cKDTree


def points(occupancy):
    assert occupancy.dtype == np.bool_ and occupancy.shape == (64, 64, 64)
    return (np.argwhere(occupancy).astype(np.float64)+.5)/64-.5


def subset(p, n):
    if len(p) <= n:
        return p
    return p[np.random.default_rng(37).choice(len(p), n, replace=False)]


def distances(p, q):
    return cKDTree(q).query(p, workers=1)[0], cKDTree(p).query(q, workers=1)[0]


def metrics(p, q):
    if not len(p) or not len(q):
        return {'empty': True, 'precision_1v': 0., 'recall_1v': 0., 'fscore_1v': 0.,
                'precision_2v': 0., 'recall_2v': 0., 'fscore_2v': 0.,
                'mean_distance': None, 'rms_distance': None, 'p95_distance': None}
    a, b = distances(p, q)
    result = {'empty': False, 'mean_distance': float((a.mean()+b.mean())/2),
              'rms_distance': float(np.sqrt((np.mean(a*a)+np.mean(b*b))/2)),
              'p95_distance': float(max(np.quantile(a,.95),np.quantile(b,.95)))}
    for v in (1, 2):
        precision, recall = float(np.mean(a <= v/64+1e-12)), float(np.mean(b <= v/64+1e-12))
        result.update({f'precision_{v}v': precision, f'recall_{v}v': recall,
                       f'fscore_{v}v': 2*precision*recall/(precision+recall) if precision+recall else 0.})
    return result


def proper_axes():
    output = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1,1), repeat=3):
            r = np.eye(3)[list(perm)]*np.array(signs)[:,None]
            if np.linalg.det(r) > .5:
                output.append(r)
    return output


def kabsch(a, b, weights):
    weights = weights/weights.sum()
    ca, cb = weights@a, weights@b
    u, _, vt = np.linalg.svd(((a-ca)*weights[:,None]).T@(b-cb))
    fix = np.eye(3); fix[-1,-1] = np.linalg.det(vt.T@u.T)
    r = vt.T@fix@u.T
    return r, cb-ca@r.T


def objective(p, q, qtree=None):
    a = (qtree or cKDTree(q)).query(p, workers=1)[0]
    b = cKDTree(p).query(q, workers=1)[0]
    return float((np.mean(a*a)+np.mean(b*b))/2)


def refine(p, q, r, t, iterations):
    qt = cKDTree(q); best = (objective(p@r.T+t,q,qt),r.copy(),t.copy())
    performed = 0
    for iteration in range(iterations):
        moved = p@r.T+t
        forward = qt.query(moved,workers=1)[1]
        backward = cKDTree(moved).query(q,workers=1)[1]
        a = np.concatenate((moved,moved[backward]))
        b = np.concatenate((q[forward],q))
        weights = np.concatenate((np.full(len(p),.5/len(p)),np.full(len(q),.5/len(q))))
        dr, dt = kabsch(a,b,weights)
        r, t = dr@r, t@dr.T+dt
        score = objective(p@r.T+t,q,qt); performed = iteration+1
        previous = best[0]
        if score < best[0]: best = score,r.copy(),t.copy()
        if abs(previous-score) < 1e-12: break
    return (*best, performed)


def register(p, q, known=None):
    if not len(p) or not len(q):
        return {'rotation':np.eye(3).tolist(),'translation':[0.,0.,0.],
                'metrics':metrics(p,q),'empty':True,'starts':0}
    pc,qc = subset(p,1024),subset(q,1024)
    pu=np.linalg.eigh(np.cov(pc.T))[1];qu=np.linalg.eigh(np.cov(qc.T))[1]
    # Eigenvector bases can have either handedness; force proper bases.
    if np.linalg.det(pu)<0:pu[:,0]*=-1
    if np.linalg.det(qu)<0:qu[:,0]*=-1
    candidates=[('identity',np.eye(3),np.zeros(3))]
    if known is not None:candidates.append(('known_inverse',known,np.zeros(3)))
    for label, matrices in [('axes',proper_axes()),('pca',[qu@a@pu.T for a in proper_axes()])]:
        for i,r in enumerate(matrices):
            candidates.append((f'{label}_{i}',r,q.mean(0)-p.mean(0)@r.T))
    qt=cKDTree(qc)
    ranked=sorted([(objective(pc@r.T+t,qc,qt),i,label,r,t) for i,(label,r,t) in enumerate(candidates)],key=lambda x:x[0])
    selected=ranked[:6]
    # Always refine analytic controls, even if subsampling disfavors them.
    for item in ranked:
        if item[2] in ('identity','known_inverse') and all(x[1]!=item[1] for x in selected):selected.append(item)
    coarse=[]
    for _,_,label,r,t in selected:
        score,r,t,it=refine(pc,qc,r,t,35);coarse.append((score,label,r,t,it))
    coarse.sort(key=lambda x:x[0]); pf,qf=subset(p,4096),subset(q,4096)
    fine=[]
    for _,label,r,t,it in coarse[:3]:
        score,r,t,n=refine(pf,qf,r,t,35)
        fine.append((objective(p@r.T+t,q),label,r,t,it+n))
    # Full-cloud scoring includes unrefined analytic hypotheses.
    for label,r,t in candidates[:2 if known is not None else 1]:
        fine.append((objective(p@r.T+t,q),label+'_unrefined',r,t,0))
    fine.sort(key=lambda x:x[0]);_,label,r,t,it=fine[0]
    assert np.allclose(r.T@r,np.eye(3),atol=1e-8) and np.isclose(np.linalg.det(r),1,atol=1e-8)
    return {'rotation':r.tolist(),'translation':t.tolist(),'metrics':metrics(p@r.T+t,q),
            'empty':False,'winner':label,'starts':len(candidates),'coarse_refined':len(coarse),
            'winning_iterations':it,'finalist_rms':[float(np.sqrt(x[0])) for x in fine],
            'rotation_angle_degrees':float(np.rad2deg(np.arccos(np.clip((np.trace(r)-1)/2,-1,1))))}

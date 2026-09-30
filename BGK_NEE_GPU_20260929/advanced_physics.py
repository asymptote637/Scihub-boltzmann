"""Geometry, open-boundary reconstruction and point sampling for experiments."""
import itertools
import json
import math


def parse_probes(value, dim):
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            value = json.loads(text) if text.startswith('[') else [part.replace('，', ',').split(',') for part in text.replace('；', ';').split(';')]
        except (ValueError, TypeError):
            raise ValueError('探针格式为 x,y[ ,z]; x,y[ ,z]，坐标范围 0–1') from None
    if not isinstance(value, (list, tuple)) or len(value) > 8:
        raise ValueError('最多设置 8 个探针')
    result = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != dim:
            raise ValueError(f'每个探针需要 {dim} 个归一化坐标')
        try:
            point = [float(v) for v in point]
        except (ValueError, TypeError, OverflowError):
            raise ValueError('探针坐标必须是有限数') from None
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in point):
            raise ValueError('探针坐标范围为 0–1')
        result.append(point)
    return result


def obstacle_centers(p, dim):
    axes = [[p['obstacle_'+a] + (i-(p['obstacle_count_'+a]-1)/2)*p['obstacle_spacing_'+a]
             for i in range(p['obstacle_count_'+a])] for a in 'xyz'[:dim]]
    return list(itertools.product(*axes))


def geometry_mask(xp, coords, sizes, p, require_resolved=False):
    mask = xp.zeros(coords[0].shape, dtype=bool)
    if p['scenario'] not in {'obstacle', 'cylinder_wake'}:
        return mask
    lengths=[n-(1 if a==0 and p['scenario']=='cylinder_wake' else 0) for a,n in enumerate(sizes)]
    radius = p['obstacle_size']*min(lengths)
    for center in obstacle_centers(p, len(sizes)):
        ds = [c+(0 if a==0 and p['scenario']=='cylinder_wake' else .5)-center[a]*lengths[a] for a,c in enumerate(coords)]
        inside = (sum(d*d for d in ds) <= radius**2 if p['obstacle']=='circle'
                  else xp.all(xp.stack([abs(d)<=radius for d in ds]), axis=0))
        if require_resolved and not bool(inside.any()):
            raise ValueError('阵列中存在未占据任何格点的障碍物；请增大网格或半径')
        mask |= inside
    return mask


def zou_he(solver):
    """D2Q9 west velocity / east density, v=0; executed on the chosen device.

    The channel x faces lie on boundary nodes; y faces lie half a link outside.
    At the four end nodes reconstruction follows wall streaming.
    """
    f, xp, p = solver.f, solver.xp, solver.p
    before = f[:,0,:].sum()+f[:,-1,:].sum()
    t = solver.iteration+1
    ramp = min(1,t/p['ramp_steps']) if p['ramp_steps'] else 1
    inlet = p['lid_speed']*ramp
    y = (xp.arange(solver.ny)+.5)/solver.ny
    u = inlet*(4*y*(1-y) if p['inlet_profile']=='parabolic' else xp.ones_like(y))
    a = f[:,0,:]
    rho = (a[:,0]+a[:,2]+a[:,4]+2*(a[:,3]+a[:,6]+a[:,7]))/(1-u)
    a[:,1] = a[:,3]+2*rho*u/3
    a[:,5] = a[:,7]+rho*u/6-.5*(a[:,2]-a[:,4])
    a[:,8] = a[:,6]+rho*u/6+.5*(a[:,2]-a[:,4])
    a = f[:,-1,:]
    rho = p['outlet_rho']
    u = (a[:,0]+a[:,2]+a[:,4]+2*(a[:,1]+a[:,5]+a[:,8]))/rho-1
    a[:,3] = a[:,1]-2*rho*u/3
    a[:,7] = a[:,5]-rho*u/6+.5*(a[:,2]-a[:,4])
    a[:,6] = a[:,8]-rho*u/6-.5*(a[:,2]-a[:,4])
    solver.boundary_exchange += float(f[:,0,:].sum()+f[:,-1,:].sum()-before)


def sample_probes(solver):
    # Sample the owning device, so reporting never needs a full host-field copy.
    h, p = solver, solver.p
    sizes = [solver.nx,solver.ny,solver.nz][:solver.dim]
    result = []
    for point in p['probes']:
        indices = [min(n-1,max(0,int(round(v*(n-1) if axis==0 and solver.open_boundary else v*n-.5))))
                   for axis,(v,n) in enumerate(zip(point,sizes))]
        idx = tuple(reversed(indices))
        actual = [(j/(n-1) if a==0 and solver.open_boundary else (j+.5)/n)
                  for a,(j,n) in enumerate(zip(indices,sizes))]
        row = dict(requested=point, coordinates=actual, valid=not bool(h.solid_mask[idx]))
        for field in ('ux','uy','uz','rho','T'):
            data = getattr(h,field,None)
            row[field] = float(data[idx]) if row['valid'] and data is not None else None
        row['pressure'] = (row['rho']-p['rho0'])*solver.lat.cs2 if row['valid'] else None
        result.append(row)
    return result

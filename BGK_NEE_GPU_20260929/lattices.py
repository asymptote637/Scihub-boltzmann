"""Hydrodynamic quadratures and separate scalar-transport velocity sets."""
from dataclasses import dataclass
from itertools import product
import math
import numpy as np


@dataclass(frozen=True)
class Lattice:
    name: str
    e: np.ndarray
    w: np.ndarray
    cs2: float
    opp: np.ndarray

    @property
    def dim(self):
        return self.e.shape[1]

    @property
    def q(self):
        return len(self.w)


def _make(name, e, w, cs2):
    e = np.asarray(e, dtype=np.int32)
    index = {tuple(v): k for k, v in enumerate(e)}
    return Lattice(name, e, np.asarray(w, dtype=np.float64), cs2,
                   np.array([index[tuple(-v)] for v in e], dtype=np.int32))


def lattice(name):
    if name == "D2Q9":
        return _make(name, [(0,0),(1,0),(0,1),(-1,0),(0,-1),(1,1),(-1,1),(-1,-1),(1,-1)],
                     [4/9]+[1/9]*4+[1/36]*4, 1/3)
    if name == "D3Q19":
        e = [v for v in product((-1,0,1), repeat=3) if sum(abs(a) for a in v) <= 2]
        return _make(name, e, [{0:1/3,1:1/18,2:1/36}[sum(abs(a) for a in v)] for v in e], 1/3)
    if name in {"D3Q27", "D2Q25"}:
        if name == "D3Q27":
            velocities, weights, dim, theta = (-1,0,1), {-1:1/6,0:2/3,1:1/6}, 3, 1/3
        else:
            # D1Q5 zero/one/three quadrature, lower-temperature root.
            # Matches Gaussian moments through order six (tensor product in 2D).
            theta = 1 - math.sqrt(2/5)
            w1, w3 = (9*theta-3*theta**2)/16, (3*theta**2-theta)/144
            velocities, dim = (-3,-1,0,1,3), 2
            weights = {-3:w3,-1:w1,0:1-2*w1-2*w3,1:w1,3:w3}
        e = list(product(velocities, repeat=dim))
        return _make(name, e, [math.prod(weights[a] for a in v) for v in e], theta)
    if name in {"D2Q5", "D3Q7"}:
        dim = 2 if name == "D2Q5" else 3
        cs2 = 1/3 if dim == 2 else 1/4
        e = [tuple([0]*dim)]
        for axis in range(dim):
            for sign in (-1,1):
                v = [0]*dim
                v[axis] = sign
                e.append(tuple(v))
        return _make(name, e, [1-dim*cs2]+[cs2/2]*(2*dim), cs2)
    raise ValueError(f"Unknown lattice: {name}")

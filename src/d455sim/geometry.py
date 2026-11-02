"""bpy-free geometry: T-slot profiles, L-brackets, surface sampling, symmetries, depth maths."""
from __future__ import annotations

import numpy as np

from .config import load_config

Poly = np.ndarray  # (N, 2)


# ---------------------------------------------------------------- 2D helpers
def signed_area(p: Poly) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def circle(cx: float, cy: float, d: float, n: int, ccw: bool = False) -> Poly:
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    p = np.stack([cx + d / 2 * np.cos(a), cy + d / 2 * np.sin(a)], 1)
    return p if ccw else p[::-1].copy()


def profile_outline(size: float, prm: dict) -> Poly:
    """CCW outer outline of a 4-slot T-slot profile (slot openings included, bore excluded)."""
    h, o, lip = size / 2, prm["opening"] / 2, prm["lip"]
    t, dep = prm["t_width"] / 2, prm["depth"]
    edge = np.array([(h, h), (o, h), (o, h - lip), (t, h - lip), (t, h - dep),
                     (-t, h - dep), (-t, h - lip), (-o, h - lip), (-o, h), (-h, h)])[:-1]
    rot = np.array([[0, -1], [1, 0]], float)
    pts, cur = [], edge
    for _ in range(4):
        pts.append(cur)
        cur = cur @ rot.T
    return np.concatenate(pts)


def profile_cross_section(size: float, prm: dict, segs: int) -> tuple[Poly, list[Poly]]:
    """(outer CCW, [bore CW])."""
    return profile_outline(size, prm), [circle(0, 0, prm["bore_d"], segs)]


def profile_area(size: float, prm: dict, segs: int) -> float:
    outer, holes = profile_cross_section(size, prm, segs)
    return signed_area(outer) + sum(signed_area(h) for h in holes)


def _seg_cross(a, b, c, d, eps=1e-12) -> bool:
    def o(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    d1, d2, d3, d4 = o(c, d, a), o(c, d, b), o(a, b, c), o(a, b, d)
    return d1 * d2 < -eps and d3 * d4 < -eps


def _inside(pt, poly: Poly) -> bool:
    x, y = pt
    res = False
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            res = not res
    return res


def triangulate(outer: Poly, holes: list[Poly]) -> tuple[np.ndarray, np.ndarray]:
    """Ear-clipping with hole bridging. Returns (verts2d, tris) indexing the stacked rings."""
    rings = [outer] + list(holes)
    verts = np.concatenate(rings)
    offs = np.cumsum([0] + [len(r) for r in rings])
    edges = [(offs[k] + i, offs[k] + (i + 1) % len(r)) for k, r in enumerate(rings) for i in range(len(r))]
    poly = list(range(offs[1]))
    order = sorted(range(len(holes)), key=lambda k: -holes[k][:, 0].max())
    for k in order:
        hidx = list(range(offs[k + 1], offs[k + 2]))
        m = max(hidx, key=lambda i: verts[i, 0])
        mi = hidx.index(m)
        cands = sorted(set(poly), key=lambda v: np.linalg.norm(verts[v] - verts[m]))
        for v in cands:
            mid = (verts[v] + verts[m]) / 2
            ok = _inside(mid, outer) and not any(_inside(mid, h) for h in holes)
            ok = ok and not any(
                _seg_cross(verts[m], verts[v], verts[a], verts[b]) for a, b in edges)
            if ok:
                break
        else:
            raise RuntimeError("hole bridging failed")
        edges.append((m, v))
        loop = hidx[mi:] + hidx[:mi] + [m]
        pos = poly.index(v)
        poly = poly[:pos + 1] + loop + [v] + poly[pos + 1:]
    tris = []
    eps = 1e-14
    guard = 0
    while len(poly) > 3 and guard < 10000:
        guard += 1
        n = len(poly)
        for i in range(n):
            a, b, c = poly[i - 1], poly[i], poly[(i + 1) % n]
            pa, pb, pc = verts[a], verts[b], verts[c]
            cr = (pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0])
            if cr <= eps:
                continue
            bad = False
            for q in poly:
                pq = verts[q]
                if any(np.allclose(pq, p, atol=1e-12) for p in (pa, pb, pc)):
                    continue
                d1 = (pb[0] - pa[0]) * (pq[1] - pa[1]) - (pb[1] - pa[1]) * (pq[0] - pa[0])
                d2 = (pc[0] - pb[0]) * (pq[1] - pb[1]) - (pc[1] - pb[1]) * (pq[0] - pb[0])
                d3 = (pa[0] - pc[0]) * (pq[1] - pc[1]) - (pa[1] - pc[1]) * (pq[0] - pc[0])
                if d1 >= -eps and d2 >= -eps and d3 >= -eps:
                    bad = True
                    break
            if not bad:
                tris.append((a, b, c))
                poly.pop(i)
                break
        else:
            raise RuntimeError("ear clipping stalled")
    tris.append(tuple(poly))
    return verts, np.array(tris)


def extrude(outer: Poly, holes: list[Poly], z0: float, z1: float) -> tuple[np.ndarray, np.ndarray]:
    """Prism mesh (V, F tri) from a polygon with holes; outward-facing normals."""
    v2, tris = triangulate(outer, holes)
    n = len(v2)
    V = np.concatenate([np.c_[v2, np.full(n, z0)], np.c_[v2, np.full(n, z1)]])
    F = [t[::-1] for t in tris] + [tuple(i + n for i in t) for t in tris]
    off = 0
    for r in [outer] + list(holes):
        m = len(r)
        for i in range(m):
            a, b = off + i, off + (i + 1) % m
            F += [(a, b, n + b), (a, n + b, n + a)]
        off += m
    return V, np.array(F)


# ---------------------------------------------------------------- 3D parts
def _geom_cfg(geom_cfg: dict | None) -> dict:
    return geom_cfg if geom_cfg is not None else load_config("scene")["geometry"]


def profile_mesh(size: float, length: float, geom_cfg: dict | None = None):
    """Long axis = z, centred at the origin."""
    g = _geom_cfg(geom_cfg)
    prm = g["profile_params"][str(int(round(size * 1000)))]
    outer, holes = profile_cross_section(size, prm, g["circle_segments"])
    return extrude(outer, holes, -length / 2, length / 2)


def _rect(a0, a1, b0, b1) -> Poly:
    return np.array([(a0, b0), (a1, b0), (a1, b1), (a0, b1)], float)


def bracket_mesh(geom_cfg: dict | None = None):
    """L-bracket: leg1 along +x (z in [0,t]), leg2 along +z (x in [0,t]); y centred on width."""
    b = _geom_cfg(geom_cfg)["bracket"]
    segs = _geom_cfg(geom_cfg)["circle_segments"]
    L, w, t = b["leg_m"], b["width_m"], b["thickness_m"]

    def holes_for(a_lo, a_hi):
        ca = a_lo + b["hole_along_frac"] * (a_hi - a_lo)
        return [circle(ca, (f - 0.5) * w, b["hole_d_m"], segs) for f in b["hole_across_fracs"]]

    V1, F1 = extrude(_rect(0, L, -w / 2, w / 2), holes_for(0, L), 0, t)
    # leg 2 local (a=across, b=along, c=thickness) -> world (c, a, b): cyclic, keeps orientation
    # swapping columns turns (along, across) into (across, along) and reverses winding; undo it
    h2 = [h[:, ::-1][::-1].copy() for h in holes_for(t, L)]
    V2l, F2 = extrude(_rect(-w / 2, w / 2, t, L), h2, 0, t)
    V2 = np.stack([V2l[:, 2], V2l[:, 0], V2l[:, 1]], 1)
    return np.concatenate([V1, V2]), np.concatenate([F1, F2 + len(V1)])


def mesh_for_spec(obj_spec: dict, geom_cfg: dict | None = None):
    if obj_spec["class"] == "profile":
        s, _, length = obj_spec["dims_m"]
        return profile_mesh(s, length, geom_cfg)
    return bracket_mesh(geom_cfg)


def sample_model_points(obj_spec: dict, n: int, geom_cfg: dict | None = None, seed: int = 0) -> np.ndarray:
    """Area-weighted surface samples (N,3) in the object frame."""
    V, F = mesh_for_spec(obj_spec, geom_cfg)
    tri = V[F]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(F), n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    t = tri[idx]
    return t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])


def _rz(a: float) -> np.ndarray:
    T = np.eye(4)
    T[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    return T


def symmetries(cls: str, geom_cfg: dict | None = None) -> list[np.ndarray]:
    """4x4 object-frame symmetry transforms (identity included). Mirrors have det -1."""
    g = _geom_cfg(geom_cfg)["symmetry"]
    if cls == "profile":
        out = [_rz(k * np.pi / 2) for k in range(4)]  # includes the 180 deg turn
        if g["profile_end_flip"]:
            flip = np.diag([1.0, -1.0, -1.0, 1.0])  # 180 deg about x
            out += [flip @ s for s in out]
        return out
    out = [np.eye(4)]
    if g["bracket_mirror"]:
        out.append(np.diag([1.0, -1.0, 1.0, 1.0]))
    if g["bracket_leg_swap"]:
        D = np.eye(4)
        D[:3, :3] = [[0, 0, 1], [0, -1, 0], [1, 0, 0]]  # proper rotation, det +1
        out += [D] + ([D @ out[1]] if g["bracket_mirror"] else [])
    return out


# ---------------------------------------------------------------- depth maths
def dist_to_zdepth(dist: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Euclidean ray distance -> z-depth for a pinhole camera (integer pixel centres)."""
    h, w = dist.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    x = (u - K[0, 2]) / K[0, 0]
    y = (v - K[1, 2]) / K[1, 1]
    return (dist / np.sqrt(x * x + y * y + 1.0)).astype(np.float32)

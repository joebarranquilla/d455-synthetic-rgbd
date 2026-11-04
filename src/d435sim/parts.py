"""bpy meshes and materials built from geometry.py."""
from __future__ import annotations

import bpy
import blenderproc as bproc
import numpy as np

from . import geometry as geo


def make_mesh(name: str, V: np.ndarray, F: np.ndarray) -> MeshObject:
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(V.tolist(), [], F.tolist())
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    for p in mesh.polygons:
        p.use_smooth = False
    return bproc.types.MeshObject(obj)


def _u(rng: np.random.Generator, lohi) -> float:
    return float(rng.uniform(lohi[0], lohi[1]))


def sample_material(cls: str, mat_cfg: dict, rng: np.random.Generator):
    """Create a Principled material; returns (material, meta dict)."""
    kind = str(rng.choice(mat_cfg[cls]))
    t = mat_cfg["types"][kind]
    color = [_u(rng, c) for c in t["base_color"]]
    meta = {"type": kind, "roughness": _u(rng, t["roughness"]), "metallic": _u(rng, t["metallic"]),
            "anisotropy": _u(rng, t["anisotropy"])}
    m = bproc.material.create(f"{kind}_{rng.integers(1 << 30)}")
    m.set_principled_shader_value("Base Color", color + [1.0])
    m.set_principled_shader_value("Roughness", meta["roughness"])
    m.set_principled_shader_value("Metallic", meta["metallic"])
    m.set_principled_shader_value("Anisotropic", meta["anisotropy"])
    return m, meta


def make_part(cls: str, geom_cfg: dict, mat_cfg: dict, rng: np.random.Generator, inst_id: int, cls_id: int):
    """One part object + its spec (dims_m, material meta)."""
    if cls == "profile":
        size = float(rng.choice(geom_cfg["profile_sizes_m"]))
        length = _u(rng, geom_cfg["profile_length_m"])
        spec = {"class": "profile", "dims_m": [size, size, length]}
    else:
        b = geom_cfg["bracket"]
        spec = {"class": "bracket", "dims_m": [b["leg_m"], b["width_m"], b["leg_m"]]}
    V, F = geo.mesh_for_spec(spec, geom_cfg)
    obj = make_mesh(f"{cls}_{inst_id}", V, F)
    mat, meta = sample_material(cls, mat_cfg, rng)
    obj.replace_materials(mat)
    obj.set_cp("inst_id", inst_id)
    obj.set_cp("class_id", cls_id)
    spec["material"] = meta
    return obj, spec

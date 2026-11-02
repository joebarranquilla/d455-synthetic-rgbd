"""Scene construction: table, bin, rigid-body dropped parts, lights, world, eye-in-hand camera."""
from __future__ import annotations

import blenderproc as bproc
import numpy as np

from . import parts


def _u(rng, lohi) -> float:
    return float(rng.uniform(lohi[0], lohi[1]))


def _passive_box(name: str, size, loc, color_cfg: dict):
    o = bproc.object.create_primitive("CUBE", scale=[s / 2 for s in size], location=loc)
    o.set_name(name)
    m = bproc.material.create(name + "_mat")
    m.set_principled_shader_value("Base Color", list(color_cfg["base_color"]) + [1.0])
    m.set_principled_shader_value("Roughness", color_cfg["roughness"])
    m.set_principled_shader_value("Metallic", color_cfg["metallic"])
    o.replace_materials(m)
    o.set_cp("inst_id", 0)
    o.set_cp("class_id", 0)
    o.enable_rigidbody(active=False, collision_shape="BOX")
    return o


def build_scene(cfg: dict, rng: np.random.Generator) -> list[dict]:
    """Build and settle a scene; returns [{obj, spec, id}] for surviving parts. Bin centre is the origin."""
    bproc.clean_up()
    tb, bn = cfg["table"], cfg["bin"]
    _passive_box("table", tb["size_m"], [0, 0, -tb["size_m"][2] / 2], cfg["materials"]["table"])
    bx, by = bn["size_m"]
    wh, wt = bn["wall_height_m"], bn["wall_thickness_m"]
    for i, (sz, loc) in enumerate([
            ([bx + 2 * wt, wt, wh], [0, by / 2 + wt / 2, wh / 2]), ([bx + 2 * wt, wt, wh], [0, -by / 2 - wt / 2, wh / 2]),
            ([wt, by, wh], [bx / 2 + wt / 2, 0, wh / 2]), ([wt, by, wh], [-bx / 2 - wt / 2, 0, wh / 2])]):
        _passive_box(f"bin_wall{i}", sz, loc, cfg["materials"]["bin"])

    pc, classes = cfg["parts"], cfg["classes"]
    n = int(rng.integers(pc["count"][0], pc["count"][1] + 1))
    items = []
    for i in range(1, n + 1):
        cls = "bracket" if rng.random() < pc["bracket_prob"] else "profile"
        obj, spec = parts.make_part(cls, cfg["geometry"], cfg["materials"], rng, i, classes[cls])
        obj.set_location([rng.uniform(-bx / 2 + 0.05, bx / 2 - 0.05), rng.uniform(-by / 2 + 0.05, by / 2 - 0.05),
                          wh + _u(rng, pc["drop_height_m"])])
        obj.set_rotation_euler(rng.uniform(0, 2 * np.pi, 3))
        obj.enable_rigidbody(active=True, mass=pc["mass_kg"], friction=pc["friction"],
                             collision_shape=pc["collision_shape"][cls])
        spec["id"] = i
        items.append({"obj": obj, "spec": spec})

    lc = cfg["lights"]
    for _ in range(int(rng.integers(lc["count"][0], lc["count"][1] + 1))):
        light = bproc.types.Light()
        light.set_type("AREA")
        light.set_location([rng.uniform(-1, 1) * lc["xy_spread_m"], rng.uniform(-1, 1) * lc["xy_spread_m"],
                            _u(rng, lc["height_m"])])
        light.set_rotation_euler([np.pi, 0, 0])  # area light faces -z by default; point at table
        light.blender_obj.data.size = _u(rng, lc["size_m"])
        light.set_energy(_u(rng, lc["energy_w"]))

    w = cfg["world"]
    bproc.renderer.set_world_background([_u(rng, c) for c in w["color"]], w["strength"])

    bproc.object.simulate_physics_and_fix_final_poses(
        min_simulation_time=pc["sim_time_s"][0], max_simulation_time=pc["sim_time_s"][1],
        check_object_interval=1, substeps_per_frame=10, solver_iters=20)

    m = pc["keep_margin_m"]
    kept = []
    for it in items:
        p = it["obj"].get_location()
        if abs(p[0]) <= bx / 2 + m and abs(p[1]) <= by / 2 + m and p[2] > -m:
            kept.append(it)
        else:
            it["obj"].delete()
    return kept


def sample_camera_pose(cfg: dict, rng: np.random.Generator) -> np.ndarray:
    """T_world_cam in CV convention (x right, y down, z forward)."""
    c = cfg["camera"]
    h, tilt = _u(rng, c["height_m"]), np.deg2rad(_u(rng, c["tilt_deg"]))
    az, yaw = rng.uniform(0, 2 * np.pi), np.deg2rad(_u(rng, c["yaw_deg"]))
    target = np.append(rng.uniform(-1, 1, 2) * c["target_jitter_m"], 0.0)
    pos = target + h * np.array([np.sin(tilt) * np.cos(az), np.sin(tilt) * np.sin(az), np.cos(tilt)])
    pos = pos + rng.uniform(-1, 1, 3) * c["pos_jitter_m"]
    z = target - pos
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 1.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    x, y = np.cos(yaw) * x + np.sin(yaw) * y, -np.sin(yaw) * x + np.cos(yaw) * y
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, pos
    return T


def cv_to_gl(T_cv: np.ndarray) -> np.ndarray:
    return T_cv @ np.diag([1.0, -1.0, -1.0, 1.0])

# Rules (apply to every task)
- Python 3.10+, numpy, Open3D, OpenCV, pandas, pyyaml. NO torch. Functions over classes. Type hints.
- All parameters in configs/*.yaml. No magic numbers. Everything seeded from config.
- Units SI (m). Depth = float32 z-depth (not ray distance), 0 = invalid. Camera frame: x right, y down, z forward.
- Modules importing bpy: only render.py, scene.py, parts.py. geometry.py (profile cross-section, bracket mesh
  maths, model point-cloud sampling) must be bpy-free and shared by parts.py and loc.py.
- Splits: dev (tune/debug) and test (frozen, never used for tuning). Finalize parameters using dev before running test experiments.
- Do not read files you don't need or echo file contents. Reply: code, then summary <=5 lines. No questions:
  choose the simplest option, leave `# ASSUMPTION: ...`. No new deps without 1-line justification.
- Tests: pytest, <10 s, no GPU/Blender. Never commit data/ or outputs/.

# Sample format data/<split>/<id:05d>/
rgb.png, depth_gt.npy (float32), normals.npy (float16, camera frame), inst.png (uint16, 0=bg),
cls.png (uint8: 0 bg,1 profile,2 bracket,3 other),
meta.json (K, T_world_cam, baseline_m, seed, objects[{id,class,dims_m,T_cam_obj,material:{roughness,metallic,anisotropy,type},n_visible_px}])
Later: depth_<LEVEL>.npy per sensor level; ir_l.png, ir_r.png only for IR subset.
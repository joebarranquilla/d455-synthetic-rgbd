PY ?= .venv/bin/python
BP ?= .venv/bin/blenderproc
# Optional BLENDER=/path/to/dir: custom Blender (macOS: a copy of Blender.app outside an .app bundle, with a 4.2 symlink to Contents/Resources/4.2)
BLENDER ?=
BPFLAGS = $(if $(BLENDER),--custom-blender-path $(BLENDER),)

.PHONY: deps sample test

deps:
	$(PY) -m pip install -r requirements.txt
	$(BP) pip install pyyaml $(BPFLAGS)  # BlenderProc bundles opencv-contrib and numpy

sample:
	rm -rf outputs/sample && $(BP) run $(BPFLAGS) scripts/render_samples.py --split dev --n 5 --seed 0 --out outputs/sample
	$(PY) scripts/montage.py outputs/sample/dev 5 outputs/montage.png

test:
	$(PY) -m pytest -q

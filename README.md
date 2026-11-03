# manim_screw

Interactive ManimGL scenes that visualize screw axes and pose constraint fields, built on a fork of 3b1b/manim.

## Overview

This repository is a fork of [3b1b/manim](https://github.com/3b1b/manim) (ManimGL, package `manimgl`). The `manimlib` package and the other upstream files are kept as in upstream (based on upstream commit `e529838`, October 2025). On top of that, this fork adds a few standalone, interactive 3D scripts for exploring joint constraints: a screw axis that can be switched between locked, revolute, prismatic and screw motion, and mouse-driven "Pose B" scenes that show guidance fields toward a constraint manifold around an axis through a fixed "Pose A".

## What this fork adds

| File | Scene | Description |
| --- | --- | --- |
| `screwnet.py` | `ScrewNetAxisKeyboardLM` | Screw axis given by a direction `l` and moment `m = p* × l`, controlled from the keyboard. A red ball follows the mouse and is projected onto the current constraint; a live matplotlib window ("Pose Tracking (Realtime)") plots the red-ball, projection and blended poses and their errors. Also listens for parameter updates over UDP on `127.0.0.1:5005`. |
| `ui.py` | – | Tkinter control panel ("ScrewNet Control UI") with `yaw`, `pitch`, `alpha`, `beta` sliders and a mode selector (1 Locked, 2 Revolute, 3 Prismatic, 4 Screw); sends JSON over UDP to `127.0.0.1:5005`. |
| `run_all.sh` | – | Starts `ui.py` and `screwnet.py` together; Ctrl+C stops both. |
| `pose_test.py` | `PoseMouseLocalConstraintField` | Pose B follows the mouse. Shows a constraint-only local guidance field for a cylinder-plane constraint (distance to axis = `L`, axial coordinate = `s_des`) and the projection of Pose B onto the constraint manifold. |
| `pose_fixed.py` | `PoseMouseLocalConstraintField` | Same setup with an A-centered guidance field (radial + axial + tangential/orbit terms) that is blended into the mouse-following motion as Pose B gets close to the ring. |
| `pose_mixed.py` | `PoseMouseLocalConstraintField` | Shows both the A-centered guidance field and a B-centered constraint-only local field. |
| `pose_screw.py` | `PoseMouseLocalConstraintField` | Screw-joint constraint: radius `rho = L` and helical constraint `s = s0 + pitch * phi` (`pitch = 0` is revolute; a very large pitch is treated as prismatic). |
| `RUN_CMD.md` | – | Notes with run commands. |

### Controls

`screwnet.py`:

- `1`–`4` (top row or numpad): mode 1 Locked, 2 Revolute, 3 Prismatic, 4 Screw
- `J` / `L`: axis yaw; `I` / `K`: axis pitch
- `C` / `Z` and `S` / `X`: move the axis position (`alpha` / `beta`) in the plane perpendicular to it

`pose_mixed.py`, `pose_screw.py`: `E` / `Q` raise / lower the mouse target's z coordinate.

## Requirements and installation

Install this fork in editable mode from the repository root (this installs the `manimgl` command from the local `manimlib`):

```bash
pip install -e .
```

ManimGL's system requirements still apply (FFmpeg, OpenGL, and on Linux Pango with its development headers); see the [upstream installation guide](https://github.com/3b1b/manim#installation). `screwnet.py` uses matplotlib (listed in `requirements.txt`) and `ui.py` uses Tkinter.

## Usage

Each script defines a single scene. Examples (flags as used in `RUN_CMD.md`):

```bash
manimgl pose_test.py  PoseMouseLocalConstraintField -p -ql
manimgl pose_fixed.py PoseMouseLocalConstraintField -p -ql
manimgl pose_mixed.py PoseMouseLocalConstraintField -p -ql
manimgl pose_screw.py PoseMouseLocalConstraintField -p -ql
manimgl screwnet.py   ScrewNetAxisKeyboardLM -p -ql
```

`-p` is presenter mode (the scene stays paused at `wait` calls), `-q` is quiet and `-l` renders at 480p.

To run the ScrewNet scene together with the slider UI:

```bash
./run_all.sh
```

This runs `python ui.py` and `manimgl screwnet.py ScrewNetAxisKeyboardLM -p --resolution 1024x768`.

## Repository layout

- `screwnet.py`, `ui.py`, `run_all.sh`, `pose_*.py`, `RUN_CMD.md`: this fork's additions (see above)
- `manimlib/`: the ManimGL library (upstream)
- `example_scenes.py`: upstream ManimGL example scenes
- `docs/`: upstream ManimGL documentation sources
- `logo/`: upstream logo files
- `setup.py`, `setup.cfg`, `pyproject.toml`, `requirements.txt`, `MANIFEST.in`: upstream packaging files (package `manimgl`)

## ManimGL documentation

For general ManimGL usage, see the upstream [README](https://github.com/3b1b/manim) and the [documentation](https://3b1b.github.io/manim/).

## Status

Experimental visualization scripts; last updated in January 2026. Not synced with later upstream changes.

## License and credits

ManimGL is developed by 3Blue1Brown ([3b1b/manim](https://github.com/3b1b/manim)). This fork keeps upstream's MIT License; see [LICENSE.md](LICENSE.md).

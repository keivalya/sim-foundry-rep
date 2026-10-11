# SimFoundry reproduction — Northeastern Explorer

Reproducing NVlabs/SimFoundry (arXiv 2606.28276) on Explorer: reconstruct a real
video into an OmniGibson scene, then drive a robot arm in it.

Everything for this work lives under `/scratch/pandya.kei/simfoundry-repro`.
Deep detail, every failure and its cause, is in **[docs/STATUS.md](docs/STATUS.md)**.

---

## Checking what is running

```bash
bash /scratch/pandya.kei/simfoundry-repro/jobs/status.sh
```

Prints the Slurm queue, jobs finished in the last 24 h, every blue-cube campaign
summary, all scored rollouts ranked, and where the key videos are. Reads only
Slurm and disk, so it works from any terminal or session.

Long-running work is launched so it does NOT depend on an interactive shell:
`jobs/bluecube_launch.sbatch` runs `jobs/bluecube_campaign.sh` **on the cluster**,
which submits the rollouts plus a `--dependency=afterany` finisher that scores
them, picks the best and submits the render. Submit it and close the terminal.

---

## Start here — the results

| what | where |
|---|---|
| **SO-101 pick-and-place video** (the good one) | `results/videos/so101_pick_place.mp4` |
| Franka pick-and-place video | `results/videos/franka_pick_place.mp4` |
| **Real vs reconstructed sim**, same camera | `results/reference_scene/real_vs_sim_aligned.png` |
| Reconstructed scene, 6 angles | `results/reference_scene/multiview.png` |
| SO-101 stills | `results/so101/filmstrip.png` |
| **SO-101 episode from 6 angles** (incl. wrist cam) | `results/so101/multiview/videos/` |
| 2x3 grid video of all 6 angles | `results/so101/multiview/videos/so101_grid_3x2.mp4` |
| Labelled 6-angle contact sheet | `results/so101/multiview/contact_sheet.png` |
| SO-101 high-res inspection renders | `results/so101/inspect/` |
| **pi0.5 closed-loop, live wrist + top cam** | `results/so101/vla/framed/videos/so101_pi05_live_pair.mp4` |
| pi0.5 live frames, labelled filmstrip | `results/so101/vla/framed/live_filmstrip.png` |
| pi0.5 commanded vs achieved joint traces | `results/so101/vla/framed/episode_traces.png` |
| pi0.5 episode from 6 offline angles | `results/so101/vla/framed/multiview_preview/videos/` (full-res `multiview/` lands later) |
| **pi0.5 picks up the blue cube** — 8-step sequence | `results/so101/vla/bluecube/d1_shift/pick_sequence.png` |
| the same episode, live policy cameras | `results/so101/vla/bluecube/d1_shift/videos/so101_pi05_live_pair.mp4` |
| blue-cube rollouts, scored | `results/so101/vla/bluecube/*/score.json` |
| sim vs the real training frames, both cameras | `results/so101/vla/bluecube/*/real_vs_sim_both_cams.png` |
| sim joints vs all 50 training episodes | `results/so101/vla/bluecube/*/vs_training.png` |

```bash
# copy the main video to your laptop
scp pandya.kei@login.explorer.northeastern.edu:/scratch/pandya.kei/simfoundry-repro/results/videos/so101_pick_place.mp4 .
```

### What actually works

* **Pipeline A reproduced end to end** on `fruits_demo`, `fruits_pro`, and on your
  own `reference_video.mp4` → `scene_data/reference_demo/`.
* **Scripted pick-and-place** in the reconstructed scene, with both a Franka and
  an SO-101. SO-101 results: `orange_block_toy` grasped and moved 0.078 m,
  `red_boat_toy` 0.178 m, 0.00 mm joint drift.
* The reconstructed paper cup (12.8 cm) is too wide for the SO-101's ~5–6 cm jaw
  and outside its 0.32 m reach. The Franka picks the block and boat but not the cup.
* **pi0.5 closed-loop** driving the SO-101 from live Isaac camera images — 450
  control steps, 45 inferences, 0.196 s/call, 0.045 s/`env.step()`, zero joint
  drift. The plumbing works end to end. The only SO-101-shaped pi0.5 checkpoint
  is a **blue-cube-in-gray-box** fine-tune and this scene has neither, so the
  arm folds back and parks and **moves nothing**. See the section below — that
  is reported as a result, not hidden.

### Not reproducible (established, not assumed)

Sim-to-real robotics results (code unreleased), YAM scenes (robot asset never
published), Pipeline B digital cousins on a free Gemini tier (no `flux` branch
in `B_augmentation`).

---

## Layout

```
simfoundry-repro/
├── README.md              you are here
├── docs/STATUS.md         full handover: every fix, every gotcha, re-run commands
├── results/               all output (see table above)
│   ├── videos/            final videos
│   ├── reference_scene/   reconstruction of reference_video.mp4 (+ extract/, views/)
│   ├── so101/             SO-101 stills, inspection renders, probe data
│   ├── franka/            Franka stills
│   └── nvidia_gallery/    NVIDIA's own published images, fetched for comparison
├── robot/                 SO-101 asset + OmniGibson robot class
│   ├── so101_usd/so101.usd    the loadable asset (built by tools/deinstance_so101.py)
│   ├── so101_src/             upstream URDF + STLs
│   └── so101_og.py            OmniGibson SO101 class (source of truth)
├── policies/              pick_and_place.py, probe_so101.py, inspect_so101_scene.py
│                          wrist_cam_probe.py, pi05_closed_loop.py (the VLA sim client)
├── vla/                   pi0.5 half of the VLA bridge
│   ├── pi05_server.py         policy server (openpi env)
│   ├── so101_units.py         LeRobot units <-> radians (+ round-trip self-test)
│   ├── plot_episode.py        commanded vs achieved traces
│   ├── smoke_so101.py         offline inference smoke test
│   └── ckpt_so101_pi05_lerobot061/   the SO-101 pi0.5 fine-tune, lerobot-0.6.1-ready
├── tools/                 renderers and scene editing (see below)
├── pipeline/              Pipeline A drivers, env install, keepalive
├── jobs/                  sbatch wrappers
├── assets_hf/             nadunRanawaka1/simfoundry-assets (343 MB) — NVIDIA's
│                          published scenes, used by tools/make_gallery.py
├── tools/legacy/          superseded renderers, kept for provenance
└── symlinks → logs/, SimFoundry/, scene_data/, conda_envs/
```

`robot/so101_og.py` is the source of truth for the robot class; it is **copied**
into `SimFoundry/deps/.../omnigibson/robots/so101.py` to be importable. After
editing it, re-copy:

```bash
cp robot/so101_og.py SimFoundry/deps/BEHAVIOR-1K/OmniGibson/omnigibson/robots/so101.py
```

---

## Re-running things

`$E` below is `conda_envs/simfoundry-editor/bin/python` (CPU-only utility env).
Anything touching Isaac Sim must go through `sbatch` on the `sharing` partition
(RT cores, 1 h cap).

```bash
cd /scratch/pandya.kei/simfoundry-repro
E=conda_envs/simfoundry-editor/bin/python
D=scene_data/reference_demo/s14_og
```

**Reconstruct a video into a scene** (hours, `gpu` partition):
```bash
sbatch --export=ALL,SCENE=myscene,VIDEO=/path/to/video.mp4,\
STAGES=1b+2+3+4+5+6+7+8,FAITHFUL=1,KEEPALIVE=1 pipeline/run_pipelineA.sbatch
```

**Put the SO-101 into a reconstructed scene:**
```bash
$E tools/swap_robot_so101.py --scene-dir scene_data/reference_demo
```

**Run pick-and-place and record poses** (~3 min — no rendering):
```bash
sbatch --export=ALL,SCENE=reference_demo,\
SCENE_JSON=$D/reconstructed_og_scene_so101.json,\
SETTLED_JSON=$D/settled_poses_so101.json,\
PICK=orange_block_toy+red_boat_toy,\
REACH=0.32,MAX_GRASP_W=0.05,DOWN_AXIS=-z,ROBOT_H=0.35,\
GRIP_OPEN=0.364,GRIP_RELEASE=1.0,APPROACH_H=0.06,LIFT_H=0.09,POS_TOL=0.007,\
PLACE_AZ=55,POSES_ONLY=1,VIDEO_NAME=pick_place_so101 jobs/run_pick_place.sbatch
```

**Render that episode to video** (offline, no GPU, ~1 h):
```bash
$E tools/render_episode.py \
  --poses scene_data/reference_demo/pick_place/pick_place_so101_poses.npz \
  --out results/videos/so101_pick_place.mp4 --width 1120 --height 700 --fps 15
```

**Render the same episode from 6 angles at once**, including a wrist-eye camera
bolted to the `gripper` link (offline, no GPU, ~1.5 h for all 327 frames):
```bash
$E tools/render_episode_multiview.py --width 720 --height 450 --fps 15 --stride 1
# preview in ~1/4 the time:  --stride 4 --fps 8
```
One mp4 per angle plus a 2x3 grid video land in
`results/so101/multiview/videos/`, the labelled contact sheet in
`results/so101/multiview/contact_sheet.png`, and the timings in
`render_info.json`. The per-frame geometry build is done once and shared by all
six cameras; only the backface cull and the rasterisation are per-camera.

To redo **one** camera without repeating the ~98 min full pass, render just that
camera and then rebuild the videos and sheet from the PNGs already on disk
(`--skip-render` deliberately leaves `render_info.json` alone, since its own
timings would be meaningless):
```bash
$E tools/render_episode_multiview.py --cameras wrist --width 720 --height 450 --fps 15
$E tools/render_episode_multiview.py --skip-render --width 720 --height 450 --fps 15
```
Pass the same `--width/--height/--stride/--start` both times or the frame
indices stop lining up with the other cameras.

The wrist camera **reproduces the live sensor**, so the offline view stands in
for what a policy's wrist camera would see. Its defaults are read out of the
`UsdGeom.Camera` at `/so101_new_calib/gripper/gripper_cam` in
`robot/so101_usd/so101_wristcam.usd`: gripper-local position
`(-0.005, 0.060, -0.062)`, view direction `(0, -0.7071, -0.7071)` (the gripper's
-Z approach axis tilted 45 deg toward -Y, onto the jaw pads), focal length
13.5 mm over a 20.955 x 15.716 mm aperture = fx = fy = 412.312 px at 640x480.
Override with `--wrist-mount` / `--wrist-forward` / `--wrist-up` /
`--wrist-intrinsics`, or fall back to a plain `--wrist-fov`.

**Mind the axis convention.** A USD camera looks along its own local **-Z** with
**+Y up**; `sfrender` is OpenCV (+x right, +y **DOWN**, +z forward). The mapping
is `(right, down, forward) = (X, -Y, -Z)`. Getting it backwards is silent, so it
is checked against geometry, not asserted.

Everything above is re-derived from the asset and re-checked against physics by:
```bash
$E tools/check_wrist_cam.py
```
which (1) diffs the renderer's constants against the authored prim's own
transform and aperture, (2) confirms -Z is the view direction because the unit
vector from the camera to the jaw pads dots **+0.976** with it and **-0.976**
with +Z, and checks the derived OpenCV basis is orthonormal and right-handed,
and (3) projects the grasped object into the wrist camera on every frame where
it is actually in the jaws: `orange_block_toy` f95-177 and `red_boat_toy`
f225-300, **159/159 frames inside the image**, 6-16 deg off the optical axis.

The underlying -Z approach axis was established independently of that asset:
the gripper link's visual meshes span local z = +0.001 .. -0.1044 with nothing
on the +Z side, and `gripper_pos + R_gripper @ (0,0,-0.101)` closes to 5 mm of
`orange_block_toy` and 9 mm of `red_boat_toy` at the grasp frames while the +Z
mirror stays ~200 mm away.

Re-check the two things the speed of all this rests on:
```bash
$E tools/check_fast_raster.py --frames 0,140,300
```

---

## Live Isaac rendering, and the SO-101 wrist camera

**Isaac renders this scene correctly and reliably. Use it in the loop.**
Established 2026-10-06 by job 10882722 (`policies/wrist_cam_probe.py`): 105
consecutive control steps, three live cameras, 315 RGB+depth frames, no crash,
no dropped frame, 0.081 s per `env.step()` (12 Hz) at 3 x 640x480.

### The wrist camera

`tools/add_wrist_cam.py` authors a `UsdGeom.Camera` at
`/so101_new_calib/gripper/gripper_cam` into `robot/so101_usd/so101_wristcam.usd`.
OmniGibson's `BaseRobot._load_sensors` wraps any `Camera` prim that is a DIRECT
child of a link into a `VisionSensor`, so it needs no per-script plumbing:

```python
env   = og.Environment(configs=cfg)      # scene json robot0 obs_modalities incl. "rgb"
robot = env.robots[0]
cam   = robot.sensors["robot0:gripper:Camera:0"]
obs, *_ = env.step(action)
rgb = obs["robot0"]["robot0:gripper:Camera:0"]["rgb"]      # (480, 640, 4) uint8
```

Pose and intrinsics are NVIDIA's own, from
`source/sim_to_real_so101/tasks/task_env_cfg.py` (kept at
`robot/so101_usd/nvidia_task_env_cfg.py.ref`) — note their WITH-camera USD,
`SO-ARM101-USD.usd`, contains **no Camera prim at all**, only the camera body
geometry, so there was nothing to import from it.

| | |
|---|---|
| prim | `/World/scene_0/controllable__so101__robot0/gripper/gripper_cam` |
| offset in the `gripper` link | `(-0.005, 0.06, -0.062)` m, `Rx(-45 deg)` (USD/OpenGL camera frame) |
| resolution | 640 x 480 |
| focal / aperture | 13.5 mm / 20.955 mm horizontal |
| **K** | `fx = fy = 412.31`, `cx = 320`, `cy = 240` |
| FOV | 75.63 deg x 60.41 deg |
| clipping | 0.01 .. 1e6 m |

Cross-check on the offset: the lens mesh of NVIDIA's `camera_mount`, measured in
the gripper link frame, is centred at `(-0.0039, 0.0518, -0.0599)` — 8 mm from
the offset they configure. The runtime intrinsics come from
`SO101_WRIST_CAM_CFG` in `robot/so101_og.py`, **not** from the USD: OmniGibson
overwrites focal length / aperture / clipping from `sensor_config` on load, and
that dict must nest them under `sensor_kwargs` (`create_sensor()` has no
`**kwargs`, so a flat dict raises `unexpected keyword argument 'image_width'`).

### What the "detached jaw" actually was

`tools/check_render_lag.py` back-projects a camera's `depth_linear` to world
using the pose the sim reports with that image, and measures how far each link's
reported surface is from the nearest surface Isaac actually drew. The floor plane
comes back at z = +0.0010 m (external camera) and z = -0.0002 m (wrist camera),
which validates the reported camera poses.

| link | render err, median | when the link is **still** | when it moves > 40 mm/step | r(motion, err) |
|---|---|---|---|---|
| base | 1.46 mm | 1.46 mm | — | — |
| upper_arm | 2.52 mm | 2.29 mm | — | 0.61 |
| lower_arm | 6.24 mm | 2.08 mm | — | 0.75 |
| wrist | 7.41 mm | 1.35 mm | 26.3 mm | 0.81 |
| gripper | 16.0 mm | 15.5 mm | 50.4 mm | 0.76 |
| jaw | 17.3 mm | 3.47 mm | 28.2 mm | **0.92** |

**It is latency, not corruption.** Everything collapses to 1-3 mm when the link
is stationary and grows in proportion to how fast it is moving. The same shows
up in the wrist camera's own viewpoint: with the camera still the floor comes
back 0.066 mm and 0.024 deg off; with the camera moving 42.6 mm and 9.43 deg per
step it comes back 36.6 mm and 8.55 deg off — i.e. the image is **~0.9 control
steps stale**. The jaw is the fastest-moving link (up to 88 mm per step in this
sweep), so a stale frame puts it up to ~7 cm from the pose logged beside it,
which is exactly what "a detached floating lever" looks like when you compare a
rendered frame against poses logged at that step.

For a VLA this is ordinary observation latency, not a blocker. If you need the
image and the state to agree exactly, either slow the arm down, raise
`physics_frequency` / `rendering_frequency`, or pair each image with the pose
read one step earlier.

Honest loose end: `gripper` keeps a ~15 mm residual even in low-motion frames,
the only link that does. Most likely a measurement artefact — it is the one link
whose visible surface is a thin lever routinely occluded by a *moving*
neighbour (the jaw) — but it was not run to ground.

Re-run the whole thing:
```bash
D=/scratch/pandya.kei/repos/SimFoundry/Data/reference_demo/s14_og
sbatch --export=ALL,SCENE_JSON=$D/reconstructed_og_scene_so101.json,\
STEPS=120,STRIDE=3,ATTEMPTS=4 jobs/run_wrist_cam.sbatch
$E tools/check_render_lag.py --probe-dir results/so101/wristcam/run_<jobid>
```

---

## The one thing to know before touching the renders

**The Isaac SIGSEGV is one bad compute node, not this scene.** Corrected
2026-10-06 — the earlier claim here ("Isaac's RTX renderer corrupts this scene
and eventually crashes it") was wrong about the cause.

Kit writes a py-spy dump of every thread next to each crash minidump, in
`<appdata>/local/data/Kit/OmniGibson/3.8/*.py.txt`. In all four SIGSEGVs the main
thread is inside **`SimulationApp.__init__`** — `_wait_for_viewport` (jobs
10873560, 10872515, 10882447) or `_prepare_ui` (10872275) — with
`_launch_app → og.Environment.__init__` below it. That is Kit bringing up the
RTX scene renderer and its viewport. **No scene, no robot and no camera exist
yet**, so the SO-101 asset cannot be the cause.

What those four have in common is the node:

| node | jobs | result | kernel | NVIDIA driver |
|---|---|---|---|---|
| **d4050** | 10871335, 10872275, 10872515, 10873560, 10882447, 10882524 | **0 / 6 ok** — every one segfaulted in `librtx.scenedb` during app startup (or hung in `_wait_for_viewport` for 25 min first) | 5.14.0-611.5.1.**el9_7** | **595.58.03** |
| d4049 | 10871158, 10872270, 10876100, 10875292 | 3 / 4 ok (the 4th crashed only in `og.shutdown()`, after its data was written) | 5.14.0-362.13.1.el9_3 | 570.86.15 |
| d4047 | 10871766 | 1 / 1 ok | el9_3 | 570.86.15 |
| d4051 | 10875341, 10882579, 10882646 | no startup crash (one shutdown crash) | el9_3 | 570.86.15 |

d4050 is the only `sharing` l40s node that has been upgraded ahead of the fleet;
Isaac Sim 5.1 / Kit 107.3 predates the 595 driver branch. **So: exclude it.**
`jobs/run_wrist_cam.sbatch` carries `#SBATCH --exclude=d4050`, prints the driver
version, and warns if it is ≥ 580. `d4074`/`d4075` (rtx6000) are on the same new
kernel — assume they are affected too. Check before adding a node:

```bash
scontrol show node d40NN | grep -oP 'OS=Linux \K\S+'   # el9_7 => suspect
```

### `no_proxy='*'` hangs Isaac Sim. Do not set it.

Found the hard way, job **10884015**. The cluster's squid proxy intercepts
`127.0.0.1`, so the obvious defence for a localhost client is `no_proxy='*'`.
Do not. Omniverse Client makes an outbound HTTPS call during scene load.
Behind squid it fails **fast**. With the proxy disabled it tries to reach AWS
**directly**, this cluster blackholes the packets, and it sits in TCP SYN retry
forever:

```
ss -tnp -> SYN-SENT  10.99.104.51:45012 -> 52.92.202.250:443  users:(("python",...))
```

The symptom is maximally unhelpful: the job hangs inside `og.Environment()`
immediately after `Welcome to OmniGibson`, **GPU at 0%, all 415 threads idle,
no log line, no traceback and no timeout** — for 20 minutes, until the walltime
kills it. It looks exactly like a renderer deadlock and is not one.

Scope it to loopback only, which is what `jobs/run_pi05_closed_loop.sbatch`
does, and which is how every Isaac job that has ever worked here was set up:

```bash
export no_proxy='127.0.0.1,localhost,::1' NO_PROXY='127.0.0.1,localhost,::1'
```

Better still, do not rely on proxy variables at all for local IPC: a **raw TCP
socket is never proxied**, which is why the policy bridge uses one instead of
HTTP.

Separately, `og.shutdown()` can segfault *during teardown* on the good nodes too
(10875292, 10875341). Harmless if, and only if, every output is written before
shutdown — which is why the scripts write their artefacts and then shut down.

The asset is *not* the problem either — it measures identical to NVIDIA's
original on mesh counts, vertex positions, materials, winding and visibility,
with every visual mesh at 0.00 mm offset from its link and 0.00 mm joint drift
across a full episode. Several hours were lost "fixing" the asset before that
was established.

The offline route below is still the right way to make *videos* (it is ~10×
faster and needs no GPU), but it is no longer the only way to get pixels — see
`policies/wrist_cam_probe.py` for live in-the-loop camera images.

So videos are made in two steps:

1. run the episode with `POSES_ONLY=1` — Isaac renders **nothing**, just logs
   each link's and object's world pose per frame (also ~10× faster);
2. rebuild the video with `tools/render_episode.py`, which transforms the USD
   visual meshes by those logged poses and rasterises in numpy.

(The corrupted Isaac-rendered video was deleted on request; the evidence for
this is the crash in the job log, not a kept artefact.)

Two traps inside `tools/render_episode.py`:
* `--decimate` is **disabled**. Each link's geometry is a concatenation of
  disconnected parts, and quadric simplification across that shreds the arm.
* Backface culling is on and is safe — verified pixel-identical to full
  resolution, and it is what makes the render tractable.

`tools/sfrender.py` also has an opt-in `fast_small=N` argument (default **0**,
i.e. off, so `render_views.py` and `render_aligned.py` are untouched). It pulls
triangles whose screen bbox is <= N px out of the per-triangle Python loop and
rasterises them in one vectorised, size-tiered batch. Episode frames are almost
entirely sub-pixel triangles, so this is a 9-15x speedup. It is a real speedup,
not a quality trade: `tools/check_fast_raster.py` renders frames both ways and
asserts **zero** differing 8-bit pixels (max float difference exactly 0.0), and
re-asserts the same for backface culling. Run it if you touch the rasteriser.

---

## pi0.5 closed-loop on the SO-101 (the VLA bridge)

A real VLA driving the arm inside the reconstructed scene, with **live Isaac
camera images in the loop** — not a scripted waypoint policy, and not replayed
frames.

> **The checkpoint is for a different task, and it shows.** The only pi0/pi0.5
> checkpoint that fits a 5-DOF SO-101 is a community fine-tune,
> `pdhinaka/pi05_so101_blue_cube_gray_box_cleaned_20k`, trained on *"put the
> blue cube in the gray box"*. This scene contains an orange block, a red boat
> toy and a white paper cup — no blue cube, no gray box. **Do not read the
> arm's behaviour as task performance.** What is demonstrated here is the
> plumbing: live images + live joint state → pi0.5 → executable joint targets,
> closed, for hundreds of steps. Task success was never on the table and is not
> claimed.

### Architecture: two envs, one job

OmniGibson (numpy 1.26) and LeRobot (numpy 2.2, plus a conflicting transformers
pin) **cannot share a conda env**. So:

| | env | file |
|---|---|---|
| policy **server** | `conda_envs/openpi` | `vla/pi05_server.py` |
| sim **client** | `conda_envs/simfoundry` | `policies/pi05_closed_loop.py` |
| both, one Slurm job, one L40S | | `jobs/run_pi05_closed_loop.sbatch` |

They talk over a **raw TCP socket on loopback**, length-prefixed `.npz`. Two
reasons it is not HTTP:

* The cluster's **squid proxy intercepts `127.0.0.1`** — `http_proxy` /
  `https_proxy` are set in every job. A proxied localhost call comes back as a
  squid **403 that reads exactly like an application auth error**. A raw socket
  cannot be proxied, so the failure mode cannot occur. Note that the obvious
  alternative fix — `no_proxy='*'` — **hangs Isaac for 20 minutes with no log
  line**; see the section above. `no_proxy` here is scoped to loopback only.
* numpy 2.2 pickles arrays as `numpy._core.*`, which numpy 1.26 cannot
  unpickle. The NPY container is version-stable and is read with
  `allow_pickle=False`.

bf16 pi0.5 is ~8.8 GB and the l40s has 46 GB, so it sits alongside Isaac
comfortably.

### Units: LeRobot normalized motor units ↔ OmniGibson radians

`vla/so101_units.py`. LeRobot's `so101_follower` declares body joints as
`RANGE_M100_100` and the gripper as `RANGE_0_100`, and `Motor.normalize()` is an
affine map from the motor's **calibrated mechanical range** onto that interval.
The sim's equivalent of that range is the joint's USD limit pair, so:

```
body     n = 200 * (q - lo) / (hi - lo) - 100
gripper  n = 100 * (q - lo) / (hi - lo)
```

| joint (USD) | LeRobot name | limits (rad) |
|---|---|---|
| Rotation | shoulder_pan | ±1.9199 |
| Pitch | shoulder_lift | ±1.7453 |
| Elbow | elbow_flex | −1.7453 … +1.5708 |
| Wrist_Pitch | wrist_flex | ±1.6581 |
| Wrist_Roll | wrist_roll | ±2.7925 |
| Jaw | gripper | −0.1745 … +1.7453 |

Limits are **read off the live robot** at run time (`SO101Units.from_sim_limits`)
and cross-checked against the table; a mismatch warns loudly rather than
silently changing the units underneath the policy.

```bash
conda_envs/simfoundry/bin/python -I vla/so101_units.py   # round-trip self-test
```

The **gripper map is exact**, not approximate: `MultiFingerGripperController` in
`"smooth"` mode computes `u = lo + (cmd+1)/2 * (hi-lo)` over the jaw's own
limits, which is the identical affine map, so a 0..100 gripper action converts
with zero error.

**The one assumption, stated plainly:** this assumes the sim joint's zero and
sign convention agrees with whatever physical arm recorded the training set.
That is **not verifiable** — the checkpoint ships no calibration, and a real
SO-101's calibration is per-unit. If the fine-tune's arm had, say, `elbow_flex`
wired with the opposite sign, every action would come back mirrored on that
joint and nothing here would detect it.

### The `np.digitize` guard

pi0.5 writes the state into the **prompt** as 256 discrete bins
(`"Task: {text}, State: {bins};\nAction: "`). lerobot's binning is an
**unclipped `np.digitize`**, so a normalized value outside `[-1, 1]` yields bin
`-1` and a malformed prompt — visible in the original smoke test as
`'State: 149 121 152 -1 115 118'`.

Normalization is `QUANTILES`, so clamping the **raw** state into the
checkpoint's own `[q01, q99]` (read out of
`policy_preprocessor_step_3_normalizer_processor.safetensors`, dumped to
`vla/so101_state_quantiles.json`) is exactly equivalent to clamping the
normalized state to `[-1, 1]`. The server does that and **counts every clamp**,
because a clamp means the arm is in a configuration this checkpoint never saw.

### Controller

`tools/swap_robot_so101.py` grew a `--controller {ik,joint}` flag.

* `ik` (default, unchanged) — `InverseKinematicsController`, `pose_delta_ori`.
  This is what `policies/pick_and_place.py` drives; it is byte-for-byte the same
  scene JSON as before.
* `joint` — `JointController`, `motor_type="position"`,
  `use_delta_commands=False`, no input/output scaling, **no smoothing filter**.
  The command vector *is* the absolute joint angles in radians.

pi0.5 emits **absolute joint positions**, so the VLA path needs `joint`. Feeding
those targets to the IK controller would read them as Cartesian pose *deltas*
and fling the arm.

### Cameras

The checkpoint wants two: `wrist_cam` and `top_cam`.

* **wrist** — the existing `robot0:gripper:Camera:0` (640×480, fx=fy=412.31).
* **top** — a new scene-frame `VisionSensor` added by
  `policies/pi05_closed_loop.py` as an env `external_sensors` entry, framed on
  the bounding box of {robot base, all objects}, azimuth 180° / elevation 62°
  (`--top-az` / `--top-el` / `--top-dist`), i.e. over the robot's shoulder onto
  the work area. 640×480, f=17 mm.

Both are resized to 224×224 before being sent (the model resizes to 224
internally anyway; this just keeps the payload small).

**Never request a `seg_*` modality** — it needs a BEHAVIOR dataset that is not
installed, and every `get_obs()` then raises `FileNotFoundError`.

### Chunk execution — a real choice, so it is a flag

pi0.5 returns a chunk of **50** absolute joint targets. `--exec-steps` decides
how many of them are executed before re-planning:

| value | what it means |
|---|---|
| 50 | pure open loop — the policy acts blind for 50/30 s ≈ 1.7 s and never sees the consequence of its own motion inside a chunk |
| 1 | re-plan every control step — one 4.1 B forward pass (10 denoising steps) per 1/30 s, ~15× slower than real time |
| **10 (default)** | 1/3 s of open-loop execution, re-planning at ~3 Hz — a prefix of the chunk, which is what openpi's own serving loop does |

Control runs at **30 Hz** (`--action-frequency`), matching the training set's
likely rate: 18 736 frames over 50 episodes = 375 frames/episode, which for a
~13 s SO-101 demo is ~30 fps, LeRobot's record default. Playing the chunk at the
wrong rate would stretch or compress every motion the policy learned.

### Running it

```bash
cd /scratch/pandya.kei/simfoundry-repro
D=/scratch/pandya.kei/repos/SimFoundry/Data/reference_demo/s14_og

# once: write the JointController variant of the scene
conda_envs/simfoundry-editor/bin/python tools/swap_robot_so101.py \
  --scene-dir scene_data/reference_demo --controller joint --out-suffix _so101_vla

# the rollout (ONE job: policy server + sim client on one l40s)
sbatch --export=ALL,SCENE_JSON=$D/reconstructed_og_scene_so101_vla.json,\
STEPS=450,EXEC_STEPS=10,ATTEMPTS=2 jobs/run_pi05_closed_loop.sbatch
```

That is the whole thing — it writes the live `wrist_cam` / `top_cam` mp4s
itself. Then, offline on CPU (no GPU, ~2 h for 450 frames x 6 angles):

```bash
sbatch --export=ALL,RUN=results/so101/vla/run_<jobid>,W=720,H=450,FPS=30 \
       jobs/run_multiview_vla.sbatch
# quick look instead: add MV_OUT=<dir>,STRIDE=5,W=640,H=400,FPS=10
conda_envs/simfoundry/bin/python -I vla/plot_episode.py --run results/so101/vla/run_<jobid>
```

### It ran. Here is exactly what happened.

Job **10885036**, `d4051`, one l40s: **450 closed-loop control steps, 45
inferences**, 476 s of rollout. Policy load 96 s, Isaac load 183 s.

| | |
|---|---|
| policy inference | **0.196 s/call** mean, 0.215 s p95 |
| `env.step()` | **0.045 s** mean (≈22 Hz) with both cameras live |
| joint limits read in-sim vs the recorded table | **max diff 0.00004 rad** |
| articulation integrity over all 450 frames | **0.000 mm drift** on all six joints |
| objects displaced | **0.0 mm** — all four |

**Four rollouts were run, as four parallel jobs.** Three of them vary only the
chunk-execution strategy, and **they all end in the same place**:

| run | job | `--exec-steps` | steps | inferences | `shoulder_lift` command clipped |
|---|---|---|---|---|---|
| `run_10885036` | 10885036 | 10 | 450 | 45 | 253/450 |
| `openloop50` | 10885405 | 50 (pure open loop) | 450 | 9 | 230/450 |
| `replan1` | 10885406 | 1 (re-plan every step) | 120 | 120 | 104/120 |
| `framed` | 10885537 | 10, better `top_cam` angle | 450 | 45 | 200/450 |

So the behaviour below is **the policy's**, not an artefact of how much of the
chunk gets executed. `framed` (`--top-az 140 --top-el 55`) is the one to watch:
the default `--top-az 180` puts the camera directly behind the robot, so the arm
occludes the very objects the camera is meant to show. Everything else about the
two runs is the same.

**What the arm actually did:** it folded back on itself and parked. It drove
`shoulder_lift` down to the simulator's joint limit by about step 200 and held
there for the rest of the episode. It never approached an object, never closed
the gripper on anything, and **moved nothing — 0.0 mm for the orange block, the
red boat and the cup.** The wrist camera spends the second half of the episode
pointing at empty space. This is what an out-of-distribution rollout looks like,
and it is the expected and honest outcome: there is no blue cube and no gray box
in this scene for a blue-cube-in-gray-box policy to act on.

The control itself is clean, which is the part that was being built: in
`results/so101/vla/run_10885036/episode_traces.png` the commanded (red) and
achieved (blue) curves lie on top of each other on every joint, at every step,
with the single exception noted below. No divergence, no thrash; the largest
commanded single-step joint jump in 450 steps was 0.165 rad.

### Two findings worth keeping

**1. The checkpoint's `shoulder_lift` lives below the SO-101's USD joint limit.**
The policy keeps commanding `shoulder_lift ≈ -102 … -104`, and the sim clips it
at **-100**, which is what `Pitch`'s ±1.7453 rad limit maps to. That is not
noise: the checkpoint's own `observation.state.q01` **and** `q50` for that joint
are **-104.19** and **-104.17** — the training set spends most of its time at a
`shoulder_lift` the sim arm cannot physically reach. So the affine
limits→[-100,100] map is right in *form*, but the physical arm that recorded the
data was calibrated to a **wider** range than the USD's limits. Anyone
fine-tuning for this sim should either re-calibrate the map against the dataset
quantiles or widen the joint limit; the mismatch is real and measurable.

**2. The policy partly collapses onto the training set's modal pose.** The
checkpoint's `q50` state is `[-5.18, -104.17, 1.05, 76.53, -13.93, 2.87]`. The
rollout settles at `shoulder_pan ≈ -8`, `shoulder_lift` at the limit (commanding
≈ -104) and `gripper ≈ 3` — three joints essentially *on* the training median.
`elbow_flex` and `wrist_flex` do not. Given an image it has never seen, it falls
back toward where the data usually is.

State clamping into `[q01, q99]` (the `np.digitize` guard) fired on
`gripper` 17/45 inferences and `wrist_roll` 1/45; the four other joints were
never out of the checkpoint's range, which is a mild sanity check on the unit
mapping.

### Outputs

```
results/so101/vla/framed/            exec-steps=10, 450 steps   <- WATCH THIS ONE
  live_filmstrip.png                   labelled 2x6 strip, both policy input cameras
  videos/so101_pi05_live_pair.mp4      top + wrist side by side, LIVE Isaac frames
  multiview_preview/   offline 6-angle render (hero, 2 orbits, close, top, wrist-eye)
                       + so101_grid_3x2.mp4 and a labelled contact_sheet.png.
                       stride 5 / 640x400 / 10 fps -- 14 min to produce.
  multiview/           the same six angles at stride 1 / 720x450 / 30 fps.
                       ~2 h of CPU (job 10885931); identical camera rig.
  ... (all the files listed below, too)
results/so101/vla/run_10885036/      exec-steps=10, 450 steps, default top_cam angle
  videos/so101_pi05_live_pair.mp4      top + wrist side by side, LIVE Isaac frames
  videos/so101_pi05_top_cam.mp4        640x480, 450 frames, 30 fps = real time
  videos/so101_pi05_wrist_cam.mp4
  wrist_cam/rgb_####.png  top_cam/rgb_####.png    every frame the policy saw
  poses.npz          keys/poses -- the SAME format tools/render_episode.py and
                     tools/render_episode_multiview.py consume
  episode.npz        state, action, commanded radians, clamp + clip masks, timings
  episode_traces.png commanded vs achieved per joint, with the training band
  meta.json          camera intrinsics, settings, full summary
results/so101/vla/openloop50/        exec-steps=50
results/so101/vla/replan1/           exec-steps=1
```


---

---

## The blue-cube scene — building the task the checkpoint was actually trained for

The section above ran a blue-cube-in-gray-box policy in a scene with no blue
cube and no gray box. This section builds that scene and runs it properly.
**It still does not succeed**, and the honest result is below, with numbers.

### Two things were wrong before, and both are fixed

**1. A joint-calibration mismatch that clipped the policy's commands.** The
checkpoint's `shoulder_lift` **action** range is −109.71 … +40.79 and its
`wrist_flex` action range is +57.85 … **+102.68** — both outside the ±100 that
the SO-101's USD limits map to, because LeRobot's `Motor.normalize()` is
unclamped and the real arm travelled past the range its calibration sweep
covered. This cannot be fixed by widening the USD limit, because the normalized
value is *defined* by the range you normalize against. The two roles are now
separated (`vla/so101_units.py`):

* `NORM_LIMITS` pins ±100 to the joint's **nominal** limits and never changes;
* the **physical** limits are widened by `tools/widen_so101_limits.py`
  (`Pitch` lower −100° → −115°, `Wrist_Pitch` upper +95° → +105°, `Elbow` upper
  +90° → +95°) into `robot/so101_usd/so101_wristcam_wide.usd`, so the policy's
  whole commanded range is reachable.

`conda_envs/simfoundry/bin/python -I vla/so101_units.py` checks it and prints
`SO101_UNITS_OK`: the unit map is bit-identical before and after widening (max
diff **0.000e+00** over 20 k configurations), the nominal asset clips exactly
`['shoulder_lift', 'wrist_flex']` (asserted, so the widening silently not being
applied is caught), the widened asset clips nothing, and the checkpoint's median
state and median action both survive `clip_rad` untouched. **In the rollouts the
commanded angle is now clipped on 0 of 450 steps on every joint**, against
200–253/450 on `shoulder_lift` before.

**2. The scene.** `tools/make_bluecube_scene.py` writes a standalone scene
(`scene_data/bluecube/bc_*.json`; `reconstructed_og_scene_so101*.json` is not
touched) with a blue cube, an open-topped gray box built from five fixed slabs,
a navy desk mat and a cream desk, all as OmniGibson `PrimitiveObject`s. Nothing
about the layout is guessed:

* the training set (`pdhinaka/so101_blue_cube_in_gray_box_pi05_cleaned`, 50
  episodes, 18 736 frames) was downloaded and its recorded joint angles run
  through FK, which puts the cube at robot-frame **(0.231, −0.086)** and the box
  at **(0.237, +0.116)**, ~0.20 m apart, both ~0.24 m out, repeatable to ~5 mm
  across episodes;
* the start pose is the real **home pose** `(−1.0, −104.2, 97.4, 76.7, −3.0,
  3.0)` that every episode begins and ends in — which only exists at all under
  the widened limits;
* the prompt is the dataset's own, **"Pick up the blue cube and place it in the
  gray box"**;
* `top_cam` is placed to reproduce the training rig (a third-person view from
  the far side of the mat, 30 cm up, 17 cm beyond the objects, 59° down, f=17 mm),
  which puts the cube and the box at the same image positions as the real frames;
* the arm faces world **−Y**, not +X — measured by fitting the URDF→world
  transform against 450 logged frames to 7e−6 m (`tools/so101_kin.py`). Getting
  this wrong is invisible with an IK controller and fatal for a policy whose
  `shoulder_pan` only spans [−37°, +31°].

### Did it work? Half. The policy picks the cube up and carries it to the box.

Success is measured by `tools/score_bluecube.py` from the pose log — cube
displacement, a geometric containment test for "in the box", the distance from
the cube to the **grasp centre** (the midpoint of the robot's two assisted-grasp
pads, reconstructed from the logged link poses), and how many frames the jaw was
commanded shut while the cube was between the pads. Not from a video still.

| run | what changed | cube moved | max lift | closest grasp centre | frames closed on cube | **in box** |
|---|---|---|---|---|---|---|
| `v1_small` | 2.8 cm cube | 0.0 mm | 0.0 mm | 69.6 mm | 0 | no |
| `v2_match` | 3.8 cm cube, measured colours | 15.7 mm | 0.0 mm | 28.7 mm | 15 | no |
| `w1_match2` | + mat/floor Z-fighting fixed | 28.2 mm | 0.5 mm | 26.5 mm | 12 | no |
| `w2_white` | + arm repainted white | 8.8 mm | 0.5 mm | 30.2 mm | 20 | no |
| `x1_matchcam` | + wrist camera re-aimed, 90° | 56.1 mm | 5.5 mm | 20.6 mm | 33 | no |
| `z1_offset` | + `--action-offset dataset` | 75.9 mm | 1.3 mm | 46.0 mm | 0 | no |
| **`d1_shift`** | **+ objects moved 3.7 cm to where the arm actually reaches** | **194.7 mm** | **152.6 mm** | **20.4 mm** | **211** | held over it, never released |

The policy is **not** doing nothing. In `x1_matchcam` it lifts out of the home
pose, swings `shoulder_pan` to the cube's side, descends, opens the jaw to ~29,
closes it, then sweeps `shoulder_pan` across to the box's side and lifts —
i.e. it executes a recognisable pick-transport-place, and
`results/so101/vla/bluecube/x1_matchcam/vs_training.png` shows its joint
trajectories sitting inside the envelope of all 50 training episodes for the
first ~8 s. It just never has the cube in the jaws: at the moment it closes, the
grasp centre is ~20 mm from the cube centre and ~2 cm above it.

The residual error is concentrated in two joints. At its grasp moment the sim
arm's joint vector differs from the training grasp pose by **elbow_flex −15**
and **wrist_flex +9**, everything else within 2 units — an over-extended
forearm, which puts the tip ~3.7 cm too far forward and ~2.5 cm too high. The
sign and rough size match the dataset's own steady-state follower error
(`mean(action − state)` is −0.70 on `elbow_flex` and +0.66 on `wrist_flex`)
re-applied once per re-plan, 45 times in 450 steps — see `docs/STATUS.md` for
the measurement and for `--action-offset` / `--follower-alpha`, which model it.

### The two sim-side blockers, and what fixed one of them

**The object positions were ~3.7 cm off, and the error is the one assumption
that was always flagged as unverifiable.** Turning the dataset's joint angles
into Cartesian object positions requires the sim joint's zero/sign convention to
match the arm that recorded the data; the checkpoint ships no calibration, so
that cannot be checked. It is measurably slightly wrong: at the dataset-derived
layout the arm's reach bottoms out ~3.7 cm beyond the cube, every time. Moving
**both** objects and the `top_cam` 3.7 cm along the arm's forward axis, so they
sit where the policy's own trained poses actually land in sim
(`scene_data/bluecube/bc_shift.json`, run `d1_shift`), changes the result
completely:

| | dataset-derived layout (`x1_matchcam`) | +3.7 cm (`d1_shift`) |
|---|---|---|
| cube net displacement | 56 mm | **195 mm** |
| cube path length | — | **1232 mm** |
| max lift off the table | 5.5 mm | **152.6 mm** |
| frames with the jaw shut on the cube | 33 | **211** |
| commanded actions inside the checkpoint's own [q01,q99] | 43–100 % per joint | **100 % on all six joints** |
| commands clipped by a joint limit | 0/450 | 0/450 |

The policy **grasps the blue cube, lifts it 15 cm, carries it over the gray box
and holds it there** — final cube position (0.1006, −0.2627, **0.1271**) against
a box centre of (0.1378, −0.2582): 37 mm off centre over an 80 mm interior,
12.7 cm up. That is a real closed-loop manipulation episode driven by a real VLA
off live sim images, and it is the first time anything in this repo has been
picked up by a policy rather than a scripted waypoint controller.

**What it cannot do is let go — and that is OmniGibson's rule, not the
policy's.** `ManipulationRobot._handle_assisted_grasping` computes, for a
position-controlled finger, `applying_grasp = control < joint_upper_limit`, so
the magic constraint is dropped only when the jaw is commanded to its **full
mechanical limit**. pi0.5's gripper action never exceeds **37.3/100** (the real
SO-101's gripper calibration covered about a third of the jaw's travel, and its
own *release* commands in the data are ~23/100), so `applying_grasp` is
permanently true, `_ag_freeze_gripper` stays set, and the jaw freezes.
`d1_shift/vs_training.png` shows it: from t = 3 s the policy's gripper target
oscillates between 5 and 30 for ten seconds while the achieved jaw position sits
pinned flat at 24.3. `sticky` mode uses the same test.
`tools/make_bluecube_scene.py --grasping-mode physical` removes the magic
constraint — but then the jaw cannot hold the cube at all. Two 700-step runs
with finger friction raised to 2.5: `d4_phys` had the jaw shut on the cube for
2 frames and lifted it 5.3 mm; `d5_phys2` had it shut for **181 frames** — six
seconds of contact in the right place — and lifted the cube **3.1 mm**. The
SO-101's gripper is a single lever closing against a static face under position
control; without the constraint the cube slips out every time.

So it is a pincer, and neither side of it is the policy's fault:

| grasping mode | picks the cube up? | puts it down? |
|---|---|---|
| `assisted` / `sticky` | **yes** — 1 of 2 attempts, 15 cm lift, carried over the box | **no** — release needs a full-limit jaw command the policy never issues |
| `physical` | **no** — 181 frames of jaw-on-cube contact bought 3.1 mm of lift | n/a |

The pick is also not deterministic: pi0.5 samples its trajectory, and
`d2_shift_long` (identical scene, 800 steps) closed on the cube 113 times but
never lifted it. Closing the loop properly needs either a release rule keyed to
the *relative* jaw command instead of the absolute limit, or a compliant
two-finger gripper.

The appearance gap that is left is visible in
`results/so101/vla/bluecube/x1_matchcam/real_vs_sim_both_cams.png`: `top_cam`
matches the training frames closely, but `wrist_cam` does not — the real rig's
wrist camera sits much closer to the mat, looks almost straight down between
**two** white jaws, and fills the frame with the mat's printed lettering, while
the sim's (authored from NVIDIA's Isaac Lab config) sees one yellow jaw and a
lot of empty mat. That camera is the one that matters during the final 2 cm.

### Running it

```bash
cd /scratch/pandya.kei/simfoundry-repro
E=conda_envs/simfoundry-editor/bin/python

$E -I tools/widen_so101_limits.py                 # once: the widened asset
conda_envs/simfoundry/bin/python -I vla/so101_units.py   # verify the fix
$E -I tools/make_bluecube_scene.py --out scene_data/bluecube/bc_match2.json \
    --cube-size 0.038 --box-inner 0.080 --box-wall 0.010 --box-height 0.035

bash jobs/submit_bluecube.sh myrun $PWD/scene_data/bluecube/bc_match2.json
# variants: SO101_USD=... (white arm / re-aimed wrist cam),
#           ACTION_OFFSET=dataset, FOLLOWER_ALPHA=dataset

bash jobs/finish_bluecube.sh myrun match2 render   # score + sheets + 6-angle render
conda_envs/simfoundry/bin/python -I tools/plot_bluecube_vs_training.py --run <dir>
```

Outputs land in `results/so101/vla/bluecube/<run>/`: `score.json`,
`live_filmstrip.png`, `real_vs_sim.png`, `real_vs_sim_both_cams.png`,
`vs_training.png`, `videos/` (the live `top_cam`/`wrist_cam` pair the policy
actually saw) and `multiview/` (six offline angles + a 2x3 grid +
`contact_sheet.png`; ~1 h of CPU on `short`, so it lands after the rollout).
`results/so101/vla/bluecube/dataset_ref/` holds the real training frames
everything was matched against.

The full scoreboard (`results/so101/vla/bluecube/*/score.json`):

| run | cube moved | max lift | closest grasp centre | frames jaw shut on cube |
|---|---|---|---|---|
| `v1_small` | 0.0 mm | 0.0 mm | 69.6 mm | 0 |
| `v2_match` | 15.7 mm | 0.0 mm | 28.7 mm | 15 |
| `w1_match2` | 28.2 mm | 0.5 mm | 26.5 mm | 12 |
| `w2_white` | 8.8 mm | 0.5 mm | 30.2 mm | 20 |
| `x1_matchcam` | 56.1 mm | 5.5 mm | 20.6 mm | 33 |
| `z1_offset` | 75.9 mm | 1.3 mm | 46.0 mm | 0 |
| `z2_cam_offset` | 18.4 mm | 0.0 mm | 46.3 mm | 0 |
| **`d1_shift`** | **194.7 mm** | **152.6 mm** | **20.4 mm** | **211** |
| `d2_shift_long` | 55.9 mm | 1.9 mm | 24.6 mm | 113 |
| `d4_phys` | 40.3 mm | 5.3 mm | 22.2 mm | 2 |
| `d5_phys2` | 51.4 mm | 3.1 mm | 23.2 mm | 181 |

None ends with the cube in the box. `d1_shift` ends with it **in the gripper,
12.7 cm above the box**.

## Everything is on /scratch

`/scratch` is not backed up and is subject to purge. `pipeline/keepalive_touch.sh`
runs weekly from `scrontab` (Sundays 03:15) to keep it from being reaped.
Nothing here is in git — if it matters, copy it off.

# SimFoundry reproduction — handover

> **Paths changed 2026-10-06.** Everything now lives under
> `/scratch/pandya.kei/simfoundry-repro/`. Start at `../README.md`.
> Older commands in this document that reference `/scratch/pandya.kei/<script>.py`,
> `/scratch/pandya.kei/renders/`, `/scratch/pandya.kei/policies/` or
> `/scratch/pandya.kei/robots/` map to `tools/`, `results/`, `policies/` and
> `robot/` inside the project.


Last updated: 2026-10-04 ~10:45 EDT

---

## ✅ PIPELINE A IS COMPLETE — reproduced end to end

Finished overnight 2026-10-04. Both jobs exited clean:

```
10791566  COMPLETED  10m23s  stages 11+12  (d0004, CPU, short)
10794519  COMPLETED  15m33s  stages 13+14  (d4049, L40S, sharing)
```

### Deliverables — `Data/fruits_demo/s14_og/`

| File | |
|---|---|
| `reconstructed_og_scene.json` | 19 KB — valid OmniGibson scene, **7 entities** |
| `reconstructed_scene.png` | 184 KB — Isaac Sim render of the settled scene |
| `settled_poses.json` | per-object position + quaternion after PyBullet settling |
| `stage_info.json` | stage metadata |

Scene contents: `robot0` (FrankaPanda) + 6 `DatasetObject`s —
`iter_0` yellow_pear, `iter_20` orange_square_plate, `iter_28` red_apple,
`iter_29` yellow_banana, `iter_30` orange_fruit, `iter_31` teal_square_plate.
Versions: omnigibson 3.8.0, bddl 3.7.0.

### Comparison figure

`/scratch/pandya.kei/renders/comparison_real_vs_sim.png` — real video frame 12
beside the Isaac Sim reconstruction. Built by `make_comparison.py`.
Note: crop the Isaac panel on **colour saturation**, not median background —
the render has a gradient backdrop that defeats background differencing.

**Assessment.** Object identity, per-object geometry, colour and relative layout
all transferred correctly. Visible errors: the plates are under-scaled relative
to the fruit (monocular depth gives the weakest absolute metric scale, and flat
wide objects suffer most), the orange plate desaturated to amber during
Hunyuan3D's PBR texture pass, and inter-object spacing is looser than the photo.

### Full stage timings (final run)

```
1b   31.8s | 2   42.8s | 3   88.8s | 4  19.3s
5  2027.0s  (SAM3 + FLUX decomposition)
6  4717.8s  (FLUX upsampling)
7   790.8s  (Hunyuan3D, resolution 1536, 80GB A100)
8   518.0s  (FoundationPose)
10   43.5s | 11+12 10m23s | 13+14 15m33s
```

---

## What is still NOT reproducible (unchanged)

| Target | Status |
|---|---|
| Pipeline B — digital cousins | **Not on free tier.** `B_augmentation` has no `flux` branch; `prompt_cousin_structured.image_model` goes straight into `Gemini(...)` and all Gemini image models are quota-blocked |
| Sim-to-real robotics results (DROID/YAM table) | **Not reproducible at any budget** — README states data-gen + policy training code is unreleased |
| YAM scenes | **Impossible** — the YAM robot asset was never published |

---

## Re-running from scratch

The pipeline is resumable and all patches are in place. To redo the whole thing:

```bash
# stages 1b-10 need a GPU (gpu partition, a100)
sbatch /scratch/pandya.kei/run_pipelineA.sbatch
# stages 11+12 are CPU-only -> short partition, no GPU queue
sbatch --partition=short --gres=none --export=ALL,RESUME=1,NO_GPU=1,STAGES=11+12 \
       /scratch/pandya.kei/run_pipelineA.sbatch
# stages 13+14 need RT cores -> sharing (1h cap), zero Gemini calls
sbatch --partition=sharing --gres=gpu:l40s:1 --time=01:00:00 \
       --export=ALL,RESUME=1,STAGES=13+14 /scratch/pandya.kei/run_pipelineA.sbatch
```

**Budget the Gemini calls before starting.** A clean full run needs far more
than 20 requests across stages 3, 5, 8 and 11, so it cannot complete in one day
on the free tier. Split across days with `object_indices`, or reuse the existing
`Data/fruits_demo/` outputs with `RESUME=1`.

---

## Goal and what is actually achievable

Reproducing NVlabs/SimFoundry (arXiv 2606.28276) on Northeastern Explorer.

| Target | Status |
|---|---|
| **Pipeline A — reconstruction** | ACHIEVABLE; 11/12 stages done |
| Pipeline B — digital cousins | **NOT achievable on free tier.** `B_augmentation` has no `flux` branch; `prompt_cousin_structured.image_model` goes straight into `Gemini(...)` and every Gemini image model is quota-blocked |
| Sim-to-real robotics results (DROID/YAM table) | **NOT reproducible at any budget** — README states data-gen + policy training code is not released |
| YAM scenes | **NOT possible** — the YAM robot asset was never published |

---

## Current artifacts — `Data/fruits_demo/` (1.2 GB)

| Dir | Size | State |
|---|---|---|
| s1_video | 451M | done |
| s2_da | 11M | done — DA3 depth |
| s3_ground | 3.8M | done — canonical frame **12** |
| s4_frame | 12M | done |
| s5_scene | 176M | done — SAM3 + FLUX decomposition |
| s6_upsample | 93M | done — FLUX upsampling |
| **s7_mesh** | **370M** | done — **13 `.glb`**, resolution 1536 |
| **s8_pose** | **23M** | done — FoundationPose |
| s10_compile | 512B | done |
| s11_sim | — | **done — all 6 objects** |
| s12_physics | — | done — PyBullet settled |
| **s14_og** | — | **done — `reconstructed_og_scene.json` + render** |

### The scene is 6 objects, NOT 32
`s5_scene/masked_object/iter_N.png` is per-**iteration**, not per-object.
`obj_cat_list/iter_0.json` lists the truth:
**teal plate, orange plate, orange, red apple, yellow banana, yellow pear.**

Local FLUX repeatedly failed to remove the elongated banana — re-detected 24
times with `valid_pixel_proportion` decaying to 0.0; **16 of 32 iterations are
flagged `is_valid_removed_obj=false`**. `s6_upsample` has `check_valid=false` +
`include_invalid=true`, so all 32 were passed downstream regardless.

Exactly six iterations have `valid_pixel_proportion == 1.0`, one per object.
Hence the pinned override (do not drop this):

```
s7_mesh/s8_pose/s11_sim.object_indices=[0,20,28,29,30,31]
  0  yellow pear          29  yellow banana
 20  orange square plate  30  orange fruit
 28  red apple            31  teal square plate
```

---

## THE constraint: Gemini free tier = 20 requests/day

```
quotaId:    GenerateRequestsPerDayPerProjectPerModel-FreeTier
quotaValue: 20          model: gemini-3-flash
```

User has chosen to **stay on the free tier**. Implications:

* Only `gemini-3-flash-preview` works at all. Every pro model 404s
  ("no longer available to new users"); every image model is 429.
* Stage 11 costs **6** calls (one per object, unconditional — it does NOT skip
  objects already on disk).
* Stages 12, 13, 14 cost **0**.
* **Retry amplification is the real danger.** An earlier run used a 25-attempt
  budget and a single 503 spike ate the whole day. Now pinned:
  `SIMFOUNDRY_REMOTE_RATE_LIMIT_ATTEMPTS=3` → worst case 6×3 = 18 < 20.
* `retryDelay: 1763s` in the 429 payload is **misleading** — it is a hard daily
  cap, not a 29-minute window. Retrying inside the same day cannot work.
* **Do not make probe/test calls to the API.** Each one spends from the 20.

If stage 11 still fails: split it across two days via
`s11_sim.object_indices=[0,20,28]` then `[29,30,31]`.

---

## Scripts (all in /scratch/pandya.kei)

| File | Purpose |
|---|---|
| `install_simfoundry.sbatch` | builds the 3 conda envs. **Already succeeded** — do not re-run |
| `run_pipelineA.sbatch` | the pipeline driver. Env knobs: `RESUME=1` (keep outputs), `STAGES=7+8+10` (**`+`-separated — sbatch `--export` splits on commas**), `NO_GPU=1` (CPU-only stages), `OBJ_IDX`, `SIMFOUNDRY_REMOTE_RATE_LIMIT_ATTEMPTS` |
| `validate_envs.sh` | post-install import check across all 3 envs |
| `validate_faiss.sbatch` | finishes the faiss GPU probe deferred by the CPU install |
| `keepalive_touch.sh` | weekly scratch anti-purge (installed in `scrontab`, Sundays 03:15) |
| `raster2.py` | first-cut pure-numpy renderer — superseded by `sfrender.py`, kept only for old call sites |
| `sfrender.py` | **the renderer.** Pure numpy (no OpenGL/EGL anywhere on these nodes). Real pinhole camera (fx,fy,cx,cy, OpenCV axes), z-buffer, textured faces, tessellated floor, per-pixel contact shadow + grid + vignette. Styling knobs: `grid_on`, `shadow_strength`, `shadow_radius`, `vignette` |
| `render_views.py` | driver: orbit + top-down + close-up contact sheet. `<extract_dir> <out_dir> [n_views]` |
| `render_aligned.py` | driver: renders the sim from the **real camera** (stage-4 `cam2world` + DA3 intrinsics) → real/sim/50-50-overlay strip |
| `make_views_sheet.py` | assembles the multi-view contact sheet |
| `reposition_robot.py` | moves the Franka base so objects land inside the 0.855 m reach; writes `*_closer.json` scene + settled poses |
| `robots/deinstance_so101.py` | makes NVIDIA's SO-101 USD loadable by OmniGibson (de-instance, author scale, per-mesh CollisionAPI, hull colliders) |
| `swap_robot_so101.py` | rewrites a reconstructed scene to use the SO-101 (class+module, 6-DOF state, resized controller filters, base moved for 0.32 m reach) |
| `robots/so101_og.py` | the OmniGibson `SO101` robot class (installed to `omnigibson/robots/so101.py`) |
| `policies/probe_so101.py` | loads the arm alone and measures jaw opening / AG point placement |
| `policies/pick_and_place.py` | scripted pick-and-place. `--scene-json/--settled-json` overrides, `--pick cat1,cat2`, auto-selects by measured AABB vs real gripper travel, places to a free spot (no plate needed) |
| `.install_jobid`, `.pipeline_jobid`, `.og_jobid` | last job ids |

---

## Rendering the reconstructed scenes (no OpenGL on these nodes)

Everything is rasterised in numpy. `sfrender.Scene` loads a `light_editor/extract.py`
manifest, bakes per-face texture colours and shading, and `.render(R_wc, t, K, w, h)`
projects with a **real pinhole camera in OpenCV axes** (+x right, +y DOWN, +z forward).

Three traps, all hit and fixed:

1. **A single large floor quad renders as nothing.** Its far corners fall behind
   the camera, the near-plane cull rejects the whole triangle, and you get objects
   floating in a void. The floor must be tessellated (60x60 here).
2. **Baking shadows/grid per floor triangle looks broken.** An 11 cm floor cell
   against a 6 cm shadow falloff gives blocky triangular shadows, and grid lines
   come out as zigzags. Both are evaluated **per pixel** instead — for a plane
   that is exact: intersect the pixel ray with `z=floor_z` and evaluate there.
3. **Ambient of 0.34 reads as black** on anything facing away from the lights.
   A white cup rendered dark grey and looked untextured when the texture was
   sampling fine. Hemisphere term + ambient 0.46 fixed it.

```bash
E=/scratch/pandya.kei/conda_envs/simfoundry-editor/bin/python
cd /scratch/pandya.kei
# multi-angle contact sheet (~4 min)
$E render_views.py renders/reference_demo_extract renders/reference_demo_views 4
$E make_views_sheet.py renders/reference_demo_views renders/reference_demo_multiview.png
# real-vs-sim from the ACTUAL source camera (~25 s)
$E render_aligned.py --extract renders/reference_demo_extract \
    --scene-data repos/SimFoundry/Data/reference_demo --out-dir renders/reference_demo_views
```

`render_aligned.py` is the one that is actually *evidence*: it reads
`s3_ground/frame_selection.json` for the chosen frame, `s4_frame/image_<i>_cam2world.npy`
for its pose in the reconstructed world frame, and the per-frame intrinsics from
`s2_da/da/exports/npz/results.npz`, so sim and real are in pixel correspondence.
The 50/50 overlay shows the cup, boat and block landing close — and the white
whale toy present in the real frame but **absent from sim**, which is exactly
the object the boundary filter pruned.

---

## Pick-and-place in a reconstructed scene

Stage 14 drops the robot at a fixed default pose. In `reference_demo` that is
x = -1.0 while the objects sit near the origin, so **everything was past the
0.855 m reach**. `reposition_robot.py` rewrites
`state.registry.object_registry.robot0.root_link.pos` (the only place the base
pose lives — `init_info` carries no position) and invalidates the saved `arm_0`
IK goal, which was captured against the old base and would yank the arm on step 1.

```bash
$E reposition_robot.py --scene-dir repos/SimFoundry/Data/reference_demo
D=/scratch/pandya.kei/repos/SimFoundry/Data/reference_demo/s14_og
sbatch --export=ALL,SCENE=reference_demo,\
SCENE_JSON=$D/reconstructed_og_scene_closer.json,\
SETTLED_JSON=$D/settled_poses_closer.json,\
PICK=orange_block_toy+red_boat_toy+white_paper_cup,\
VIDEO_NAME=pick_place_reference run_pick_place.sbatch
```

**`PICK` uses `+` separators** — `sbatch --export` splits its own value on
commas; the sbatch translates `+` back to `,`.

**Gripper fit is a hard limit.** The default Franka hand is
`panda_finger_joint1/2`, 0–0.04 m each, so the jaw opens to ~8 cm. The policy
now reads that off the articulation and measures each object's AABB:
`orange_block_toy` 2.5 cm and `red_boat_toy` 4.1 cm fit; the reconstructed
`white_paper_cup` is **13.2 cm across and cannot be grasped between the fingers**
at any approach. `--pick` overrides the filter so an impossible target can still
be attempted on purpose.

**The grasp control point is the AG midpoint, not the finger links.**
`franka.py` puts the Panda hand's assisted-grasp points at `[0, 0.001, 0.045]`
in each finger link's frame — **4.5 cm beyond the link origins** that
`finger_tip_pos()` returns. Driving the link origins to an object at z=0.018
put the real pads at z=-0.027, under the ground plane: descend never converged
(4.1 cm short after all 90 steps) and the arm swept every object ~5.8 cm aside.
`ag_point_pos()` transforms those points to world and controls their midpoint;
with that, both graspable objects lift first try. A stall guard now aborts a
waypoint after 12 steps without progress, so a jam can no longer shove the
scene around.

Result on `reference_demo` (job 10839140, l40s, ~7 min):

| object | width | grasped | displaced | outcome |
|---|---|---|---|---|
| `orange_block_toy` | 2.5 cm | `iter_0` | 0.233 m | **SUCCESS** |
| `red_boat_toy` | 4.1 cm | `iter_2` | 0.332 m | **SUCCESS** |
| `white_paper_cup` | 12.8 cm | — | 0.001 m | FAILED — stalled, jaw is 8 cm |

Video: `Data/reference_demo/pick_place/pick_place_reference.mp4`
(copy at `renders/pick_place_reference.mp4`, 269 frames @15 fps, wrist cam |
workspace cam side by side).

Two bugs worth not re-introducing in the sbatch/policy:
* `[ -n "$X" ] && ARGS+=(...)` as a standalone command under `set -e` **kills
  the job** when the test is false. Use `if/then`.
* The video is only written after the task loop, so an exception mid-task used
  to lose every frame already captured. Each task is now in its own try/except.

## SO-101 arm in OmniGibson (for sim-to-real)

OmniGibson ships no SO-100/SO-101. The asset is NVIDIA's, from
`isaac-sim/Sim-to-Real-SO-101-Workshop` — which is an **Isaac Lab 2.3.2 / NGC
Docker** project, not an OmniGibson one, and Docker is unavailable on Explorer
(Isaac Lab is not on PyPI either). So only the asset and its tuned gains were
taken; the arm was wrapped as an OmniGibson robot class instead.

```bash
E=/scratch/pandya.kei/conda_envs/simfoundry-editor/bin/python
# the USDs are Git LFS pointers and git-lfs is not installed here:
curl -sSL -o SO-ARM101-USD-NO-CAMERA.usd \
  https://media.githubusercontent.com/media/isaac-sim/Sim-to-Real-SO-101-Workshop/main/source/sim_to_real_so101/assets/usd/SO-ARM101-USD-NO-CAMERA.usd
$E robots/deinstance_so101.py robots/so101_usd/SO-ARM101-USD-NO-CAMERA.usd \
                              robots/so101_usd/so101.usd
$E swap_robot_so101.py --scene-dir repos/SimFoundry/Data/reference_demo
```

### Four asset incompatibilities, all fixed in the asset (never by patching OmniGibson)

1. **Instanceable references.** Each link's geometry is an instanceable
   reference to an internal prototype. USD instance proxies are read-only, and
   OmniGibson authors on finger collision meshes and enumerates them for
   `aabb_extent`/`aabb_center`. Cleared the flag and flattened.
2. **No authored `xformOp:scale`.** `xform_prim.py:78` does
   `th.tensor(get_attribute("xformOp:scale"))` with no `None` guard →
   `RuntimeError: Could not infer dtype of NoneType`.

   **Author the FULL `[translate, orient, scale]` triple, never scale alone.**
   Authoring only `xformOp:scale` on the 132 prims that had no ops left a
   half-specified transform stack. OmniGibson rewrites xform properties on load,
   and those visuals then got displaced from their links: the moving jaw rendered
   as a lever floating ~10 cm from the arm and the gripper's static jaw was
   effectively lost. Everything else checked clean while this was broken --
   identical mesh counts, 0.000 mm vertex displacement vs NVIDIA's original,
   identical materials/winding/visibility/purpose, all 17 visual meshes
   registered by OmniGibson, and 0.00 mm joint drift over a full episode -- so
   the only signal was the render itself. Matching the asset's own convention
   (every prim it already transforms uses that triple) fixes it.
3. **`CollisionAPI` on the group, not the meshes.** NVIDIA applies it to each
   link's `collisions` Xform and lets PhysX propagate; OmniGibson tests
   `prim.HasAPI(UsdPhysics.CollisionAPI)` **per mesh** (`rigid_prim.py:220`), so
   all 92 meshes classified as visual, zero collision meshes were found, and
   `entity_prim.aabb` died with `torch.cat(): expected a non-empty list`. Pushed
   the API down onto the meshes.
4. **Raw 3D-print STLs as colliders — 1,196,652 verts over 17 meshes.** Isaac
   Lab cooks once and caches; OmniGibson additionally runs
   `get_mesh_volume_and_com` and `check_extent_radius_ratio` per mesh in Python,
   and job 10845375 **timed out still inside `og.Environment()`** after the full
   hour. Replaced collision geometry with convex hulls (→ 34,881 verts, 2.9%);
   safe because the robot runs `self_collisions=False` and assisted grasping
   uses an AG-point ray test, not precise contact. Visual meshes stay full-res.

### Robot class — `omnigibson/robots/so101.py` (source of truth: `robots/so101_og.py`)

* **One moving jaw.** `_assisted_grasp_start/end_points` are set explicitly, as
  `franka.py` does, because the default derivation walks finger links expecting
  exactly two.
* **`finger_link_names = ["gripper", "jaw"]`.** In `assisted` mode
  ManipulationRobot requires `len({finger_links} & contacts) >= 2`. With `jaw`
  alone that is **unsatisfiable** — the arm reached the object, closed on it, and
  `in_hand` stayed `None` every time. Listing `gripper` too is physically honest:
  the static jaw face is part of that body. `finger_joint_names` stays one entry.
* **AG points are MEASURED, not guessed**: the closed-jaw contact patch between
  the collision meshes (mean gap 2.3 mm over 400 verts). The pads sit 6.5–8.5 cm
  out along the lever; a first guess near the hinge reported a 3 cm "opening"
  because the parts are ~2 mm apart there at *every* jaw angle.

### Geometry facts that drive the policy settings

| quantity | value |
|---|---|
| arm DOF | 5 + 1 jaw (`Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw`) |
| reach | ~0.32 m (vs Franka 0.855) |
| approach axis | gripper-local **−Z** (pads at z = −0.101) → `--down-axis=-z`, `DOWN_R = identity` |
| usable opening | ~5–6 cm at the pads (NOT the ~3.5 cm spec figure) |
| best jaw angle | **+30°** — closing direction is `[0.999, -0.012, 0.031]`, i.e. clean perpendicular to approach. At +100° it tilts to 54° off. → `--grip-open 0.364` |

**5 DOF means a 6-DOF pose target is rank-deficient** and IK can only
least-squares the orientation. Top-down picks are fine (with the approach axis
vertical, `Wrist_Roll` *is* jaw yaw). Joint space is the better sim2real match
anyway — it is what LeRobot's real SO-101 interface consumes.

### Working invocation (job 10869439 — both objects, 2/2)

```bash
D=/scratch/pandya.kei/repos/SimFoundry/Data/reference_demo/s14_og
sbatch --export=ALL,SCENE=reference_demo,\
SCENE_JSON=$D/reconstructed_og_scene_so101.json,\
SETTLED_JSON=$D/settled_poses_so101.json,\
PICK=orange_block_toy+red_boat_toy,\
REACH=0.32,MAX_GRASP_W=0.05,DOWN_AXIS=-z,ROBOT_H=0.35,\
GRIP_OPEN=0.364,GRIP_RELEASE=1.0,APPROACH_H=0.06,LIFT_H=0.09,POS_TOL=0.007,\
PLACE_AZ=55,CAM_LINK=base,CAM_SCALE=0.9,\
VIDEO_NAME=pick_place_so101 run_pick_place.sbatch
```

| object | width | grasped | displaced | outcome |
|---|---|---|---|---|
| `orange_block_toy` | 2.5 cm | `iter_0` | 0.079 m | **SUCCESS** |
| `red_boat_toy` | 4.1 cm | `iter_2` | 0.127 m | **SUCCESS** |

Video `renders/pick_place_so101.mp4`, stills `renders/so101_filmstrip.png`.
The cup (12.8 cm) is both too wide for the jaw and outside the 0.32 m reach.

### Settings that are load-bearing, with the symptom if wrong

* `GRIP_RELEASE=1.0` — the partial 0.364 open **holds** the assisted grasp, so
  the block was still in hand during the next task.
* `POS_TOL=0.007` — the 0.02 default is the boat's *entire 2 cm height*; descend
  stopped 2 cm above it and the grasp never latched.
* `APPROACH_H/LIFT_H = 0.06/0.09` — the Franka's 0.12/0.22 are unreachable for a
  0.32 m arm and the lift waypoint just stalls.
* `PLACE_AZ=55` — the default +y place offset put the target at 0.42 m, outside
  the reach, and its ±y fallback was out of reach too. Rotating the pick point
  about the base at constant radius is reachable by construction.
* `CAM_LINK=base` — the camera prim path was hardcoded to
  `/controllable__frankapanda__robot0/panda_link0/...`, so under the SO-101 the
  parent prim did not exist and the camera was never placed relative to the base.
* `--down-axis=-z` must be passed as **`--opt=value`**; `--down-axis -z` makes
  argparse read `-z` as another option.
* The robot must be dropped by the base link's **sit offset**. `base`'s geometry
  starts at z=+0.0301 m relative to its own origin, and that origin is the
  articulation root the scene positions, so a root at z=0 leaves the arm
  hovering 3 cm above the table. `swap_robot_so101.py` measures this.
* Do **not** try to cancel the asset's baked +90° root yaw. OmniGibson's
  `root_link.ori` replaces `/so101_new_calib`'s authored xform rather than
  composing with it, so identity already gives forward=+X. "Compensating"
  rotated the arm off its workspace and stalled every waypoint ~0.3 m short.
* Orientation is commanded as the **approach axis only, yaw free**
  (`approach_delta`). A full target rotation also pins yaw, which 5 DOF cannot
  honour alongside a position.

### Scene swap gotchas (`swap_robot_so101.py`)

* `init_info` records **both** `class_module` and `class_name`; changing only the
  name gives `module 'omnigibson.robots.franka' has no attribute 'SO101'`.
* The saved controller **smoothing filters are Franka-sized**: arm
  `control_filter.past_samples` is (5, 7) and the gripper's `vel_filter` is
  (5, 2). Restoring those under the SO-101 raised `could not broadcast input
  array from shape (5,) into shape (7,)` every step, so **the arm silently never
  moved** and every waypoint stalled. They are reset to (5, 5) and (5, 1).

### Debugging note

`og.shutdown()` calls `app.close()`, which **terminates the process**
(`omnigibson/__init__.py:144`). A bare `try/finally` around the env therefore
swallows the exception and the job exits **0 with no output** — job 10843029
reported success while doing nothing. Always print the traceback before
shutting down.

---

## Patches applied to the repo (all keep a `.orig` backup)

These are load-bearing. If the repo is ever re-cloned, re-apply them.

1. **`deps/BEHAVIOR-1K/setup.sh`** — export `LD_LIBRARY_PATH=$CONDA_PREFIX/lib`
   at the top. **This was the root cause of three straight install failures.**
   Isaac's `libcarb.so` needs `GLIBCXX_3.4.30`; Rocky 9.7 has only 3.4.29 while
   the conda env has 3.4.36. Without it, `python -c "import isaacsim"` fails and
   `set -e` kills setup.sh silently.
2. **`deps/BEHAVIOR-1K/setup.sh`** — manylinux retag: `check_glibc_old` now
   matches 2.34, and the rename targets `manylinux_2_35` (upstream's version was
   a no-op — it substituted `_2_34`, which never appears in the filename).
   Safe because `objdump` over all 109 `.so` in omniverse_kit shows max required
   symbol = `GLIBC_2.34`, i.e. the wheel tag is mislabeled.
3. **`scripts/installation/install_simfoundry.sh`** — tee setup.sh stdout to
   `behavior1k_setup.log` instead of `/dev/null` (three failures were
   undiagnosable because of that redirect); plus `LD_LIBRARY_PATH` and a
   `lerobot` install after the setup.sh call. (OmniGibson 3.8.0 declares lerobot
   only in its `eval` extra but imports it unconditionally.)
4. **`scripts/installation/install_da3.sh`** — `torch-cluster` via the PyG wheel
   index (sdist-only, imports torch at build time); strip `sm_90` to skip the
   FlashAttention-3 Hopper build (3–7 h → ~30 min).
5. **`scripts/installation/faiss_gpu.sh`** — `SIMFOUNDRY_SKIP_FAISS_GPU_VALIDATE=1`
   skips only the live-device probe, so the long install can run on a CPU node.
6. **`api_keys.txt` symlinked to repo root.** `vlm.py:106` walks only *parents*
   of `vlm.py`; the real file is in `scripts/installation/`, which is not one.
   Without the symlink the Gemini key never loads and stage 3 dies.
7. **pytorch3d rebuilt from source against the host compiler**
   (`CC=/usr/bin/gcc CUDAHOSTCXX=/usr/bin/g++`). The installer's build used
   conda's gcc-13 whose sysroot ships glibc 2.38/2.39, producing a `_C.so`
   needing `hypotf@GLIBC_2.35` — unloadable on this cluster. After rebuild max
   symbol is 2.32. **This unblocked FoundationPose in stage 8.**
   Do NOT try the `LD_LIBRARY_PATH` trick for this — mismatched libm/libc
   breaks the dynamic linker (it works for libstdc++, not libc).

Also downloaded by hand (not via `download_checkpoints.sh`, which was skipped):
`deps/Hunyuan3D-2.1/ckpt/RealESRGAN_x4plus.pth`,
`deps/FoundationPose/weights/{2023-10-28-18-33-37,2024-01-11-20-02-45}`,
and `facebook/dinov2-giant` into the HF cache.

---

## Runtime env vars the pipeline needs

```bash
export HF_HOME=/scratch/pandya.kei/hf_cache
export LD_LIBRARY_PATH=/scratch/pandya.kei/conda_envs/simfoundry/lib:$LD_LIBRARY_PATH  # GLIBCXX
export OMNI_KIT_ACCEPT_EULA=YES          # else Kit prompts and hangs a batch job
export XFORMERS_IGNORE_FLASH_VERSION_CHECK=1
```

That last one matters: `install_simfoundry.sh` installs flash-attn 2.8.3, but
xformers hard-requires `>=2.7.1,<=2.7.4` and raises on import.
`diffusers.loaders.ip_adapter` pulls xformers in, so FLUX dies — and `vlm.py:1150`
misreports it as "requires a diffusers version that provides FluxKontextPipeline",
which is wrong (diffusers is 0.40.0 and ships the class).

---

## ⚠ Idle-GPU reaper on the `gpu` partition

Two jobs were killed with `CANCELLED by 0` (root) at ~1 h, on different nodes:

| Job | Partition | Elapsed | What it was doing |
|---|---|---|---|
| `10773471` | gpu | 1:02:24 | install — downloading, GPU idle |
| `10801646` | gpu | 1:06:19 | stage 1b ffmpeg (885 MB of frames), GPU idle |

By contrast `10782438` ran **1:59:05** on `gpu` untouched — it was saturating the
card with FLUX. So the trigger is sustained LOW GPU UTILISATION, not wall time.

**Keep GPU jobs GPU-busy.** Stage GPU requirements (grepped for torch/cuda):

```
1b  cpu   ffmpeg frame extraction   <- ~1h of GPU idle if run on a GPU node
2   GPU   DA3 depth (dispatches into the da3 env)
3   GPU   ground plane (+ VLM, network-bound)
4   cpu
5   GPU   SAM3 + FLUX
6   GPU   FLUX upsample
7   GPU   Hunyuan3D
8   GPU   FoundationPose + nvdiffrast
10  cpu   11  cpu   12  cpu (PyBullet DIRECT)
13  GPU   14  GPU   (Isaac Sim -> RT cores -> `sharing` only)
```

Recommended split, one submission each:
1. `short`  + `NO_GPU=1`, `STAGES=1b`
2. `gpu`    + `STAGES=2+3+4+5+6+7+8`
3. `short`  + `NO_GPU=1`, `STAGES=10+11+12`
4. `sharing`+ `gpu:l40s:1`, `STAGES=13+14`

If a job is root-cancelled again, check for an email from Research Computing
before resubmitting the same shape.

## Cluster facts (Explorer)

| Partition | MaxTime | GPUs |
|---|---|---|
| `gpu` | 8 h | a100 (40GB **and** 80GB — scheduler does not say which), h200, t4, v100 |
| `sharing` | **1 h** | l40s, a6000, a5000, l40, rtx6000, h100 ← **only RT-core cards** |
| `short` | 2 d | CPU only; backfills in ~2 min |
| `courses-gpu` | 1 d | p100, v100 — both too old |

* **Submit from a login node** (`explorer-01/02`). Open OnDemand compute sessions
  run inside an Apptainer image with **no slurm client** at all.
* `http_proxy` intercepts even `127.0.0.1` → local servers return a squid 403.
  Use `curl --noproxy '*'`.
* **No OpenGL/EGL anywhere**, despite CUDA working. Hence `raster2.py`.
* `sbatch --export` **splits values on commas** — pass lists `+`-separated.
* `--dependency=singleton` serialises on *previously submitted* same-name jobs;
  it does not mean "whichever starts first wins".
* Stages 11 and 12 are **CPU-only** (no torch/CUDA import) — run them on `short`
  with `NO_GPU=1` and skip the gpu queue entirely.
* Stages 13 and 14 are the **only** ones importing omnigibson, so they are the
  only ones needing RT cores.

---

## Known quality caveats vs. the paper

* **Stage 9 skipped** → every object imports rigid, no articulation.
* **FLUX substituted for Gemini image removal** (free tier has no image model) →
  the banana failure above. The `object_indices` restriction is a mitigation,
  not a fix. A faithful run needs paid Gemini image generation.
* Meshes are full quality (resolution 1536) — that run landed on an 80 GB A100
  and `low_vram` is auto-selected from detected VRAM.

---

## Optional follow-ups

1. Open the scene in the light editor (no GPU, no Isaac needed; needs an SSH
   tunnel since compute nodes have no browser):
   ```bash
   mamba run -n simfoundry-editor python \
     /scratch/pandya.kei/repos/SimFoundry/scripts/interactive/light_editor/server.py \
     --scene /scratch/pandya.kei/repos/SimFoundry/Data/fruits_demo/s14_og/reconstructed_og_scene.json
   ```
2. Re-run stage 7 at full resolution if a run ever lands on a 40GB A100
   (`low_vram` is auto-selected from detected VRAM; the final run got 80GB so
   meshes are already resolution 1536):
   `--include 7 -- s7_mesh.low_vram=false`
3. Reconstruct another example video — `docs/assets/example_videos/` also has
   PutCupInBowl, PutMarkerInCup, ThrowAwayTrash, Mailbox. Use a fresh
   `--scene-name`; budget the Gemini calls across days.

---

## 2026-10-06 — SO-101 wrist camera, and the Isaac render verdict

Full write-up in `../README.md` ("Live Isaac rendering, and the SO-101 wrist
camera" + "The one thing to know before touching the renders"). Summary:

### 1. The wrist camera works, by authoring a Camera prim into the USD

NVIDIA's WITH-camera asset `SO-ARM101-USD.usd` (fetched, 23 MB, kept at
`/scratch/pandya.kei/so101_dl/`) has **no `UsdGeom.Camera` prim at all** — only
the camera *body* geometry under `gripper/visuals/camera_mount`, whose
`/meshes/camera_mount` prototype our `so101.usd` already contains. The sensor in
NVIDIA's project is created in Isaac Lab code (`task_env_cfg.py`, saved as
`robot/so101_usd/nvidia_task_env_cfg.py.ref`). So that asset was NOT re-imported.

`tools/add_wrist_cam.py so101.usd so101_wristcam.usd` authors their camera into
ours: `/so101_new_calib/gripper/gripper_cam`, offset `(-0.005, 0.06, -0.062)`,
`Rx(-45 deg)`, 640x480, f 13.5 mm / aperture 20.955 mm → `K = [412.31, 412.31,
320, 240]`, FOV 75.63 x 60.41 deg. OmniGibson auto-wraps any `Camera` prim that
is a direct child of a link, so it appears as
`robot.sensors["robot0:gripper:Camera:0"]` with no script plumbing.

Two traps:
* `sensor_config` must nest its values under `sensor_kwargs` — `create_sensor()`
  has no `**kwargs`, so a flat dict dies with `unexpected keyword argument
  'image_width'` **during scene load** (job 10882579).
* Do **not** request any `seg_*` modality. Any of them pulls in `seg_semantic`,
  whose remapper calls `get_all_system_categories()`, which `os.listdir()`s
  `deps/BEHAVIOR-1K/datasets/behavior-1k-assets/systems`. That dataset is not
  installed, so every `get_obs()` raises `FileNotFoundError` (job 10882646).
  `depth_linear` alone is enough to verify the render.

### 2. The SIGSEGV was node d4050, not this scene

The py-spy dumps Kit writes beside each minidump put the main thread inside
`SimulationApp.__init__` (`_wait_for_viewport` / `_prepare_ui`) in all four
crashes — **before any scene exists**. All four were on d4050, which is the only
`sharing` l40s node already upgraded to kernel el9_7 / driver 595.58.03; the
others are el9_3 / 570.86.15 and do not crash at startup. `jobs/run_wrist_cam.sbatch`
carries `--exclude=d4050`, warns on driver >= 580, and retries a hard crash.

### 3. Live rendering is reliable; the "detached jaw" is latency

Job 10882722: 105 control steps, 3 live cameras, 315 RGB+depth frames, zero
crashes, 0.081 s/step (12 Hz) at 3 x 640x480.

`tools/check_render_lag.py` back-projects `depth_linear` to world with the pose
the sim reports alongside the image (validated: the scene floor returns at
z = +0.0010 m external, -0.0002 m wrist) and measures per link how far the
reported surface is from the nearest surface Isaac drew. Error is 1.4-3.5 mm for
every link while it is stationary and grows in proportion to its per-step
motion (jaw r = 0.92). With the wrist camera still, the floor reconstructs
0.066 mm / 0.024 deg off; with it moving 42.6 mm and 9.43 deg per step, 36.6 mm /
8.55 deg off — the frame is **~0.9 control steps stale**.

So the renderer is not corrupting the scene. A stale frame plus a jaw moving up
to 88 mm per step is what read as "a detached floating lever". Ordinary
observation latency for a VLA.

Unresolved: `gripper` keeps a ~15 mm residual even in low-motion frames (every
other link drops to 1-3 mm). Likely a measurement artefact — it is the only link
whose thin visible surface is routinely occluded by a *moving* neighbour — but
not run to ground.

### New files
`tools/add_wrist_cam.py`, `tools/check_render_fidelity.py`,
`tools/check_render_lag.py`, `policies/wrist_cam_probe.py`,
`jobs/run_wrist_cam.sbatch`, `robot/so101_usd/so101_wristcam.usd`,
`robot/so101_usd/nvidia_task_env_cfg.py.ref`,
`results/so101/wristcam/run_10882722/`.
Edited: `robot/so101_og.py` (+ `SO101_WRIST_CAM_CFG`, re-copied into
`OmniGibson/omnigibson/robots/so101.py`), `README.md`.

---

## 2026-10-07 — pi0.5 closed-loop on the SO-101 (the VLA bridge)

Full write-up, with the unit maths and the numbers, is in
**[../README.md](../README.md) → "pi0.5 closed-loop on the SO-101"**. This is
the short version plus the things that cost time.

**It ran.** Job **10885036**, one l40s: 450 closed-loop control steps, 45
inferences, live `wrist_cam` + `top_cam` images into pi0.5 and absolute joint
targets back out. 0.196 s per inference, 0.045 s per `env.step()`, 0.000 mm
joint drift across the whole episode. Two more jobs ran the other two
chunk-execution strategies (`openloop50`, `replan1`).

**And it did nothing useful, for a known reason.** The only pi0.5 checkpoint
that fits a 5-DOF SO-101 is `pdhinaka/pi05_so101_blue_cube_gray_box_cleaned_20k`,
fine-tuned on "put the blue cube in the gray box". This scene has an orange
block, a red boat toy and a white cup. The arm folded back, drove
`shoulder_lift` into the joint limit by ~step 200, parked there, and displaced
**0.0 mm** of every object. All three chunk-execution strategies end the same
way, so that is the policy, not the execution choice. Reported, not dressed up.

### Three things that cost real time

1. **`no_proxy='*'` hangs Isaac Sim** (job 10884015, 20 min lost). Omniverse
   Client makes an outbound HTTPS call during scene load; behind squid it fails
   fast, bypassing squid it SYN-retries into a blackhole forever. Symptom:
   hang inside `og.Environment()` right after `Welcome to OmniGibson`, GPU 0%,
   415 threads idle, no log, no traceback, no timeout. Diagnosis was
   `ss -tnp` showing `SYN-SENT ... -> 52.92.202.250:443`. Scope `no_proxy` to
   loopback. Full note in README.
2. **The two envs cannot share a pickle.** numpy 2.2 (`openpi`) writes
   `numpy._core.*`, numpy 1.26 (`simfoundry`) cannot read it. The wire format
   is a length-prefixed `.npz` over a raw TCP socket (raw, so it is also
   immune to the squid-intercepts-loopback problem).
3. **`shoulder_lift` does not fit.** The checkpoint's `observation.state` q01
   *and* q50 for that joint are **-104.19 / -104.17**, below the **-100** that
   the SO-101's `Pitch` ±1.7453 rad limit maps to. The policy commands it, the
   sim clips it, 253/450 steps. The limits→[-100,100] affine map is the right
   *form*, but the arm that recorded the dataset was calibrated wider than the
   USD. Anyone fine-tuning for this sim should reconcile the two.

### New files
`vla/pi05_server.py`, `vla/so101_units.py`, `vla/plot_episode.py`,
`vla/so101_state_quantiles.json`, `policies/pi05_closed_loop.py`,
`jobs/run_pi05_closed_loop.sbatch`, `jobs/run_multiview_vla.sbatch`,
`results/so101/vla/{run_10885036,openloop50,replan1,framed}/`,
`scene_data/reference_demo/s14_og/reconstructed_og_scene_so101_vla.json`
(+ `settled_poses_so101_vla.json`).

### Edited shared files
* `tools/swap_robot_so101.py` — new `--controller {ik,joint}` flag. **The `ik`
  default is byte-for-byte unchanged** (verified by regenerating and `cmp`-ing
  against the previous file), so `policies/pick_and_place.py` is untouched.
* `README.md` — new "pi0.5 closed-loop" section, the `no_proxy` note, results
  table rows, layout.

---

## 2026-10-07 — the blue-cube scene, and the joint-calibration fix

Goal: stop reporting "the checkpoint is for a different task" and actually build
the task it was fine-tuned for, then measure whether it works.

### 1. The calibration mismatch, and why widening the limit alone is wrong

The previous run's finding was that pi0.5 keeps commanding `shoulder_lift`
≈ −104 and the sim clips at −100. The full picture, dumped from the
checkpoint's own normalizer safetensors:

| joint | state min … max | **action** min … max |
|---|---|---|
| shoulder_pan | −33.76 … +27.52 | −33.98 … +28.09 |
| **shoulder_lift** | **−104.26** … +41.41 | **−109.71** … +40.79 |
| elbow_flex | −24.97 … +97.76 | −27.30 … +99.03 |
| **wrist_flex** | +58.42 … +98.59 | +57.85 … **+102.68** |
| wrist_roll | −34.07 … +32.57 | −34.33 … +32.75 |
| gripper | +2.87 … +35.64 | 0.00 … +37.27 |

Two joints run outside ±100. That is not a bug in the data: LeRobot's
`Motor.normalize()` is an **unclamped** affine map onto ±100 over the motor's
*calibrated* range, so a real arm that travels past the range its calibration
sweep covered records values outside it. The physical arm reached ~8.5° below
the mechanical lower limit the SO-101 URDF declares.

**The trap.** You cannot fix this by widening the USD limit on its own, because
the normalized value is *defined* by the range you normalize against: widen the
range and −104 silently stops meaning the same joint angle. `vla/so101_units.py`
originally built its converter from the live robot's limits
(`SO101Units.from_sim_limits`), so widening the asset would have changed the
units underneath the policy and hidden the problem rather than fixing it.

**The fix is to separate the two roles**, which were conflated:

* **`NORM_LIMITS`** — the range ±100 is pinned to. Fixed, equal to the joint's
  nominal limit pair (also what `so101_new_calib.urdf` declares as the
  mechanical range). `rad_to_norm` / `norm_to_rad` use only this, and it never
  tracks the asset.
* **the physical limits** — measured in-sim, used only by `clip_rad()` and by
  the gripper controller's own map. `tools/widen_so101_limits.py` writes
  `robot/so101_usd/so101_wristcam_wide.usd` with

  | joint | limit | was | now | reachable (LeRobot) |
  |---|---|---|---|---|
  | `Pitch` | lower | −100° | **−115°** | −100 → **−115.0** |
  | `Wrist_Pitch` | upper | +95° | **+105°** | +100 → **+110.5** |
  | `Elbow` | upper | +90° | **+95°** | +100 → **+105.3** |

  which covers the dataset's whole action range with headroom. Nothing else in
  the dataset leaves ±100, so nothing else is touched. The original asset is
  untouched — `tools/check_wrist_cam.py`, `tools/episode_geom.py` and every
  existing result read it.

**Verification** (`conda_envs/simfoundry/bin/python -I vla/so101_units.py`,
prints `SO101_UNITS_OK`):

1. round trip rad→norm→rad and norm→rad→norm, 100 k random configurations, max
   error 8.9e−16 rad / 2.8e−14 units;
2. the unit map is **bit-identical** before and after widening (max diff
   **0.000e+00** over 20 k configurations) — the widening cannot move the units;
3. with the **nominal** asset the checkpoint's action range is unreachable on
   exactly `['shoulder_lift', 'wrist_flex']` (the test asserts this, so the
   widening silently not being applied is caught), and with the **widened**
   asset nothing is clipped;
4. the checkpoint's median state q50 and median action q50 both round-trip and
   both survive `clip_rad` untouched under the widened limits.

`from_sim_limits` now prints a per-joint table of nominal vs physical vs
reachable vs required, and warns loudly if a limit is *narrower* than nominal.

### 2. What the training set actually contains (`pdhinaka/so101_blue_cube_in_gray_box_pi05_cleaned`)

50 episodes, 18 736 frames, 30 fps, two 640×480 cameras. Things that changed the
scene design:

* **The prompt is `"Pick up the blue cube and place it in the gray box"`**,
  verbatim from `meta/tasks.parquet` — not "put the blue cube in the gray box",
  which is what the earlier runs used.
* **`top_cam` is not overhead.** It is a third-person camera on the far side of
  the mat, ~30 cm up, ~17 cm beyond the objects, pitched ~59° down, ~63° HFOV,
  with the robot base just past the top edge and the arm reaching toward the
  camera.
* **The layout is recovered by FK, not by eye.** Running the recorded
  `observation.state` through `tools/so101_kin.fk_urdf` at the frames where the
  gripper closes and opens gives, in the robot's own frame, with a ~5 mm spread
  over episodes 0/1/12/24:

  | | tip position (robot frame) |
  |---|---|
  | home (every episode starts **and** ends here) | (0.163, 0.000, 0.024) |
  | grasp — the blue cube | (0.231, −0.086, 0.008) |
  | release — over the gray box | (0.237, +0.116, 0.050) |

  so the two objects sit side by side ~0.20 m apart across the arm's forward
  axis, both ~0.24 m out. Episodes 25–49 are the same scene with the mat turned
  180° and the objects swapped. There is **essentially no randomisation** in
  either half, so the sim uses the recorded positions.
* The **home pose** is `(−1.0, −104.2, 97.4, 76.7, −3.0, 3.0)` — shoulder_lift
  on the bottom stop with the elbow near its top, i.e. the gripper resting on
  the mat 16 cm in front. Note this is NOT the marginal median `q50`
  (`elbow_flex` q50 is 1.05): a vector of per-joint medians is not a pose the
  arm ever holds. The sim's start pose is the real home pose, and it needs the
  widened limits to be representable at all.

### 3. Which way the arm faces — measured, and it was wrong before

`tools/so101_kin.py` fits the URDF→world transform against 450 logged frames of
the previous rollout (5 links × every 13th frame) and gets, with max residual
**7e−6 m**:

```
p_world = root_pos + Rz(-90 deg) @ p_urdf + (0.020790, 0.015760, 0.032480)
```

So with an identity root orientation the arm's forward (+x in the URDF) points
along world **−Y**, not +X. `tools/swap_robot_so101.py` assumes +X and places
objects along it; the scripted pick-and-place never noticed because
`InverseKinematicsController` simply yawed `shoulder_pan` ~70° to reach them.
pi0.5 cannot do that — its `shoulder_pan` only ever spans [−33.98, +28.09]
normalized = [−37°, +31°] — so for the VLA the objects must be in front of the
arm, and front is −Y. The new scene places them there and checks it with IK
**bounded by the checkpoint's own per-joint min/max**: a pose the policy never
visits in training does not count as reachable.

### 4. Scene construction

`tools/make_bluecube_scene.py` writes a standalone scene JSON
(`scene_data/bluecube/bc_*.json`) — it does **not** touch
`reconstructed_og_scene_so101*.json`. Everything is an OmniGibson
`PrimitiveObject` Cube, so no dataset asset and no reconstruction is needed:
a blue cube, a gray box as five fixed slabs (OmniGibson has no concave
primitive, and a convex hull over a hollow box is a solid block the cube could
never enter), a navy desk mat whose top face is flush with z = 0 (so it changes
the look and nothing about the physics) and a cream desk slab under it.

Three ways a hand-written scene JSON fails to load, all found the hard way:

| symptom | cause |
|---|---|
| `KeyError: 'goal'` in `ControllerBase._load_state` | every controller's saved state needs `"goal": null`, not just `goal_is_valid` |
| `KeyError: 'size'` in `PrimitiveObject._load_state` | every Cube needs `"size"` in its state — and it must be **1.0**, not the edge length, because with `scale` given `_extents` stays `ones(3)` and the size setter would rescale the mesh on top of the scale |
| box walls thin along the wrong axis | `scale` is in the body's **local** frame, so every placed body gets the robot's yaw (`Rz(-90°)`); a square box hides the mistake |


### 5. The thing that actually breaks the rollout: the sim arm is 5x too fast

This checkpoint is a **relative-action** policy (`use_relative_actions: true`,
`relative_exclude_joints: ["gripper"]`): what it predicts is `action − state`,
and the postprocessor adds the state back. That makes the *follower's servo
behaviour* part of the control loop, which is not obvious and is easy to get
wrong in sim.

Fitting `s[t+1] = s[t] + α·(a[t] − s[t])` to all 18 736 frames of the training
set, per joint:

| joint | α | τ | per-step residual |
|---|---|---|---|
| shoulder_pan | 0.272 | 122 ms | 0.08 |
| shoulder_lift | 0.198 | 168 ms | 0.59 |
| elbow_flex | 0.235 | 142 ms | 0.25 |
| wrist_flex | 0.199 | 168 ms | 0.22 |
| wrist_roll | 0.273 | 122 ms | 0.08 |
| gripper | 0.071 | 469 ms | 0.32 |

and, restricted to frames where the arm is actually moving, the commanded
position matches the achieved one **3–4 frames later** (residual drops from
~9 units at lag 0 to ~1.3 at lag 4). Over the whole set
`mean(action − state)` on `shoulder_lift` is **−2.37** (−3.96 while the shoulder
sits on its bottom stop) and on the gripper **−2.36**, and that offset does not
vanish at any lag — it is steady-state servo error under gravity and squeeze
force, not latency.

OmniGibson's `JointController` in position mode reaches its target inside one
1/30 s step — measured: commanded and achieved are identical in
`results/so101/vla/framed/episode.npz`. So **α_sim = 1 against α_real ≈ 0.2**,
and because the action is relative, the sim arm advances the *whole* predicted
delta per step where the real arm advanced a fifth of it: the policy's chunk
moves the sim arm through space about **5× faster** than it did in training.

That is visible in the pose log, not inferred. In `w1_match2` the grasp centre
tracks straight to the cube — at step 56 it is at robot-frame
(0.240, −0.103) against the cube's (0.231, −0.086), 27 mm away, with the jaw
already opened to 25 — and then sails straight past it to x = 0.30 while the jaw
stays open. The policy's own open→descend→close sequence takes ~45 frames in the
training data (jaw opens at f41, grasp at f90); in sim the arm covers the whole
approach in ~10 frames and is 80 mm past the cube by the time the close command
arrives.

`policies/pi05_closed_loop.py --follower-alpha dataset` filters the commanded
joint target with the measured α, which makes the sim servo behave like the one
that recorded the data. `--action-offset dataset` separately models the
steady-state error. Both default to OFF, so the uncorrected behaviour stays the
default result.

### 6. Seven rollouts, what each changed, and what it bought

All scored by `tools/score_bluecube.py` from the pose log — cube displacement, a
geometric containment test, the distance from the cube to the **grasp centre**
(the midpoint of the robot's two assisted-grasp pads, reconstructed from the
logged link poses, not the gripper link origin 10 cm behind them), and the
number of frames the jaw was commanded shut while the cube sat between the pads.

| run | what changed vs the row above | cube moved | max lift | closest grasp centre | frames closed on cube | in box |
|---|---|---|---|---|---|---|
| `v1_small` | 2.8 cm cube (sized from the jaw encoder) | 0.0 mm | 0.0 mm | 69.6 mm | 0 | no |
| `v2_match` | 3.8 cm cube + colours measured off the training frames | 15.7 mm | 0.0 mm | 28.7 mm | 15 | no |
| `w1_match2` | mat/floor-plane Z-fighting fixed | 28.2 mm | 0.5 mm | 26.5 mm | 12 | no |
| `w2_white` | arm repainted off-white (`tools/recolor_so101.py`) | 8.8 mm | 0.5 mm | 30.2 mm | 20 | no |
| **`x1_matchcam`** | **wrist camera re-aimed down the approach axis, 90° FOV** | **56.1 mm** | **5.5 mm** | **20.6 mm** | **33** | no |
| `z1_offset` | `--action-offset dataset` on the dataset-matched scene | 75.9 mm | 1.3 mm | 46.0 mm | 0 | no |

Reading them:

* **The cube size matters, and the two ways of measuring it disagree.** The
  jaw encoder says ~2.5 cm (the training set holds the cube at `gripper.pos`
  ≈ 17.5, which on this asset is a 2.9 cm pad separation, and 0.65 cm of that is
  the calibration offset at "closed empty"); the training images say ~3.8 cm
  (the cube is 62 px and the box 162 px where the known 0.202 m cube-to-box
  spacing is 325 px). 3.8 cm wins — the 2.8 cm cube was never touched at all.
* **Z-fighting was doing real damage.** With the mat's top face at exactly z = 0,
  coplanar with OmniGibson's visible floor plane, the wrist camera — a grazing
  wide-angle view a few cm off the table — filled with speckle. Separating the
  surfaces roughly doubled the cube displacement.
* **The wrist camera's aim is the single biggest remaining lever.** Re-aiming it
  down the gripper's approach axis at 90° (`tools/add_wrist_cam.py`, written to
  `so101_wrist_matchcam_wide.usd`) nearly halved the closest approach and
  tripled the frames the jaw was shut on the cube. It is still not the training
  rig: see `real_vs_sim_both_cams.png` — the real wrist camera sits much closer
  to the mat and looks between **two** white jaws filling the bottom third,
  while NVIDIA's authored one sees one jaw off to the side.
* **`--action-offset dataset` made the grasp worse, not better.** It moved the
  cube further (the gripper *body* shoved it) but the grasp centre never came
  within 46 mm. So the per-re-plan steady-state offset is not, on its own, what
  puts the tip 3.7 cm long.
* Repainting the arm white changed nothing measurable, which is itself useful:
  the appearance gap that matters is not the arm's colour.

**What the policy actually does, and it is not nothing.** In `x1_matchcam` it
lifts out of the home pose, swings `shoulder_pan` to the cube's side, descends,
opens the jaw to ~29, closes it, then sweeps `shoulder_pan` across to the box's
side and lifts — a recognisable pick-transport-place, with every joint inside
the envelope of all 50 training episodes for the first ~8 s
(`vs_training.png`). It simply never has the cube in the jaws: at the moment it
closes, the grasp centre is ~20 mm from the cube centre and ~2 cm above it.

The residual is concentrated in two joints. At its grasp moment the sim arm's
joint vector differs from the training grasp pose by **elbow_flex −15** and
**wrist_flex +9**, with everything else inside 2 units — an over-extended
forearm that puts the tip ~3.7 cm too far forward and ~2.5 cm too high.

### 7. The policy CAN do the task. Two sim-side blockers, found by doing it.

**Blocker A — the object positions.** Converting the dataset's joint angles into
Cartesian object positions requires assuming the sim joint's zero and sign
conventions match whatever physical arm recorded the data. That assumption is
unverifiable (the checkpoint ships no calibration) and it is flagged in
`vla/so101_units.py`. It is also, measurably, slightly wrong: in every rollout at
the dataset-derived layout the arm's reach bottoms out ~3.7 cm further forward
than the cube. Moving BOTH objects (and the `top_cam`, so the framing is
unchanged) 3.7 cm along the arm's forward axis, to where the policy's own
trained poses actually land in sim — `scene_data/bluecube/bc_shift.json`, run
`d1_shift` — changes everything:

| | dataset-derived layout (`x1_matchcam`) | +3.7 cm (`d1_shift`) |
|---|---|---|
| cube net displacement | 56 mm | **195 mm** |
| cube path length | — | **1232 mm** |
| max lift off the table | 5.5 mm | **152.6 mm** |
| frames with the jaw shut on the cube | 33 | **211** |
| commanded actions inside the checkpoint's own [q01, q99] | 43–100% per joint | **100% on all six joints** |
| commands clipped by a joint limit | 0/450 | 0/450 |

The policy **grasps the blue cube, lifts it 15 cm, carries it 1.2 m of path and
holds it over the gray box** — final cube position (0.1006, −0.2627, 0.1271)
against a box centre of (0.1378, −0.2582), i.e. 37 mm off centre horizontally
over an 80 mm interior, 12.7 cm up. That is a real closed-loop manipulation
episode driven by a real VLA from live sim images.

**Blocker B — it cannot let go, and that is OmniGibson's rule, not the
policy's.** `ManipulationRobot._handle_assisted_grasping` computes, for a
position-controlled finger,

```python
applying_grasp = control < joint_upper_limit
```

so the assisted-grasp constraint is released only when the jaw is commanded to
its **full mechanical limit**. pi0.5's gripper action never exceeds **37.3/100**
— the real SO-101's gripper calibration only ever covered about a third of the
jaw's travel, and the dataset's own *release* commands are ~23/100. So
`applying_grasp` is permanently true, `_ag_freeze_gripper` stays set, and the
jaw freezes. `d1_shift/vs_training.png` shows it exactly: from t = 3 s the
policy's gripper target (orange) oscillates between 5 and 30 for ten seconds
while the achieved jaw position (red) is pinned flat at 24.3 and the cube never
drops. `sticky` mode uses the same test.

So "assisted" grasping — the mode every other result in this repo uses — is
structurally incompatible with a policy whose gripper calibration is narrower
than the simulated joint's. `tools/make_bluecube_scene.py --grasping-mode
physical` removes the magic constraint entirely, which is the only release
condition this policy can express.

### 8. Reproducibility of the pick, and the files

The pick is **not** deterministic. pi0.5 samples a flow-matching trajectory, so
two runs of the identical scene differ. At the +3.7 cm layout:

| run | steps | frames jaw shut on cube | max lift | outcome |
|---|---|---|---|---|
| `d1_shift` | 450 | 211 | **152.6 mm** | grasped, carried, held over the box |
| `d2_shift_long` | 800 | 113 | 1.9 mm | closed on it repeatedly, never lifted |

So 1 of 2 attempts picked the cube up. That is worth stating plainly rather than
reporting the good one: the policy is close to the edge of competence here, not
comfortably inside it.

**And `--grasping-mode physical` does not rescue the release, because it cannot
hold.** Two runs on the same scene and the same 3.7 cm shift, with
`finger_static_friction` and `finger_dynamic_friction` raised to 2.5, 700 steps
each:

| run | frames jaw shut on cube | max lift | cube moved |
|---|---|---|---|
| `d4_phys` | 2 | 5.3 mm | 40.3 mm |
| `d5_phys2` | **181** | **3.1 mm** | 51.4 mm |

`d5_phys2` is the decisive one: the jaw closed on the cube for 181 frames — six
seconds of contact at the right place — and the cube never came more than 3 mm
off the mat. The SO-101's gripper is a single lever closing against a static
face under position control; without the magic constraint there is simply not
enough grip to lift a 3.8 cm block, and the cube slips out every time.

**So the complete picture is a pincer:**

| grasping mode | can it pick the cube up? | can it put it down? |
|---|---|---|
| `assisted` / `sticky` | **yes** (1 of 2 attempts, 15 cm lift, carried over the box) | **no** — release needs a full-limit jaw command the policy never issues |
| `physical` | **no** — 181 frames of jaw-on-cube contact, 3.1 mm of lift; a 1-DOF lever under position control cannot hold a 3.8 cm block | n/a |

Neither mode can complete pick **and** place. This is an OmniGibson/SO-101
limitation, not a statement about the policy: the policy produced a correct
pick-up and transport, with 100 % of its commanded actions inside its own
training distribution on all six joints and zero joint-limit clipping. Closing
the loop would need either a release rule that triggers on the *relative* jaw
command rather than the absolute limit, or a compliant two-finger gripper.

### New files

```
tools/widen_so101_limits.py        the physical-limit widening (writes a new usd)
tools/recolor_so101.py             repaint the printed parts (writes a new usd)
tools/so101_kin.py                 URDF FK + the measured URDF->world transform
tools/make_bluecube_scene.py       the scene generator
tools/score_bluecube.py            the objective scorer (writes score.json)
tools/bluecube_sheet.py            live filmstrip + real-vs-sim sheets
tools/plot_bluecube_vs_training.py sim joints vs all 50 training episodes
tools/make_bluecube_geom_cache.py  procedural geometry for the offline renderers
jobs/submit_bluecube.sh            one rollout, all knobs as env vars
jobs/finish_bluecube.sh            score + sheets + optional 6-angle render
robot/so101_usd/so101_wristcam_wide.usd         widened joint limits
robot/so101_usd/so101_wristcam_wide_white.usd   + printed parts off-white
robot/so101_usd/so101_wrist_matchcam_wide.usd   + wrist cam re-aimed, 90 deg
scene_data/bluecube/bc_*.json                   the scenes (+ *_layout.json)
results/so101/vla/bluecube/dataset_ref/         the real training frames
results/so101/vla/bluecube/<run>/               per-run outputs + score.json
```

### Edited shared files

* `vla/so101_units.py` — `NORM_LIMITS` (the unit system) separated from the
  physical limits; `from_sim_limits` reports both; the self-test now proves the
  separation. `LIMITS` is kept as an alias, and `SO101Units(limits=...)` still
  means "both", so every existing caller is unchanged.
* `policies/pi05_closed_loop.py` — `--top-eye` / `--top-target` / `--top-focal`
  / `--frame-objects` (explicit `top_cam` placement), `--action-offset`,
  `--follower-alpha`, and `policy_target_lerobot` in `episode.npz`. All default
  to the previous behaviour.
* `jobs/run_pi05_closed_loop.sbatch` — passes the new flags through.

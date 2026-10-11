# sim-foundry-rep

Reproducing **NVlabs/SimFoundry** on an HPC cluster, then driving the reconstructed
scenes with a robot arm — first a scripted policy, then the **π0.5 vision-language-action**
model.

Three questions, answered with measurements rather than impressions:

1. Can SimFoundry turn a phone video of a desk into a physics-ready simulation? **Yes.**
2. Can a robot arm manipulate the reconstructed objects? **Yes** — scripted, with an SO-101.
3. Can an off-the-shelf VLA policy do the same? **No**, and the reason is specific and
   measured. See [Why π0.5 doesn't work here](#why-π05-does-not-work-here).

---

## Getting it

Media (videos, figures) is stored with **Git LFS**. Without it you get small
pointer files instead of the actual `.mp4`/`.png`.

```bash
git lfs install                 # once per machine
git clone https://github.com/keivalya/sim-foundry-rep.git
```

Already cloned without LFS? `git lfs install && git lfs pull`.

No root? LFS installs fine to userspace:

```bash
curl -sSL -o lfs.tar.gz https://github.com/git-lfs/git-lfs/releases/download/v3.5.1/git-lfs-linux-amd64-v3.5.1.tar.gz
tar xzf lfs.tar.gz && mkdir -p ~/bin && cp git-lfs-*/git-lfs ~/bin/ && export PATH=~/bin:$PATH
```

Code, docs and the campaign scoreboards are plain git — clone without LFS and
everything except the media still works.

---

## Results

### Reconstruction: real video → simulation

![real vs sim](results/reference_scene/real_vs_sim_aligned.png)

Left: a frame from the source video. Middle: the reconstructed scene rendered from the
**same camera** — recovered from the pipeline's own `cam2world` and Depth-Anything-3
intrinsics, so the two are in pixel correspondence. Right: a 50/50 overlay.

The cup, boat and block land close. The white whale toy visible in the real frame is
*absent* from sim — it was pruned by a boundary filter, which is the kind of failure this
comparison is designed to surface.

### Manipulation that works: scripted SO-101

[`results/so101/multiview/videos/so101_grid_3x2.mp4`](results/so101/multiview/videos/so101_grid_3x2.mp4)
— six camera angles, both objects picked up and placed.

| object | width | grasped | displaced | outcome |
|---|---|---|---|---|
| orange block toy | 2.5 cm | yes | 0.079 m | **success** |
| red boat toy | 4.1 cm | yes | 0.127 m | **success** |
| white paper cup | 12.8 cm | — | — | too wide for the 8 cm jaw |

### π0.5 closed-loop: runs, but does not solve the task

[`results/so101/vla/bluecube/c1007_1702_s2/videos/so101_pi05_live_pair.mp4`](results/so101/vla/bluecube/c1007_1702_s2/videos/so101_pi05_live_pair.mp4)
— the two camera feeds the policy actually consumed.

The best of 29 rollouts: the policy grasped the cube, lifted it 185 mm, carried it 1.8 m
and held it **over the box** — then never released. No run ever completed the task.

---

## Why π0.5 does not work here

This is the main finding, so it is stated plainly.

**No official π0/π0.5 checkpoint fits a 5-DOF SO-101.** `openpi` ships exactly three
embodiments — DROID, ALOHA, LIBERO. Their action spaces are 8, 14 and 7 dimensional; the
SO-101 is 6. The DROID normalisation statistics settle it: dimensions 0–7 carry real
quantiles, 8–31 are all exactly `0.0`, so the model only ever learned 8 dimensions.
A search for `so-101`/`so100` across the whole repo returns nothing.

**A community fine-tune exists and we got it running** —
`pdhinaka/pi05_so101_blue_cube_gray_box_cleaned_20k`, 6-dim state and action, matching our
joint set. It was trained on *"put the blue cube in the gray box"*, so we built exactly
that scene from the training data (joint angles run through forward kinematics to recover
where the cube and box actually sat, repeatable to ~5 mm).

**It still does not transfer.** Across 29 rollouts spanning four threshold settings:

| | path | lift |
|---|---|---|
| best run (`c1007_1702_s2`) | 1805 mm | 185 mm |
| second best | 257 mm | 15 mm |
| all others | ≤151 mm | ≤13 mm |

That best run is a 7× outlier. A **controlled replication** — four runs with its exact
configuration, varying only the policy's stochastic sampling — produced 57–228 mm and
nothing above 800 mm. So it was a lucky sample, not a configuration effect.

The honest conclusion: the checkpoint was trained on **real** SO-101 hardware, and the
real→sim appearance gap means the policy mostly does not recognise the scene. It
occasionally stumbles into a grasp the way a weak prior occasionally lands.

**The route to a working policy is fine-tuning `lerobot/pi05_base` on demonstrations
collected in this simulator** — which closes the appearance gap by construction, and which
the scripted policy can generate. Everything needed is in place: calibration, scene,
camera rig, grasp mechanics, scoring harness.

---

## Things that cost days, written down so they don't cost yours

**Joint calibration beats everything.** The checkpoint commands `shoulder_lift` to −109.7
and `wrist_flex` to +102.7 — both outside LeRobot's ±100. The real arm travelled past the
range its calibration sweep covered. Widening the USD limit alone is wrong, because the
normalised value is *defined* by the range you normalise against. `vla/so101_units.py`
separates the two roles: `NORM_LIMITS` pins ±100 and never moves; physical limits come from
the asset and feed only clipping. Before the fix, `shoulder_lift` was clipped on 253/450
steps and the arm parked on the stop.

**Isaac Sim 5.1 only works on the NVIDIA 570 driver branch.** Nodes outside it segfault
during `SimulationApp.__init__`, before any scene exists. Measured: driver 545 → 0/5 jobs
succeeded; 595 → 0/7; 570.86 → 17/26. `jobs/run_pi05_closed_loop.sbatch` now checks the
driver at startup and exits fast with a distinct code rather than burning an allocation.

**A renderer crash is not an asset bug.** Several days went into "fixing" a USD that
measured identical to the original on mesh counts, vertex positions (0.000 mm), materials,
winding and visibility. The crashes were one bad node, and the apparent "detached gripper
jaw" was ~0.9 control steps of ordinary observation latency — every link sits within
1–3.5 mm when stationary and the error scales with its own motion (jaw correlation
r = 0.92).

**OmniGibson's assisted grasp releases only at the full mechanical limit.** π0.5's gripper
command never exceeds 37/100, so the constraint never drops. `robot/so101_og.py` replaces
that test with a calibrated threshold measured from the training data.

**`sbatch --export` splits its value on commas.** `TOP_EYE=0.03,-0.42,0.30` arrives as
`0.03`. Vector-valued knobs use `+` as a separator and translate it back.

**Do not set `no_proxy='*'` on this cluster.** Omniverse Client makes an outbound HTTPS
call during scene load; with the proxy disabled the cluster blackholes it and it SYN-retries
forever — no log line, no timeout, just a hung job.

---

## Layout

```
├── policies/     pick_and_place.py, pi05_closed_loop.py, probes
├── robot/        SO101 OmniGibson robot class (so101_og.py)
├── tools/        renderers, scene building, scoring, asset preparation
├── vla/          π0.5 server, unit conversion, checkpoint prep, smoke tests
├── jobs/         Slurm scripts + the self-contained campaign runner
├── docs/         STATUS.md (full engineering log), RUNNING_ON_EXPLORER.md
└── results/      videos, figures, campaign scoreboards
```

**`docs/STATUS.md` is the detailed engineering log** — every fix, every failure mode, every
measurement, with the reasoning. Start there if you are picking up the work.

### What is not in this repo, and why

| excluded | size | how to get it |
|---|---|---|
| `robot/so101_usd/*.usd` | 156 MB | `tools/deinstance_so101.py` rebuilds them from NVIDIA's asset |
| raw per-frame PNGs | ~10 GB | the mp4s built from them *are* committed |
| `vla/lerobot_src`, `vla/openpi_src` | 24 MB | upstream clones; see below |
| `assets_hf/` | 343 MB | `nadunRanawaka1/simfoundry-assets` on HuggingFace |
| NVIDIA's published gallery | 181 MB | from the SimFoundry repo |
| failed-variant rollout videos | ~500 MB | 28 of 29 runs; scoreboards are committed |

---

## Reproducing

This was built on Northeastern's Explorer cluster (Rocky 9, Slurm, no OpenGL on compute
nodes). **`docs/RUNNING_ON_EXPLORER.md` has the cluster-specific detail**; the outline:

```bash
# 1. SO-101 asset: fetch NVIDIA's USD (Git LFS) and make it OmniGibson-loadable
curl -sSL -o SO-ARM101-USD-NO-CAMERA.usd \
  https://media.githubusercontent.com/media/isaac-sim/Sim-to-Real-SO-101-Workshop/main/source/sim_to_real_so101/assets/usd/SO-ARM101-USD-NO-CAMERA.usd
python tools/deinstance_so101.py SO-ARM101-USD-NO-CAMERA.usd robot/so101_usd/so101.usd
python tools/add_wrist_cam.py robot/so101_usd/so101.usd robot/so101_usd/so101_wristcam.usd
python tools/widen_so101_limits.py        # the calibration fix — do not skip

# 2. put the SO-101 into a reconstructed scene
python tools/swap_robot_so101.py --scene-dir <scene> [--controller joint]

# 3. scripted pick-and-place
sbatch jobs/run_pick_place.sbatch

# 4. π0.5 closed-loop (policy server + sim client, one GPU)
bash jobs/bluecube_campaign.sh 6 shift 700
```

`tools/deinstance_so101.py` applies four fixes NVIDIA's asset needs before OmniGibson can
load it: de-instancing (USD instance proxies are read-only), authoring a full
`[translate, orient, scale]` triple (a partial stack gets mangled on load), per-mesh
`CollisionAPI` (OmniGibson tests it per mesh, not per group), and convex-hull colliders
(the raw STLs are 1.2 M vertices and never finish cooking).

### Rendering without a GPU

Compute nodes here have CUDA but **no OpenGL/EGL at all**, so `tools/sfrender.py` is a pure
NumPy rasteriser — real pinhole camera, z-buffer, textured faces, per-pixel contact
shadows. `tools/render_episode_multiview.py` renders a recorded episode from six angles
including a wrist-eye view matching the real sensor's extrinsic.

It is also how the VLA videos are made: run with `POSES_ONLY=1` so Isaac renders nothing
and only logs poses, then rebuild offline. That is ~10× faster and sidesteps the renderer
entirely.

---

## Checking what is running

```bash
bash jobs/status.sh
```

Slurm queue, recent completions, campaign scoreboards, all rollouts ranked, key output
paths. Reads only Slurm and disk, so it works from any session.

Long campaigns are launched via `jobs/bluecube_launch.sbatch`, which runs the campaign
*on the cluster*: it submits the rollouts plus a `--dependency=afterany` finisher that
scores them, picks the best and submits the render. Submit it and close your terminal.

---

## Credits

[NVlabs/SimFoundry](https://github.com/NVlabs/SimFoundry) · 
[StanfordVL/BEHAVIOR-1K + OmniGibson](https://github.com/StanfordVL/BEHAVIOR-1K) · 
[TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) · 
[isaac-sim/Sim-to-Real-SO-101-Workshop](https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop) · 
[huggingface/lerobot](https://github.com/huggingface/lerobot) · 
[Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)

"""Drop-in replacement for vla/pi05_server.py that serves a RECORDED training
episode instead of pi0.5's samples.

WHY THIS EXISTS
---------------
It separates two questions that the closed-loop runs conflate:

  (a) does the SIM let a correct SO-101 trajectory pick the cube up and put it
      down -- i.e. is the assisted-grasp latch/release rule right?
  (b) does pi0.5, from live sim images, PRODUCE a correct trajectory?

Answering (a) needs a trajectory known to be correct, and there are 50 of them:
the episodes the checkpoint was fine-tuned on. This server ignores the images
entirely and replies with the next chunk of
`pdhinaka/so101_blue_cube_in_gray_box_pi05_cleaned`'s recorded `action` column,
which is exactly the format the client already expects (absolute LeRobot
normalized units). So the whole rig -- scene, robot, cameras, pose log, scorer,
video pipeline -- runs unchanged, and the only thing that changes is who is
driving. If the replay grasps and releases, the sim-side rule is right and any
remaining failure is the policy's trajectory, not OmniGibson's grasping.

It is an OPEN-LOOP replay: the cursor advances by --exec-steps per request,
which is exactly how often policies/pi05_closed_loop.py asks for a new chunk,
so request k serves recorded frames [k*exec_steps, k*exec_steps + chunk).

Run it against the DATASET-DERIVED layout (scene_data/bluecube/bc_matchcam.json),
not the +3.7 cm shifted one: the recorded joint angles reach where the real
objects were, and bc_shift deliberately moves them.

Wire format, arguments and the proxy/pickle reasoning are pi05_server.py's;
this reuses its pack/read_frame so the two cannot drift apart.
"""
import argparse
import glob
import os
import socket
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pi05_server import JOINTS, pack, read_frame  # noqa: E402

SNAP = ("/scratch/pandya.kei/hf_cache/hub/datasets--pdhinaka--"
        "so101_blue_cube_in_gray_box_pi05_cleaned/snapshots/*/")


def load_episode(ep, log):
    import pandas as pd
    hits = sorted(glob.glob(os.path.join(SNAP, "data", "chunk-*", "*.parquet")))
    assert hits, f"no parquet under {SNAP}"
    df = pd.concat([pd.read_parquet(h) for h in hits], ignore_index=True)
    m = df["episode_index"].values == ep
    assert m.any(), f"episode {ep} not in the dataset"
    act = np.stack(df["action"].values[m]).astype(np.float32)
    st = np.stack(df["observation.state"].values[m]).astype(np.float32)
    log(f"episode {ep}: {len(act)} frames ({len(act)/30:.1f}s at 30 Hz)")
    log(f"  start state  {np.array2string(st[0], precision=2)}")
    log(f"  gripper action: min {act[:,5].min():.2f} max {act[:,5].max():.2f}")
    return act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PI05_PORT", 48051)))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--ready-file", default="")
    ap.add_argument("--episode", type=int, default=int(os.environ.get("REPLAY_EPISODE", 0)))
    ap.add_argument("--exec-steps", type=int, default=int(os.environ.get("EXEC_STEPS", 10)),
                    help="MUST match the client's --exec-steps; it is how far the "
                         "cursor advances per request.")
    ap.add_argument("--chunk", type=int, default=50, help="chunk length to serve")
    ap.add_argument("--hold-last", action="store_true",
                    help="past the end of the episode, keep serving its last "
                         "frame instead of stopping (default: hold)")
    ap.add_argument("--max-idle", type=float, default=1800.0)
    # Accepted and ignored, so this is a drop-in for pi05_server.py's command line.
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--seed", type=int, default=-1)
    a = ap.parse_args()

    def log(m):
        print(f"[replay] {m}", flush=True)

    act = load_episode(a.episode, log)
    n = len(act)
    log(f"serving chunks of {a.chunk}, cursor += {a.exec_steps} per request")

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((a.host, a.port))
    srv.listen(4)
    srv.settimeout(a.max_idle)
    log(f"listening on {a.host}:{a.port}")
    if a.ready_file:
        with open(a.ready_file, "w") as f:
            f.write(f"{a.port}\n")

    conn, stop, calls = None, False, 0
    try:
        while not stop:
            conn, addr = srv.accept()
            conn.settimeout(a.max_idle)
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            log(f"client connected from {addr}")
            k = 0
            while True:
                try:
                    req = read_frame(conn)
                except (ConnectionError, OSError) as e:
                    log(f"client gone ({type(e).__name__}: {e})")
                    break
                if "stop" in req.files:
                    log("stop requested")
                    stop = True
                    break
                t0 = time.time()
                cur = k * a.exec_steps
                idx = np.clip(np.arange(cur, cur + a.chunk), 0, n - 1)
                chunk = act[idx]
                state = np.asarray(req["state"], float).ravel()
                conn.sendall(pack(chunk=chunk.astype(np.float32),
                                  state_fed=state,
                                  clamped=np.zeros(6, bool),
                                  infer_s=np.float64(time.time() - t0),
                                  bins=np.zeros(6, np.int32)))
                k += 1
                calls += 1
                if k <= 3 or k % 5 == 0:
                    log(f"#{k} frame {cur}/{n}  a0={np.array2string(chunk[0], precision=1)}"
                        + ("  [past end, holding last]" if cur >= n else ""))
            try:
                conn.close()
            except Exception:
                pass
            conn = None
    except socket.timeout:
        log("idle timeout, no client")
    except BaseException:
        import traceback
        traceback.print_exc()
    finally:
        log(f"served {calls} requests from episode {a.episode}")
        try:
            if conn:
                conn.close()
        finally:
            srv.close()
        log("exit")


if __name__ == "__main__":
    main()

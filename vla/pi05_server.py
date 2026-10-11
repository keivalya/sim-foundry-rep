"""pi0.5 SO-101 policy server. Runs in the `openpi` env; the sim client runs in
`simfoundry`. One Slurm job, two processes, localhost.

WHY A SERVER AT ALL
-------------------
OmniGibson and LeRobot cannot share a Python env (numpy 1.26 vs 2.2, and
conflicting transformers pins), so the policy and the simulator have to be two
processes. They are started by the same sbatch and talk over loopback.

WIRE FORMAT: RAW TCP, NOT HTTP
------------------------------
Two reasons, both bites-you-at-3am:

1. The cluster's squid proxy is in http_proxy/https_proxy and INTERCEPTS
   127.0.0.1. An HTTP client would get a squid 403 that reads exactly like an
   application auth error. A raw socket cannot be proxied, so the problem
   cannot occur. (NO_PROXY is also set in the sbatch, belt and braces.)
2. numpy 2.2 (here) pickles arrays as `numpy._core.*`, which numpy 1.26 (the
   sim env) cannot unpickle. So the payload is an .npz -- the NPY format is
   version-stable and is read with allow_pickle=False.

Frame = 8-byte big-endian length + npz bytes.
  request : state(6) float64 in LeRobot normalized units, wrist_cam / top_cam
            uint8 HxWx3, task (unicode scalar)
  reply   : chunk (T,6) float32 ABSOLUTE LeRobot normalized units,
            plus clamped(6) bool, state_fed(6), infer_s scalar

THE np.digitize GUARD
---------------------
pi0.5 writes the state into the PROMPT as 256 discrete bins. The binning in
lerobot's pi05 processor is an unclipped np.digitize, so a normalized value
outside [-1, 1] produces bin -1 and a malformed prompt (seen in the smoke test:
'State: 149 121 152 -1 115 118'). Normalization here is QUANTILES, i.e.
2*(x-q01)/(q99-q01)-1, so clamping the RAW state into [q01, q99] -- read out of
the checkpoint's own normalizer safetensors -- is exactly equivalent to clamping
the normalized state to [-1, 1], and is done before the state is handed over.
Every clamp is reported back to the client and counted, because a clamp means
the arm is outside the states this checkpoint was ever trained on.
"""
import argparse
import io
import os
import socket
import struct
import sys
import time

import numpy as np

CKPT = os.environ.get("SO101_CKPT",
                      "/scratch/pandya.kei/simfoundry-repro/vla/ckpt_so101_pi05_lerobot061")
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
HDR = struct.Struct(">Q")


# ---------------------------------------------------------------- wire ------
def pack(**arrays):
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    b = buf.getvalue()
    return HDR.pack(len(b)) + b


def recv_exactly(sock, n):
    chunks, got = [], 0
    while got < n:
        c = sock.recv(min(1 << 20, n - got))
        if not c:
            raise ConnectionError(f"peer closed after {got}/{n} bytes")
        chunks.append(c)
        got += len(c)
    return b"".join(chunks)


def read_frame(sock):
    (n,) = HDR.unpack(recv_exactly(sock, HDR.size))
    return np.load(io.BytesIO(recv_exactly(sock, n)), allow_pickle=False)


# ---------------------------------------------------------------- policy ----
class Pi05SO101:
    def __init__(self, ckpt=CKPT, device=None, log=print):
        import torch
        from lerobot.policies.pi05 import PI05Policy
        import lerobot.policies.pi05.processor_pi05  # noqa: F401  registers the state tokenizer
        from lerobot.processor import (PolicyProcessorPipeline, policy_action_to_transition,
                                       transition_to_policy_action)
        from safetensors.numpy import load_file

        self.torch = torch
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        log(f"torch {torch.__version__} device={self.dev}")
        if self.dev == "cuda":
            log(f"gpu: {torch.cuda.get_device_name(0)} {torch.cuda.get_device_capability(0)}")

        t0 = time.time()
        self.policy = PI05Policy.from_pretrained(ckpt).to(self.dev).eval()
        self.cfg = self.policy.config
        ov = {"device_processor": {"device": self.dev}}
        self.pre = PolicyProcessorPipeline.from_pretrained(
            ckpt, config_filename="policy_preprocessor.json", overrides=ov)
        self.post = PolicyProcessorPipeline.from_pretrained(
            ckpt, config_filename="policy_postprocessor.json", overrides=ov,
            to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
        # The absolute-action step needs the preprocessor's relative step to invert it.
        # This pairing does NOT wire itself up; without it the postprocessor returns
        # relative deltas that look plausible and are wrong.
        rel = next(s for s in self.pre.steps if type(s).__name__ == "RelativeActionsProcessorStep")
        next(s for s in self.post.steps
             if type(s).__name__ == "AbsoluteActionsProcessorStep").relative_step = rel

        st = load_file(os.path.join(ckpt, "policy_preprocessor_step_3_normalizer_processor.safetensors"))
        self.q01 = st["observation.state.q01"].astype(float).ravel()
        self.q99 = st["observation.state.q99"].astype(float).ravel()

        self.cams = list(self.cfg.image_features)
        log(f"loaded in {time.time()-t0:.1f}s  chunk={self.cfg.chunk_size} "
            f"denoise={self.cfg.num_inference_steps} dtype={self.cfg.dtype} cams={self.cams}")
        log("state q01: " + np.array2string(self.q01, precision=2))
        log("state q99: " + np.array2string(self.q99, precision=2))
        self.clamp_count = np.zeros(6, dtype=int)
        self.n_calls = 0

    def short_name(self, feat):
        return feat.rsplit(".", 1)[-1]          # observation.images.wrist_cam -> wrist_cam

    def infer(self, state, images, task):
        """state: (6,) LeRobot normalized. images: {short_name: uint8 HxWx3}."""
        th = self.torch
        state = np.asarray(state, float).ravel()
        fed = np.clip(state, self.q01, self.q99)
        clamped = fed != state
        self.clamp_count += clamped
        self.n_calls += 1

        batch = {"observation.state": th.tensor(fed[None], dtype=th.float32),
                 "task": [task]}
        for feat in self.cams:
            img = images[self.short_name(feat)]
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            # [H,W,3] uint8 -> [1,3,H,W] float in [0,1]; the model resizes to 224.
            batch[feat] = th.tensor(img.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)

        processed = self.pre(batch)
        prompt = processed["task"][0]
        bins = [int(x) for x in prompt.split("State: ")[1].split(";")[0].split()]
        with th.no_grad():
            raw = self.policy.predict_action_chunk(processed)
        chunk = self.post(raw).detach().float().cpu().numpy()[0]   # (T, 6) absolute
        return chunk, fed, clamped, bins, prompt


def _synth_images(cams, seed=0):
    g = np.random.default_rng(seed)
    return {c: g.integers(0, 255, (480, 640, 3), dtype=np.uint8) for c in cams}


# ---------------------------------------------------------------- main ------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PI05_PORT", 48051)))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--ready-file", default="",
                    help="touched once the model is loaded and the socket is listening")
    ap.add_argument("--selftest", action="store_true",
                    help="run one synthetic inference before serving")
    ap.add_argument("--max-idle", type=float, default=1800.0)
    ap.add_argument("--seed", type=int, default=-1,
                    help="torch RNG seed. pi0.5 SAMPLES a flow-matching "
                         "trajectory, so two runs of an identical scene differ "
                         "(1 of 2 attempts picked the cube up on 2026-10-07). "
                         "Seeding makes a sample reproducible, and lets parallel "
                         "jobs draw DIFFERENT, NAMED samples instead of relying "
                         "on process entropy. -1 (default) leaves the RNG "
                         "untouched, i.e. the previous behaviour.")
    a = ap.parse_args()

    def log(m):
        print(f"[server] {m}", flush=True)

    pol = Pi05SO101(a.ckpt, log=log)
    if a.seed >= 0:
        # Seed AFTER load (from_pretrained consumes entropy) and BEFORE the
        # selftest, so the first served inference is at a fixed RNG offset.
        pol.torch.manual_seed(a.seed)
        if pol.dev == "cuda":
            pol.torch.cuda.manual_seed_all(a.seed)
        log(f"torch seed = {a.seed}")

    if a.selftest:
        # The rest pose this scene starts from, in LeRobot units (vla/so101_units.py).
        st = np.array([-14.25, -35.00, 0.77, 91.36, -57.42, 1.46])
        t0 = time.time()
        ch, fed, cl, bins, prompt = pol.infer(
            st, _synth_images([pol.short_name(c) for c in pol.cams]),
            "put the blue cube in the gray box")
        log(f"selftest: chunk {ch.shape} in {time.time()-t0:.2f}s")
        log(f"  state in  {np.array2string(st, precision=2)}")
        log(f"  state fed {np.array2string(fed, precision=2)}  clamped={[JOINTS[i] for i in np.where(cl)[0]]}")
        log(f"  bins {bins}  (all in [0,255]: {all(0 <= b <= 255 for b in bins)})")
        log(f"  action[0] {np.array2string(ch[0], precision=2)}")
        log(f"  action[-1] {np.array2string(ch[-1], precision=2)}")
        pol.clamp_count[:] = 0
        pol.n_calls = 0

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((a.host, a.port))
    srv.listen(4)
    srv.settimeout(a.max_idle)
    log(f"listening on {a.host}:{a.port}")
    if a.ready_file:
        with open(a.ready_file, "w") as f:
            f.write(f"{a.port}\n")

    # Accept clients in a LOOP, not once: jobs/run_pi05_closed_loop.sbatch retries
    # the sim on a hard Isaac crash (it segfaults in SimulationApp.__init__ on
    # some nodes), and the model takes ~2 min to load, so the server has to
    # outlive a dead client and let the next attempt reconnect.
    conn = None
    stop = False
    try:
        while not stop:
            conn, addr = srv.accept()
            conn.settimeout(a.max_idle)
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            log(f"client connected from {addr}")
            n = 0
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
                imgs = {k: req[k] for k in req.files if k.endswith("_cam")}
                ch, fed, cl, bins, prompt = pol.infer(
                    req["state"], imgs, str(req["task"]))
                dt = time.time() - t0
                conn.sendall(pack(chunk=ch.astype(np.float32), state_fed=fed,
                                  clamped=cl, infer_s=np.float64(dt),
                                  bins=np.asarray(bins, np.int32)))
                n += 1
                if n <= 3 or n % 10 == 0:
                    log(f"#{n} {dt:.3f}s  bins={bins}  a0="
                        f"{np.array2string(ch[0], precision=1)}")
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
        if pol.n_calls:
            log(f"served {pol.n_calls} requests; state clamped per joint: " +
                ", ".join(f"{j}={int(c)}" for j, c in zip(JOINTS, pol.clamp_count)))
        try:
            if conn:
                conn.close()
        finally:
            srv.close()
        log("exit")


if __name__ == "__main__":
    main()

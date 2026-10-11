"""Minimal pi0.5 inference smoke test on synthetic data.

Case A: lerobot/pi05_base as shipped (32-dim padded state/action).
Case B: same weights, output sliced to SO-101's 6 DoF, with SO-101-shaped norm stats.

Nothing here is a claim that the actions are meaningful for SO-101 -- pi05_base is a
pre-trained *base* model and was never fine-tuned on SO-101.
"""

import os
import numpy as np
import torch

REPO = os.environ.get("PI05_REPO", "lerobot/pi05_base")
# SO-101 joint order, from the LeRobot SO-101 follower config.
SO101_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def synth_batch(state_dim, bsz=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "observation.images.base_0_rgb": torch.rand(bsz, 3, 224, 224, generator=g),
        "observation.images.left_wrist_0_rgb": torch.rand(bsz, 3, 224, 224, generator=g),
        "observation.images.right_wrist_0_rgb": torch.rand(bsz, 3, 224, 224, generator=g),
        "observation.state": torch.rand(bsz, state_dim, generator=g) * 2 - 1,
        "task": ["pick up the red block and put it in the box"] * bsz,
    }


def report(name, chunk):
    a = chunk.detach().float().cpu().numpy()
    print(f"\n--- {name} ---")
    print(f"  shape          : {tuple(a.shape)}  (batch, horizon, action_dim)")
    print(f"  dtype          : {a.dtype}")
    print(f"  min/max/mean   : {a.min():.4f} / {a.max():.4f} / {a.mean():.4f}")
    print(f"  per-dim std    : {np.array2string(a[0].std(axis=0), precision=4)}")
    print(f"  first 3 steps  :\n{np.array2string(a[0, :3], precision=4, suppress_small=True)}")
    print(f"  last step      : {np.array2string(a[0, -1], precision=4, suppress_small=True)}")


def main():
    from lerobot.policies.pi05 import PI05Policy, make_pi05_pre_post_processors
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.configs.types import FeatureType, PolicyFeature

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__}  device={dev}")
    if dev == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}  cap={torch.cuda.get_device_capability(0)}")

    print(f"\nloading {REPO} ...")
    policy = PI05Policy.from_pretrained(REPO)
    policy.to(dev).eval()
    cfg = policy.config
    nparam = sum(p.numel() for p in policy.parameters())
    print(f"  params        : {nparam/1e9:.2f} B")
    print(f"  chunk_size    : {cfg.chunk_size}  n_action_steps={cfg.n_action_steps}")
    print(f"  max_state_dim : {cfg.max_state_dim}  max_action_dim={cfg.max_action_dim}")
    print(f"  num_inference_steps: {cfg.num_inference_steps}  tokenizer_max_length={cfg.tokenizer_max_length}")
    print(f"  image_features: {list(cfg.image_features)}")
    print(f"  input  action : {cfg.input_features['observation.state'].shape}")
    print(f"  output action : {cfg.output_features['action'].shape}")

    # ---- Case A: as shipped --------------------------------------------------
    pre, post = make_pi05_pre_post_processors(config=cfg, dataset_stats=None)
    batch = synth_batch(cfg.input_features["observation.state"].shape[0])
    with torch.no_grad():
        chunk = policy.predict_action_chunk(pre(batch))
    report(f"A: {REPO} as shipped (padded 32-dim)", chunk)

    # ---- Case B: SO-101 6-DoF slice -----------------------------------------
    D = len(SO101_JOINTS)
    cfg.input_features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(D,))
    cfg.output_features["action"] = PolicyFeature(type=FeatureType.ACTION, shape=(D,))
    # Stand-in SO-101 stats; replace with real stats computed over your own dataset.
    stats = {
        "observation.state": {k: v for k, v in zip(
            ["mean", "std", "min", "max", "q01", "q99"],
            [torch.zeros(D), torch.ones(D), -torch.ones(D), torch.ones(D), -torch.ones(D), torch.ones(D)])},
        "action": {k: v for k, v in zip(
            ["mean", "std", "min", "max", "q01", "q99"],
            [torch.zeros(D), torch.ones(D), -torch.ones(D), torch.ones(D), -torch.ones(D), torch.ones(D)])},
    }
    pre_b, post_b = make_pi05_pre_post_processors(config=cfg, dataset_stats=stats)
    batch_b = synth_batch(D, seed=1)
    with torch.no_grad():
        chunk_b = policy.predict_action_chunk(pre_b(batch_b))
    report(f"B: SO-101 6-DoF slice ({'/'.join(SO101_JOINTS)})", chunk_b)
    print("\nNOTE: case-B values are NOT meaningful control signals. pi05_base has never")
    print("seen an SO-101; this only proves the shape/plumbing path end to end.")

    # ---- Case C: a real SO-101 fine-tune, with its own shipped norm stats ----
    so101_repo = os.environ.get("PI05_SO101_REPO",
                                "pdhinaka/pi05_so101_blue_cube_gray_box_cleaned_20k")
    try:
        del policy
        torch.cuda.empty_cache()
        print(f"\nloading SO-101 fine-tune {so101_repo} ...")
        p2 = PI05Policy.from_pretrained(so101_repo).to(dev).eval()
        c2 = p2.config
        print(f"  input  state : {c2.input_features['observation.state'].shape}")
        print(f"  output action: {c2.output_features['action'].shape}")
        print(f"  cameras      : {list(c2.image_features)}")
        pre_c = PolicyProcessorPipeline.from_pretrained(
            so101_repo, config_filename="policy_preprocessor.json")
        post_c = PolicyProcessorPipeline.from_pretrained(
            so101_repo, config_filename="policy_postprocessor.json")
        g = torch.Generator().manual_seed(2)
        bc = {k: torch.rand(1, 3, 480, 640, generator=g) for k in c2.image_features}
        bc["observation.state"] = torch.tensor([[0.0, -1.57, 1.57, 0.0, 0.0, 0.3]])
        bc["task"] = ["put the blue cube in the gray box"]
        with torch.no_grad():
            ch = p2.predict_action_chunk(pre_c(bc))
        report(f"C: {so101_repo} (real SO-101 fine-tune, 6-DoF)", ch)
        print("  joint order   : " + ", ".join(SO101_JOINTS))
    except Exception as e:
        print(f"\n--- C FAILED: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()

    print("\nSMOKE_TEST_OK")


if __name__ == "__main__":
    main()

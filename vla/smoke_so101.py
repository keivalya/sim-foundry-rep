"""pi0.5 inference smoke test with a REAL SO-101 fine-tune (6 DoF), synthetic images + state.

Checkpoint: pdhinaka/pi05_so101_blue_cube_gray_box_cleaned_20k, fine-tuned from
lerobot/pi05_base on an SO-101 dataset. Local copy has two edits so lerobot 0.6.1 can
load it, applied by make_so101_ckpt.py: six newer-lerobot config keys dropped, tokenizer_name
made an absolute path.
"""

import os
import numpy as np
import torch

CKPT = os.environ.get("SO101_CKPT", "/scratch/pandya.kei/simfoundry-repro/vla/ckpt_so101_pi05_lerobot061")
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def main():
    from lerobot.policies.pi05 import PI05Policy
    import lerobot.policies.pi05.processor_pi05  # noqa: F401  registers the pi05 state-tokenizer step
    from lerobot.processor import (
        PolicyProcessorPipeline,
        policy_action_to_transition,
        transition_to_policy_action,
    )

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__}  device={dev}")
    if dev == "cuda":
        print("gpu:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))

    policy = PI05Policy.from_pretrained(CKPT).to(dev).eval()
    cfg = policy.config
    print(f"\nconfig: chunk_size={cfg.chunk_size} n_action_steps={cfg.n_action_steps} "
          f"num_inference_steps={cfg.num_inference_steps} dtype={cfg.dtype}")
    print(f"  state   : {cfg.input_features['observation.state'].shape}")
    print(f"  action  : {cfg.output_features['action'].shape}")
    print(f"  cameras : {list(cfg.image_features)}")
    print(f"  relative actions: {cfg.use_relative_actions}  (absolute for {cfg.relative_exclude_joints})")
    print(f"  action_feature_names: {cfg.action_feature_names}")
    print(f"  params  : {sum(p.numel() for p in policy.parameters()) / 1e9:.2f} B")

    ov = {"device_processor": {"device": dev}}
    pre = PolicyProcessorPipeline.from_pretrained(
        CKPT, config_filename="policy_preprocessor.json", overrides=ov)
    post = PolicyProcessorPipeline.from_pretrained(
        CKPT, config_filename="policy_postprocessor.json", overrides=ov,
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
    # The absolute-action step needs the preprocessor's relative step to invert it.
    rel = next(s for s in pre.steps if type(s).__name__ == "RelativeActionsProcessorStep")
    next(s for s in post.steps if type(s).__name__ == "AbsoluteActionsProcessorStep").relative_step = rel

    # Synthetic observation. SO-101 joints are in LeRobot normalized units:
    # body joints [-100, 100], gripper [0, 100].
    g = torch.Generator().manual_seed(7)
    state = torch.tensor([[2.0, -35.0, 48.0, 25.0, -4.0, 18.0]])
    batch = {k: torch.rand(1, 3, 480, 640, generator=g) for k in cfg.image_features}
    batch["observation.state"] = state
    batch["task"] = ["put the blue cube in the gray box"]

    processed = pre(batch)
    prompt = processed["task"][0]
    print("\nprompt actually fed to the model:")
    print("  ", repr(prompt))
    bins = [int(x) for x in prompt.split("State: ")[1].split(";")[0].split()]
    oob = [JOINTS[i] for i, b in enumerate(bins) if not 0 <= b <= 255]
    print(f"  state bins {bins}" + (f"   <-- OUT OF RANGE: {oob}" if oob else "   (all in [0,255])"))
    print(f"  image tensors : {[tuple(processed[k].shape) for k in cfg.image_features]} -> resized to 224x224 in-model")
    print(f"  language tokens: {tuple(processed['observation.language.tokens'].shape)}")

    with torch.no_grad():
        chunk = policy.predict_action_chunk(processed)
    a = chunk.detach().float().cpu().numpy()
    print(f"\nRAW model output (normalized, relative): shape {tuple(a.shape)}  (batch, horizon, action_dim)")
    print(f"  dtype {a.dtype}  min/max/mean {a.min():.4f} / {a.max():.4f} / {a.mean():.4f}")
    print(f"  first step: {np.array2string(a[0, 0], precision=4)}")

    out = post(chunk)
    b = out.detach().float().cpu().numpy()
    print(f"\nPOST-PROCESSED chunk (absolute SO-101 joint targets): shape {tuple(b.shape)}")
    print("  joint order:", ", ".join(JOINTS))
    print(f"  state in    : {np.array2string(state.numpy()[0], precision=2)}")
    print(f"  per-dim min : {np.array2string(b[0].min(axis=0), precision=2)}")
    print(f"  per-dim max : {np.array2string(b[0].max(axis=0), precision=2)}")
    print(f"  first 5 steps:\n{np.array2string(b[0, :5], precision=2, suppress_small=True)}")
    print(f"  step 25     : {np.array2string(b[0, 25], precision=2)}")
    print(f"  last step   : {np.array2string(b[0, -1], precision=2)}")
    print("\nNOTE: images here are uniform noise, so these numbers are not a meaningful")
    print("trajectory. This test proves the shapes, units and plumbing only.")
    print("\nSO101_SMOKE_OK")


if __name__ == "__main__":
    main()

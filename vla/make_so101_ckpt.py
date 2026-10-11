"""Make pdhinaka/pi05_so101_blue_cube_gray_box_cleaned_20k loadable by lerobot 0.6.1.

Two edits, both inference-safe:
  1. config.json: drop 6 keys that only exist in lerobot >0.6.1. All are disabled in
     this checkpoint (use_visual_memory=False, use_proprioceptive_memory=False,
     rtc_training_max_delay=0), so dropping them changes nothing at inference.
  2. policy_preprocessor.json: tokenizer_name "tokenizer" (a repo-relative path that
     transformers resolves as the HF repo id "tokenizer") -> absolute local path.
Big files are symlinked, not copied.
"""
import glob, json, os

SRC = glob.glob("/scratch/pandya.kei/hf_cache/hub/"
                "models--pdhinaka--pi05_so101_blue_cube_gray_box_cleaned_20k/snapshots/*")[0]
DST = "/scratch/pandya.kei/simfoundry-repro/vla/ckpt_so101_pi05_lerobot061"
DROP = ["use_visual_memory", "use_proprioceptive_memory", "memory_frames",
        "memory_stride", "memory_temporal_attention_every", "rtc_training_max_delay"]

for root, _, files in os.walk(SRC):
    for fn in files:
        s = os.path.join(root, fn)
        d = os.path.join(DST, os.path.relpath(s, SRC))
        os.makedirs(os.path.dirname(d), exist_ok=True)
        if os.path.lexists(d):
            os.remove(d)
        if fn == "config.json":
            c = json.load(open(s))
            print("dropping:", [k for k in DROP if k in c])
            for k in DROP:
                c.pop(k, None)
            json.dump(c, open(d, "w"), indent=4)
        elif fn == "policy_preprocessor.json":
            c = json.load(open(s))
            for st in c["steps"]:
                if st["registry_name"] == "tokenizer_processor":
                    st["config"]["tokenizer_name"] = os.path.join(DST, "tokenizer")
                    print("tokenizer_name ->", st["config"]["tokenizer_name"])
            json.dump(c, open(d, "w"), indent=4)
        else:
            os.symlink(os.path.realpath(s), d)
print("ok ->", DST)

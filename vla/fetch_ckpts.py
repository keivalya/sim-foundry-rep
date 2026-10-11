import os
os.environ.setdefault("HF_HOME", "/scratch/pandya.kei/hf_cache")
from huggingface_hub import snapshot_download
tok = open("/scratch/pandya.kei/hf_cache/token").read().strip()
p = snapshot_download("google/paligemma-3b-pt-224", token=tok,
    allow_patterns=["tokenizer*", "special_tokens_map.json", "added_tokens.json", "preprocessor_config.json"])
print("paligemma tokenizer ->", p)
p = snapshot_download("lerobot/pi05_base", token=tok)
print("pi05_base ->", p)
print("FETCH_OK")

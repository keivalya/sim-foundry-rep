import os
os.environ.setdefault("HF_HOME", "/scratch/pandya.kei/hf_cache")
from huggingface_hub import snapshot_download
tok = open("/scratch/pandya.kei/hf_cache/token").read().strip()
print(snapshot_download("pdhinaka/pi05_so101_blue_cube_gray_box_cleaned_20k", token=tok))
print("FETCH_OK")

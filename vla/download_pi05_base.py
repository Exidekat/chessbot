#!/usr/bin/env python3
"""Download π₀.₅ base checkpoint from HuggingFace."""

import os
from huggingface_hub import snapshot_download

# Set cache directory
cache_dir = os.path.expanduser("~/.cache/openpi")
os.makedirs(cache_dir, exist_ok=True)

print("Downloading π₀.₅ base checkpoint from HuggingFace...")
print(f"Cache directory: {cache_dir}")

# Download the checkpoint
local_dir = snapshot_download(
    repo_id="lerobot/pi05_base",
    cache_dir=cache_dir,
    local_dir=os.path.join(cache_dir, "lerobot/pi05_base"),
    local_dir_use_symlinks=False,
    resume_download=True
)

print(f"\n[OK] Downloaded to: {local_dir}")

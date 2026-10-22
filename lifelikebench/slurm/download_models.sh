#!/bin/bash
# Run ONCE on a login node (it has internet; compute jobs run offline):
#   bash slurm/download_models.sh configs/local_surf.yaml
# Downloads every model in the config's `models:` registry into $HF_HOME, skipping ones
# already cached (e.g. by Character_Benchmark). Gated repos need HF_TOKEN + accepted licence.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/surf.env"
activate_env
export HF_HUB_OFFLINE=0
CONFIG=${1:-configs/local_surf.yaml}
cd "$REPO_DIR"
PYTHONPATH=src python - "$CONFIG" <<'PY'
import sys
from huggingface_hub import snapshot_download
from lifelike_bench.cli import load_config
cfg = load_config(sys.argv[1])
for name, m in cfg["models"].items():
    path = m["path"]
    if path.startswith("/"):
        print(f"{name}: local path {path}, skipping"); continue
    print(f"{name}: {path}", flush=True)
    local = snapshot_download(path, ignore_patterns=["original/*", "*.pth", "*.gguf"])
    import pathlib
    w = sum(f.stat().st_size for f in pathlib.Path(local).glob("*.safetensors"))
    print(f"  weights: {w/1e9:.1f} GB", flush=True)
    if w < 1e9:
        sys.exit(f"ERROR: {name} has no weights after download — see messages above")
print("all models cached in", __import__("os").environ["HF_HOME"])
PY

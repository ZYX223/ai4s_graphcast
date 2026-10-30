"""Verify the immutable 2026-10-30 source/evidence archive without importing JAX."""

import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "evidence/20261030/SOURCE_MANIFEST.json").read_text())
seen = set()
for row in manifest["files"]:
    path = (root / row["path"]).resolve()
    if root not in path.parents or row["path"] in seen:
        raise ValueError("Unsafe or duplicate archive path: " + row["path"])
    seen.add(row["path"])
    data = path.read_bytes()
    if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
        raise ValueError("Source archive differs: " + row["path"])
print(f"Verified {len(seen)} archived source/evidence files; no GPU execution.")

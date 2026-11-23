"""Fetch just the eight public PC-CRC data files, verifying size and SHA-256."""
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent

def valid(path, spec):
    if not path.is_file() or path.stat().st_size != spec["bytes"]:
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == spec["sha256"]

def main():
    manifest = json.loads((ROOT / "data-files.json").read_text())
    directory = ROOT / "data"
    directory.mkdir(exist_ok=True)
    missing = [name for name, spec in manifest.items() if not valid(directory / name, spec)]
    if not missing:
        print(f"Verified {len(manifest)} local data files.")
        return
    source = json.loads((ROOT / "release.json").read_text())["dataset"]
    if not re.fullmatch(r"[0-9a-f]{40}", source["revision"] or ""):
        raise ValueError("A public dataset commit is required to download missing files.")
    base = f"https://huggingface.co/datasets/{source['id']}/resolve/{source['revision']}/data/"
    for name in missing:
        if name != Path(name).name:
            raise ValueError(f"Unexpected manifest entry: {name}")
        with tempfile.TemporaryDirectory(dir=directory) as tmp:
            dest = Path(tmp) / name
            request = Request(base + name, headers={"User-Agent": "pc-crc-replay/1.0"})
            with urlopen(request, timeout=60) as response, dest.open("wb") as output:
                shutil.copyfileobj(response, output)
            if not valid(dest, manifest[name]):
                raise ValueError(f"Downloaded file failed integrity checks: {name}")
            os.replace(dest, directory / name)
    print(f"Fetched {len(missing)} files and verified all {len(manifest)} data files.")

if __name__ == "__main__":
    main()

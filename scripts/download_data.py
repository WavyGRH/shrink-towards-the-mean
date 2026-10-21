"""Download the School data (MALSAR release) and check its SHA-256.

The file is distributed with MALSAR under the GPL-2.0, so it is fetched rather than redistributed here.

    python scripts/download_data.py
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mtlpool.config import SCHOOL_SHA256, SCHOOL_URL  # noqa: E402

DEST = ROOT / "data" / "school.mat"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    DEST.parent.mkdir(exist_ok=True)
    if DEST.exists() and sha256(DEST) == SCHOOL_SHA256:
        print(f"{DEST} already present and verified")
        return
    print(f"downloading {SCHOOL_URL}")
    urllib.request.urlretrieve(SCHOOL_URL, DEST)
    digest = sha256(DEST)
    if digest != SCHOOL_SHA256:
        DEST.unlink()
        sys.exit(f"checksum mismatch: got {digest}, expected {SCHOOL_SHA256}")
    print(f"saved {DEST} ({DEST.stat().st_size / 1e6:.1f} MB), SHA-256 verified")


if __name__ == "__main__":
    main()

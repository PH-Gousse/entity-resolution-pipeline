"""Download Amazon-Google benchmark data from DeepMatcher repository."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import requests

# Hosted by the Anhai group at UW-Madison (DeepMatcher dataset collection).
_BASE_URL = (
    "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/"
    "Structured/Amazon-Google/exp_data"
)

_FILES: dict[str, str] = {
    "tableA.csv": "04a797511d83af9e27469778081c50397857b199d49e2f83c119390f14301947",
    "tableB.csv": "00c0d19c0dcd014b00d6d341e15e2b3b3caa616b4f6d710d0c9cd2dfc3f4717f",
    "test.csv": "09e376c0c5391bd4a6c6d8efde148c994441b6043ef6b7f9eab2a8772a6fb362",
    "train.csv": "cdc5e88d714e77febc26a81a61c3295a0000baa51d6debf93043f0cbd5193b9f",
    "valid.csv": "4999067ea47fa2ae03e0c4a68fee60cf68728943208fcde46415cb895844dd22",
}

_DEST_DIR = Path("data/amazon-google")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download(dest_dir: Path = _DEST_DIR, force: bool = False) -> None:
    """Download all data files, verifying SHA-256 checksums."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    for filename, expected_sha in _FILES.items():
        dest = dest_dir / filename

        if dest.exists() and not force:
            actual = _sha256(dest)
            if actual == expected_sha:
                print(f"  {filename}: already present, checksum OK")
                continue
            print(f"  {filename}: checksum mismatch, re-downloading")

        url = f"{_BASE_URL}/{filename}"
        print(f"  {filename}: downloading from {url}")

        for attempt in range(3):
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                break
            except (requests.ConnectionError, requests.Timeout) as e:
                if attempt == 2:
                    raise ConnectionError(
                        f"Failed to download {url} after 3 attempts: {e}"
                    ) from e
                print(f"    retry {attempt + 1}/2...")

        dest.write_bytes(resp.content)

        actual = _sha256(dest)
        if actual != expected_sha:
            dest.unlink()
            raise ValueError(
                f"Checksum mismatch for {filename}: "
                f"expected {expected_sha[:12]}..., got {actual[:12]}..."
            )
        print(f"  {filename}: OK ({len(resp.content):,} bytes)")


def main() -> int:
    print("Downloading Amazon-Google dataset...")
    try:
        download()
    except (ConnectionError, ValueError, requests.HTTPError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Check that a corpus is byte-identical to the reference one.

    python scripts/verify_corpus.py data/encoded

The reference corpus was built on 5 November 2026 from the September 2026
Lichess dump. Monthly dumps are immutable and the pipeline is deterministic,
so rebuilding it anywhere should reproduce these files exactly. If it does
not, results trained on the two copies are not comparable.
"""

import hashlib
import sys
from pathlib import Path

REFERENCE_SHA256 = {
    "positions.bin": "2c6351bad225054ab9a8cbc4a6abfd64099e561a2e42c739da47f54a7fa4a9b3",
    "moves.bin": "1912e8d63fe78b7609b415b0239b61d3ae056b7adf98127a21027c3dba4ff664",
    "split_labels.npy": "9366635b33b234880120a5f5b01ef8bb637467ff8ecfc3cd5f3f30e920f885ca",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 22):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    corpus = Path(sys.argv[1] if len(sys.argv) > 1 else "data/encoded")
    failed = False
    for name, expected in REFERENCE_SHA256.items():
        path = corpus / name
        if not path.exists():
            print(f"MISSING   {name}")
            failed = True
            continue
        actual = sha256(path)
        ok = actual == expected
        failed |= not ok
        print(f"{'ok       ' if ok else 'DIFFERENT'} {name}  {actual[:16]}")

    if failed:
        raise SystemExit("\nThis is not the reference corpus. Do not compare results across it.")
    print("\nIdentical to the reference corpus.")


if __name__ == "__main__":
    main()

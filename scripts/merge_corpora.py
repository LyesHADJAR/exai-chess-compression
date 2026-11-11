"""
Join several extracted corpora into one.

    python scripts/merge_corpora.py --out data/v2 data/encoded data/v2_parts/2026-08 ...

Each input is the output of `extract.py`. Positions and moves are concatenated
in the order given, and each game's offset is shifted to where its plies now
sit. Split labels are not carried over: run `build_splits.py` on the result.
That is safe for the reference corpus too, since a game's split depends only
on its ID, so the games it already placed keep their placement.

Refuses to merge corpora that share a game, since the same game in two places
would sit in two different splits at once.
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chessdt.encoding import BOARD_RECORD_BYTES  # noqa: E402

GAME_FIELDS = ("game_id", "length", "result", "white_elo", "black_elo")


def merge(parts: list[Path], out: Path) -> dict:
    """Merge the corpora in `parts` into `out`. Returns the totals."""
    out.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, list[np.ndarray]] = {name: [] for name in GAME_FIELDS}
    offsets, shift = [], 0

    for part in parts:
        meta = np.load(part / "games.npz")
        lengths = meta["length"].astype(np.int64)
        total = int(lengths.sum())
        if (part / "positions.bin").stat().st_size != total * BOARD_RECORD_BYTES:
            raise ValueError(f"{part} is inconsistent: positions.bin does not match games.npz")
        offsets.append(meta["offset"].astype(np.int64) + shift)
        for name in GAME_FIELDS:
            arrays[name].append(meta[name])
        shift += total

    ids = np.concatenate(arrays["game_id"])
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"{len(ids) - len(np.unique(ids)):,} games appear in more than one corpus")

    for name in ("positions.bin", "moves.bin"):
        with (out / name).open("wb") as target:
            for part in parts:
                with (part / name).open("rb") as source:
                    shutil.copyfileobj(source, target, length=1 << 24)

    np.savez(
        out / "games.npz",
        offset=np.concatenate(offsets),
        record_bytes=np.int32(BOARD_RECORD_BYTES),
        **{name: np.concatenate(arrays[name]) for name in GAME_FIELDS},
    )
    return {"games": len(ids), "positions": shift}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("parts", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    totals = merge(args.parts, args.out)
    print(f"Merged {len(args.parts)} corpora into {args.out}/")
    print(f"  {totals['games']:,} games, {totals['positions']:,} positions")
    print("Next: python scripts/build_splits.py --corpus", args.out)


if __name__ == "__main__":
    main()

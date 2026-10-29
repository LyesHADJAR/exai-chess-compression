# Data

Nothing here is in git. Everything here is regenerable from a command.

## Where it lives

Google Drive: **EXAI Chess Compression — Data**
<https://drive.google.com/drive/folders/1pdqyPK86jqEDl7JsqgSXywWNrUfNaX4u>

| Folder | Contents |
|---|---|
| `01-raw` | Lichess source files as downloaded, unmodified |
| `02-encoded` | positions encoded once to packed arrays, plus the per-game index |
| `03-splits` | train, val, agreement and puzzle sets after deduplication |
| `04-checkpoints` | trained reference models and their compressed variants |

## Sources

All from Lichess, all CC0.

| Source | Figure, checked 2026-10-29 |
|---|---|
| Monthly standard rated dump, September 2026 | 29.2 GB compressed, 89,616,462 games |
| Puzzle database | 6,157,341 puzzles |
| Evaluation database | 416,442,401 Stockfish-evaluated positions |

The game count above is unfiltered — every rating and time control, bullet
included. The share surviving our filters has not been measured yet, and it is
what sets how many bytes we stream.

## Collection

Stream the monthly dump through zstd, parse as it arrives, stop at the target.
The 29.2 GB is never written to disk. Filters: both players rated 1800 or
above, initial clock 180 seconds or more, at least 20 plies, normal termination.

Target is 300,000 games, roughly 24M positions.

## Splitting

Two guards, both necessary:

1. **Split by game**, never by position. Consecutive positions differ by one
   piece; splitting at position level puts near-identical boards on both sides.
2. **Drop positions whose hash appears in train.** Openings repeat across
   games, so without this a large share of any agreement score is memorised
   book lines.

The position hash covers piece placement, side to move, castling rights and the
en passant target. It **excludes the halfmove and fullmove counters** — the same
position reached at a different move number is the same position, and leaving
the counters in defeats the deduplication entirely.

Split assignment hashes the game ID rather than storing a split file, so it is
deterministic and reproducible from the dump alone.

The puzzle deduplication is mandatory, not a precaution: Lichess puzzles are
mined from Lichess games, so overlap is likely by construction.

## Storage format

Positions are encoded once and never parsed again. Packed at 4 bits per square
this is about 770 MB for 24M positions, expanded to 77 tokens in the collate
function. Alongside it sits a per-game index of start offsets, which is what
makes slicing K-ply windows cheap.

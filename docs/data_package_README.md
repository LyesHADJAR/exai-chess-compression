# Training and evaluation data for the chess Decision Transformer

This is the exact data the reference model (v2, 17.6M parameters, 200,000 steps)
was trained and evaluated on. Use it to train your own model and to compare it
with ours on identical inputs. Everything comes from the public Lichess
database (CC0). The files contain game IDs and ratings, no player names.

## Contents (about 6.0 GB unpacked)

| Folder | Size | What it is |
|---|---|---|
| `v2/` | 4.78 GB | Training corpus: 1,700,000 games, 121,217,659 positions |
| `v2_tactics/mate_in_one.npz` | 12 MB | Positions with a mate in one, used for the 25% tactical emphasis |
| `puzzles_v2/` | 297 MB | Lichess puzzles, deduplicated against the corpus (6,177,968 puzzles) |
| `encoded/` | 0.9 GB | The v1 corpus (300,000 games). Its test split is the common benchmark |
| `tactics_all/mate_in_one.npz` | 1.5 MB | Every mate-in-one position of the v1 corpus; the mate probe scores its 3,769 test positions |
| `MANIFEST_v2.sha256` | | Checksum of every file above |

## Unpack and verify

The package is split into parts of 500 MB. Put all parts in one folder, then:

```bash
cat exai_chess_data.tar.part_* | tar -xf -      # creates v2/ v2_tactics/ puzzles_v2/ encoded/ tactics_all/
sha256sum -c MANIFEST_v2.sha256                  # every line must say OK
```

Put the folders under `data/` of the repository
(https://github.com/LyesHADJAR/exai-chess-compression). On macOS use
`shasum -a 256 -c MANIFEST_v2.sha256`.

## File format

Each corpus folder (`v2/`, `encoded/`) holds five files.

| File | Content |
|---|---|
| `positions.bin` | One 36-byte record per position, in game order. Bytes 0 to 31: the 64 squares at 4 bits each (a1 is the low nibble of byte 0; 0 empty, 1 to 6 white P N B R Q K, 7 to 12 black p n b r q k). Byte 32: bit 0 side to move (1 = white), bits 1 to 4 castling KQkq. Byte 33: en passant file or 255. Byte 34: halfmove clock. Byte 35: reserved |
| `moves.bin` | One uint16 per position: the move the player made, as an index into the 1,968-move UCI vocabulary (every legal UCI move, sorted lexicographically). Built by `chessdt.encoding.MOVE_VOCABULARY` |
| `games.npz` | Per game: `game_id`, `offset` and `length` (first position and number of positions), `result` (from white's side: 1 white wins, 0 draw, -1 black wins), `white_elo`, `black_elo` |
| `game_splits.npy` | Per game: 0 train, 1 validation, 2 test. Assigned by hashing the game ID |
| `split_labels.npy` | Per position: 0 train, 1 validation, 2 test, 3 dropped |

Label 3 marks held-out positions that also occur in training (duplicates of
opening lines and the like). They are still real positions and serve as history,
but they are never scored. Never train on labels 1, 2 or 3 as targets.

`puzzles_v2/` has `positions.bin` (same 36-byte records), `moves.bin` (the one
solution move per puzzle, uint16) and `puzzles.npz` (rating, themes, and so on).
The puzzle set cannot be rebuilt identically, because Lichess keeps adding
puzzles, so always use this copy.

## How we trained the reference

```bash
env PY=python DATA=data/v2 TACTICS=data/v2_tactics PUZZLES=data/puzzles_v2 \
    STEPS=200000 EVAL_EVERY=5000 PUZZLE_SAMPLE=50000 \
    bash scripts/run_arm.sh my_run --preset m18 \
    --tactics data/v2_tactics/mate_in_one.npz --tactical-fraction 0.25
```

AdamW (betas 0.9 and 0.95, weight decay 0.01 on weight matrices), peak rate 4e-4,
1,000 warmup steps, cosine decay to 5%, batch of 256 windows of 16 plies, dropout
0.1, gradient clipping 1.0. About 8 hours on one H100 slice.

## How to compare with us

Score every model on the same benchmark: the v1 test split, the 50,000 sampled
puzzles, and the 3,769 mate-in-one positions.

```bash
python scripts/evaluate.py YOUR/model_best.pt --data data/encoded --puzzles data/puzzles_v2 \
    --puzzle-sample 50000 --windows 6000 --ablation-windows 500 --output evaluation_common.json
python scripts/probe_tactics.py YOUR/model_best.pt --data data/encoded --set data/tactics_all/mate_in_one.npz
```

Our numbers on it (reference v2, 17.6M parameters):

| Measure | Reference v2 | Previous model v1 (11.3M) |
|---|---|---|
| Test top-1 / top-3 | 50.98 / 78.38 | 44.19 / 71.64 |
| Puzzles solved (illegal moves masked) | 57.16 | 31.49 |
| Mate in one played in real games (humans 73.5) | 69.4 | 33.2 |

Run-to-run noise between two seeds is about 0.4 to 0.5 points of top-1, 0.8 to
0.9 of puzzles and 0.6 to 0.7 of mate in one, so gaps below that mean nothing.

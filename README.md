# EXAI Chess Compression Challenge

A data-free model compression challenge built on a chess Decision Transformer.

Competitors receive the weights of a reference policy and nothing else. They
shrink the file as far as they can while the model keeps playing. No training
data, no engine, no distillation from anything but the released model itself.

## The task the model solves

Given a board position, predict the move. The model is framed as a Decision
Transformer over the standard interleaved stream `[R1, s1, a1, R2, s2, a2, ...]`,
so it carries K plies of history and a conditioning channel alongside the
current position.

| | |
|---|---|
| Input | 64 square tokens per position, pooled to one vector per ply |
| Conditioning | outcome-to-go and rating of the player to move |
| Output | softmax over 1,968 UCI moves, illegal moves masked |
| Context | 16 plies |
| Parameters | about 17.6 million (`m18` preset) |
| Released artifact | 70,549,562 bytes (67.3 MiB), fp32 |
| Training data | Lichess, human games, behavioural cloning |

The parameter count is deliberately tight. A model with slack in it is trivial
to compress, which makes for a challenge nobody can win on skill.

## Repository layout

| Path | Holds |
|---|---|
| `src/chessdt/` | the library: encoding, data, model, evaluation |
| `scripts/` | entry points — collection, training, evaluation, and the gates |
| `tests/` | correctness tests, including the readout-leak guard |
| `docs/` | where the data lives, and how to train |

## Status

The reference model is **frozen**: `v3_m18_200k`, 17.6M parameters, trained for
200,000 steps on the v2 corpus (1.7M games, 121M positions) with 25% of the
windows built around a mate in one. Earlier models are kept as fallbacks (`v2_m18`
at 100,000 steps, then the 11.3M v1).

On the common benchmark (the v1 test split, 71,954 positions, plus 50,000 sampled
puzzles):

| Measure | v1 (11.3M) | Frozen reference |
|---|---|---|
| Test top-1 / top-3 | 44.19 / 71.64 | 50.98 / 78.38 |
| Puzzles solved, illegal moves masked | 31.49 | 57.16 |
| Mate in one played in real games (humans 73.5) | 33.2 | 69.4 |

The input ablations and the compressibility check were run on the frozen model:
emptying the board changes 99.7% of predictions, and the standard methods separate
widely (int8 keeps 98.1% agreement with the reference, int4 71.1%, int3 38.1%,
30% magnitude pruning 73.5%, low rank under 3%). The scoring harness and the
submission format are still a draft. Training data for teammates is described in
[docs/data_package_README.md](docs/data_package_README.md); a four-page summary is in
[docs/reports/team_report.pdf](docs/reports/team_report.pdf).

## Getting started

```bash
pip install -e ".[dev]"
pytest -q
```

Data is not in this repository and never will be. See [docs/data.md](docs/data.md)
for where it lives and how to regenerate it, and [docs/training.md](docs/training.md)
for training and evaluating the reference model.

## Validation gates

Three checks run in order, each before the step it protects. They exist because
the previous iteration of this project shipped a model that had learned to copy
its own input, and nobody checked until late.

1. **Linear probe** (`scripts/probe_linear.py`) — a single linear layer from
   the one-hot board to the move must fall far short of the model. If a linear
   model can do the task, the task is not worth compressing.
2. **Input ablation** (`scripts/evaluate.py`) — emptying the board must change
   most predictions. Any channel that can be damaged without effect is a dead
   channel, and may not be claimed in the write-up.
3. **Separation check** — standard compression methods must land at visibly
   different fidelities, reported per parameter group. If everything ties, the
   scoring cannot rank anyone.

## Licence

Code is MIT. Lichess data is CC0. See [LICENSE](LICENSE).

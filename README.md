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
| Parameters | 11,284,656 |
| Released artifact | about 43 MB, fp32 |
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

The data pipeline and the corpus are done: 300,000 games, 22.7M positions,
split and deduplicated. The model, the training loop and the evaluation are
written and tested. The reference model has not been trained yet.

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

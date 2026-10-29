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
| Input | FEN flattened to 77 tokens, one position per ply |
| Output | softmax over 1,968 UCI moves, illegal moves masked |
| Context | 16 plies |
| Parameters | about 11.3M |
| Released artifact | about 43 MB, fp32 |
| Training data | Lichess, human games, behavioural cloning |

The parameter count is deliberately tight. A model with slack in it is trivial
to compress, which makes for a challenge nobody can win on skill.

## Repository layout

| Path | Holds |
|---|---|
| `src/chessdt/` | the library: encoding, model, data, checkpoints |
| `scripts/` | entry points — collection, training, and the three gates |
| `tests/` | correctness tests, including the readout-leak guard |
| `docs/` | data locations and design notes |

## Status

Design is settled; implementation has not started. The open blocker is GPU
access for training the reference model.

## Getting started

```bash
pip install -e ".[dev]"
pytest -q
```

Data is not in this repository and never will be. See [docs/data.md](docs/data.md)
for where it lives and how to regenerate it.

## Validation gates

Three checks run in order, each before the step it protects. They exist because
the previous iteration of this project shipped a model that had learned to copy
its own input, and nobody checked until late.

1. **Linear probe** — a ridge regression from the encoded position to the move
   must fail badly. If a linear model can do the task, the task is not worth
   compressing.
2. **Input ablation** — zeroing the board must destroy the prediction. Any
   channel that can be zeroed without effect is a dead channel.
3. **Separation check** — standard compression methods must land at visibly
   different fidelities, reported per parameter group. If everything ties, the
   scoring cannot rank anyone.

## Licence

Code is MIT. Lichess data is CC0. See [LICENSE](LICENSE).

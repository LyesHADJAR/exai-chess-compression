# Training the reference model

Everything below runs from a terminal on the machine with the GPU. In
JupyterLab that is **File → New → Terminal**.

## 1. Set up

```bash
git clone https://github.com/LyesHADJAR/exai-chess-compression.git
cd exai-chess-compression
pip install -e ".[dev]"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name())"
```

The last line must print `True` and a GPU name. If it prints `False`, the
installed PyTorch has no CUDA support and training would silently run on the
CPU; fix that before going further.

## 2. Get the corpus

Rebuild it in place. The Lichess monthly dump never changes and the pipeline
is deterministic, so this reproduces the reference corpus byte for byte, and
the third command proves it.

```bash
python scripts/extract.py --games 300000 --out data/encoded     # about 30 min
python scripts/build_splits.py --corpus data/encoded            # under a minute
python scripts/verify_corpus.py data/encoded                    # must say "Identical"
```

Do not train on a corpus that fails the check: results would not be comparable
with anyone else's.

The puzzle set is the one thing that cannot be rebuilt identically, because
Lichess keeps adding puzzles to its database. Copy `03-splits/puzzles/` from
the Drive folder into `data/puzzles/`. It is only needed for step 5.

## 3. Check the task is worth the trouble

```bash
python scripts/probe_linear.py
```

Two minutes. It fits a single linear layer to the same data and prints its
accuracy. The transformer has to beat that number by a wide margin; write it
down, it goes in the results.

## 4. Train

```bash
nohup python scripts/train.py --hours 6 --out runs/reference > train.log 2>&1 &
tail -f train.log
```

`nohup` and the trailing `&` keep it running after the browser tab is closed.
`tail -f` only watches; Ctrl-C stops the watching, not the training.

`--hours` is a budget, not a guess. The script times its first hundred steps,
works out how many steps fit, and shapes the learning-rate schedule to end
exactly there. If the run is interrupted, the same command picks up from the
last checkpoint.

A healthy log looks like this, with the loss falling and accuracy climbing:

```
Calibrated: 210 ms/step, so 6 h is 102,857 steps, 20.6 passes over the data
step     100  loss 6.3621  top1 0.0230  lr 4.00e-05   19,500 pos/s    359 min left
step   2,000  loss 3.1  top1 0.24 ...
          val loss 3.05  val top1 0.25  (best so far)
```

Stop and ask if, after the first evaluation, validation accuracy is still
under 5%, or the loss prints `nan`.

## 5. Evaluate

```bash
python scripts/evaluate.py runs/reference/model_best.pt --puzzles data/puzzles
```

Prints accuracy, the ablation table and the puzzle solve rate, and writes
`runs/reference/evaluation.json`.

## 6. What to bring back

Four small files and one checkpoint, all in `runs/reference/`:

| File | What it is |
|---|---|
| `results.json` | final accuracy on the full validation and test sets |
| `evaluation.json` | ablations and puzzles |
| `metrics.jsonl` | the whole training curve |
| `config.json` | exactly what was run, on which GPU |
| `model_best.pt` | the weights, about 43 MB |

Put them in `04-checkpoints/<run name>/` in the Drive folder. The checkpoint
never goes into git.

## Reading the results

- **Validation top-1** is the headline. Strong human-move predictors reach the
  low-to-mid 50s with far more capacity and data; this model is deliberately
  small, so expect less. What matters is the gap to the linear probe.
- **Validation top-3** says how well the behaviour is cloned. A human in most
  positions chooses between a few reasonable moves, so the played move being
  among the model's three favourites is the fairer test of whether it has
  understood the position. It is reported everywhere top-1 is.
- **The ablation table is the release gate.** Emptying the board must change
  most predictions. If `outcome`, `rating` or `history` come back as inert,
  the model still works, but the write-up may not describe it as conditioned
  on that channel.
- **Puzzles** are the sharp number the compression scoring is anchored to.
  Whatever this model scores becomes the bar a compressed model is held to.

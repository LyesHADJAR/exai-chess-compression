# Experiments for v2

The reference model (v1) finds a mate in one about a third as often as the
humans it imitates. v2 is a set of changes aimed at that, tested one at a time
so we know which ones help before spending a long run on all of them.

## The rule, fixed before anything is run

Keep a change if it improves **mate in one** or the **held-out puzzles** by at
least **2 points**, without costing more than **0.5 points of top-1**.

Two copies of the baseline run with different seeds. The gap between them is how
far a result moves by chance. If it is more than half the 2-point threshold, the
table says so and small gaps should not be trusted.

## The arms

| Arm | What it changes | Tests |
|---|---|---|
| `baseline`, `baseline_seed1` | nothing | the reference, twice, to measure chance |
| `pooled8` | 8 attention-pooled vectors per ply instead of 1 | whether one vector is a bottleneck |
| `geometry` | file, rank and diagonal embeddings, relative attention bias | whether board geometry helps tactics |
| `tactical` | half of all windows are built around a mate-in-one position | whether showing mates more often helps |
| `conddrop` | the conditioning is hidden 10% of the time | whether it makes the conditioning matter |
| `pooled4` | 4 pooled vectors per ply | the middle of the bottleneck question |
| `m18`, `m25` | 18M and 25M parameters (100 MiB cap) | whether size buys tactics |
| `fulldetail` | every square is its own token in the backbone | the strongest test of the bottleneck idea; about 3.6x slower |

Every arm trains for the same number of steps on the same data, so they are
compared at equal data, and the table reports minutes as well.

## Running it on the cluster

Once per machine, from the repository root:

```bash
export PYTHONPATH=$PWD/.deps
PY=~/envs/rfdetr/bin/python

# every mate in one in the corpus (about 7 minutes)
$PY scripts/build_tactics_set.py --splits train val test --out data/tactics_all

# the puzzle set, deduplicated against this corpus (about 5 minutes)
$PY scripts/build_puzzles.py --corpus data/encoded --out data/puzzles
```

Then the arms. The first group is the cheap, informative one:

```bash
PY=$PY nohup bash scripts/run_wave.sh core > wave_core.log 2>&1 &
tail -f wave_core.log
```

It runs two arms at a time, writes each to `runs/arms/NAME/`, and prints the
comparison table at the end. Rerunning skips arms that already finished.
`bash scripts/run_wave.sh extra` runs the larger ones.

To see the table again at any point:

```bash
$PY scripts/compare_runs.py runs/arms/*/ --baseline runs/arms/baseline
```

## Reading the table

| Column | Meaning |
|---|---|
| `tok` | tokens in one window; the cost driver |
| `min` | training minutes |
| `top-1`, `top-3` | held-out accuracy |
| `puzzle` | held-out puzzles solved, illegal moves masked |
| `mate`, `no hist` | mate in one found in real games, with and without history |
| `steer` | widest gap between five differently conditioned players on puzzles |
| `board` | share of predictions that change when the board is emptied; must stay near 100 |

## If an arm is slow

`fulldetail` prints its speed in the first minute of its log. If it runs at more
than about 600 ms/step, stop it: the attention path is not fast on that machine
and the arm will not finish in useful time.

"""
Reading the Lichess puzzle database.

Puzzles are the sharp half of the fidelity gate. Agreement on ordinary
positions is forgiving — most positions in a real game have an obvious move,
so a tactically broken model still agrees on the bulk of them. A puzzle has
exactly one move that works, so losing the tactic shows up immediately.

The CSV's own convention is the thing to get right: the stored FEN is the
position *before* the opponent's mistake, and the first entry in `Moves` is
that mistake. The position the solver actually faces is the FEN with that
first move applied, and the move we score against is the second entry.
Encoding the FEN as-is would silently build a gate that tests the wrong
position every time.
"""

from dataclasses import dataclass

import chess

from .encoding import MOVE_TO_INDEX

#: Columns we rely on. The file carries more, and has gained columns over
#: time, so rows are read by name rather than by position.
REQUIRED_COLUMNS = ("PuzzleId", "FEN", "Moves", "Rating", "Themes")

#: Themes are stored fixed-width; anything longer is truncated and counted.
THEME_BYTES = 128


@dataclass(frozen=True)
class Puzzle:
    """One puzzle, resolved to the position the solver is shown."""

    puzzle_id: str
    board: chess.Board
    solution: str
    rating: int
    themes: str
    line_plies: int

    @property
    def solution_index(self) -> int:
        return MOVE_TO_INDEX[self.solution]


def parse_puzzle(row: dict) -> Puzzle | None:
    """Turn a CSV row into a Puzzle, or None if it cannot be trusted.

    Rejects anything malformed rather than guessing: a puzzle that does not
    parse cleanly is one puzzle out of six million, and a wrong one would
    quietly corrupt the gate.
    """
    moves = row.get("Moves", "").split()
    if len(moves) < 2:
        return None

    try:
        board = chess.Board(row["FEN"])
    except ValueError:
        return None

    try:
        setup = chess.Move.from_uci(moves[0])
        solution = chess.Move.from_uci(moves[1])
    except ValueError:
        return None

    # The opponent's mistake is applied first; what remains is the puzzle.
    if setup not in board.legal_moves:
        return None
    board.push(setup)

    if solution not in board.legal_moves:
        return None
    if solution.uci() not in MOVE_TO_INDEX:
        return None

    try:
        rating = int(row["Rating"])
    except (KeyError, ValueError):
        return None

    return Puzzle(
        puzzle_id=row["PuzzleId"],
        board=board,
        solution=solution.uci(),
        rating=rating,
        themes=row.get("Themes", ""),
        line_plies=len(moves),
    )

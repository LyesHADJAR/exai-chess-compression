"""
Tactical probes on ordinary game positions.

Puzzles measure tactics on positions chosen to contain one. These measure the
same thing where it matters more: on the positions a model actually meets in
games, how often does it find the tactic when one is there, compared with the
humans it was trained to imitate?

Mate in one is the cleanest case. It is unambiguous, it can be found by brute
force, and a human who is playing to win nearly always takes it, so the human
rate is a ceiling the model can be held against.
"""

import chess
import numpy as np

from .encoding import BOARD_RECORD_BYTES, MOVE_TO_INDEX, unpack_board


def mating_moves(board: chess.Board) -> set[str]:
    """Every move that checkmates immediately, as UCI strings.

    Only moves that give check can mate, so only those are tried.
    """
    mates = set()
    for move in board.legal_moves:
        if board.gives_check(move):
            board.push(move)
            if board.is_checkmate():
                mates.add(move.uci())
            board.pop()
    return mates


def scan_mates(records: bytes) -> list[tuple[int, list[int]]]:
    """Find the positions in a block of packed records that have a mate in one.

    Takes the raw bytes of consecutive records, so it can be handed to a worker
    process cheaply, and returns (row, mating move indices) for every row that
    has one. Rows with no mate are not listed.
    """
    block = np.frombuffer(records, dtype=np.uint8).reshape(-1, BOARD_RECORD_BYTES)
    found = []
    for row, record in enumerate(block):
        mates = mating_moves(unpack_board(record.tobytes()))
        if mates:
            found.append((row, sorted(MOVE_TO_INDEX[m] for m in mates)))
    return found

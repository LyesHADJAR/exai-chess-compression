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

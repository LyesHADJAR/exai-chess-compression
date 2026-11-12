"""
Tactical probes on ordinary game positions.

Puzzles measure tactics on positions chosen to contain one. These measure the
same thing where it matters more: on the positions a model actually meets in
games, how often does it find the tactic when one is there, compared with the
humans it was trained to imitate?

Mate in one is the cleanest case. It is unambiguous, it can be found by brute
force, and a human who is playing to win nearly always takes it, so the human
rate is a ceiling the model can be held against.

The other motifs here (a forced mate in two, a free piece, a fork) are looser
and say so: each is a definition of "a good move exists", chosen to be cheap to
check and to avoid the obvious false positives, not a claim about the best move.
They are used the same way as mate in one, to show the model more positions
where the tactic is there, and to measure how often it takes it against the
humans it imitates.
"""

import chess
import numpy as np

from .constants import N_MOVES
from .encoding import BOARD_RECORD_BYTES, MOVE_TO_INDEX, MOVE_VOCABULARY, unpack_board


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


#: Material values for the motif definitions. The king is worth more than any
#: piece so that attacking it always counts as attacking something valuable.
VALUE = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,
}


def mate_in_two_moves(board: chess.Board) -> set[str]:
    """First moves of a forced mate in two that begin with a check.

    Searching every first move would cost a thousand times a mate-in-one scan.
    Limiting the first move to checks leaves the replies few and keeps the scan
    affordable, at the price of missing the mates that begin quietly. A position
    that already has a mate in one is not a mate-in-two position, so it gives
    nothing here.
    """
    if mating_moves(board):
        return set()
    found = set()
    for move in list(board.legal_moves):
        if not board.gives_check(move):
            continue
        board.push(move)
        replies = list(board.legal_moves)
        forced = bool(replies)  # no replies would be mate in one, excluded above
        for reply in replies:
            board.push(reply)
            mated = bool(mating_moves(board))
            board.pop()
            if not mated:
                forced = False
                break
        board.pop()
        if forced:
            found.add(move.uci())
    return found


def free_piece_moves(board: chess.Board, previous_to: int | None = None) -> set[str]:
    """Captures that win a knight, bishop, rook or queen for next to nothing.

    The piece is either undefended, or it is defended but taken by something
    worth at least two points less. Pawns are not counted as prizes.

    `previous_to` is the square the opponent's last move landed on. Taking back
    there is a recapture, not a win, and without this exclusion those make up
    most of what the definition finds. When it is not known they are included.
    """
    found = set()
    them = not board.turn
    for move in board.legal_moves:
        if not board.is_capture(move) or board.is_en_passant(move):
            continue
        if move.to_square == previous_to:
            continue
        victim = board.piece_at(move.to_square)
        if victim is None or victim.piece_type == chess.PAWN:
            continue
        gain = VALUE[victim.piece_type]
        if board.is_attacked_by(them, move.to_square):
            board.push(move)
            defended = board.is_attacked_by(them, move.to_square)
            board.pop()
            if defended:
                gain -= VALUE[board.piece_at(move.from_square).piece_type]
        if gain >= 2:
            found.add(move.uci())
    return found


def fork_moves(board: chess.Board) -> set[str]:
    """Moves after which one piece attacks two or more worthwhile targets, safely.

    A target is an enemy knight, bishop, rook, queen or king that is worth more
    than the forking piece or is undefended. The forking piece must be safe:
    nothing cheaper than it can take it, and if something dearer can, we must
    be defending the square. Promotions are skipped.
    """
    found = set()
    us, them = board.turn, not board.turn
    for move in list(board.legal_moves):
        if move.promotion:
            continue
        board.push(move)
        forker = board.piece_at(move.to_square)
        value = VALUE[forker.piece_type]
        attackers = board.attackers(us, move.to_square)  # ours, now defending it
        threats = [
            VALUE[board.piece_at(sq).piece_type] for sq in board.attackers(them, move.to_square)
        ]
        safe = not threats or (min(threats) > value and bool(attackers))
        if safe:
            targets = 0
            for square in board.attacks(move.to_square):
                piece = board.piece_at(square)
                if piece is None or piece.color != them or piece.piece_type == chess.PAWN:
                    continue
                if VALUE[piece.piece_type] > value or not board.is_attacked_by(them, square):
                    targets += 1
            if targets >= 2:
                found.add(move.uci())
        board.pop()
    return found


#: The motifs that can be scanned for, by name. Each maps a position to the
#: set of moves that carry out the tactic (empty if there is none).
MOTIFS = {
    "mate_in_one": mating_moves,
    "mate_in_two": mate_in_two_moves,
    "free_piece": free_piece_moves,
    "fork": fork_moves,
}


#: The square each move of the vocabulary lands on, to tell a recapture from a capture.
MOVE_TO_SQUARE = np.asarray(
    [chess.parse_square(uci[2:4]) for uci in MOVE_VOCABULARY], dtype=np.uint8
)
assert len(MOVE_TO_SQUARE) == N_MOVES

#: Marks "no previous move" (the first position of a game) in the `previous` bytes.
NO_PREVIOUS = 255


def scan_motif(
    records: bytes, motif: str = "mate_in_one", previous: bytes | None = None
) -> list[tuple[int, list[int]]]:
    """Like `scan_mates`, for any motif in MOTIFS.

    `previous` holds, for each record, the square the previous move of its game
    landed on (`NO_PREVIOUS` for a game's first position). Only free_piece uses it.
    """
    find = MOTIFS[motif]
    block = np.frombuffer(records, dtype=np.uint8).reshape(-1, BOARD_RECORD_BYTES)
    landed = np.frombuffer(previous, dtype=np.uint8) if previous is not None else None
    found = []
    for row, record in enumerate(block):
        board = unpack_board(record.tobytes())
        if motif == "free_piece" and landed is not None and landed[row] != NO_PREVIOUS:
            moves = find(board, int(landed[row]))
        else:
            moves = find(board)
        if moves:
            found.append((row, sorted(MOVE_TO_INDEX[m] for m in moves)))
    return found


def scan_block(payload: tuple[bytes, bytes], motif: str) -> list[tuple[int, list[int]]]:
    """`scan_motif` for one (records, previous) pair, in the shape a worker pool hands it."""
    records, previous = payload
    return scan_motif(records, motif, previous)


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

"""
Move vocabulary and on-disk board encoding.

Two things live here, and both are load-bearing for the released artifact:
the fixed move vocabulary the policy head indexes into, and the packed record
format that positions are stored in. Changing either invalidates every encoded
corpus and every checkpoint trained on one.
"""

import chess

from .constants import N_MOVES

# ── Move vocabulary ───────────────────────────────────────────────────────

PROMOTION_PIECES = (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)


def _reachable(piece_type: int, square: int) -> chess.SquareSet:
    """Squares a lone piece of this type attacks from here on an empty board."""
    board = chess.Board.empty()
    board.set_piece_at(square, chess.Piece(piece_type, chess.WHITE))
    return board.attacks(square)


def _generate_vocabulary() -> tuple[str, ...]:
    """Every UCI move that can occur in legal chess, lexicographically sorted.

    A queen and a knight between them cover every non-promotion geometry any
    piece can move through, so the from-to pairs they reach on an empty board
    are the complete set. Promotions are then added as separate UCI strings,
    since `a7a8` and `a7a8q` are different moves.

    This comes to 1,968 moves, the same vocabulary DeepMind's searchless_chess
    uses. `N_MOVES` asserts the count so a change here cannot pass silently.
    """
    moves: set[str] = set()

    for from_square in chess.SQUARES:
        reachable = _reachable(chess.QUEEN, from_square) | _reachable(chess.KNIGHT, from_square)
        for to_square in reachable:
            moves.add(chess.Move(from_square, to_square).uci())

    for rank_from, rank_to in ((6, 7), (1, 0)):  # white 7->8, black 2->1
        for file_from in range(8):
            from_square = chess.square(file_from, rank_from)
            for offset in (-1, 0, 1):
                file_to = file_from + offset
                if not 0 <= file_to <= 7:
                    continue
                to_square = chess.square(file_to, rank_to)
                for promotion in PROMOTION_PIECES:
                    moves.add(chess.Move(from_square, to_square, promotion=promotion).uci())

    return tuple(sorted(moves))


MOVE_VOCABULARY: tuple[str, ...] = _generate_vocabulary()
MOVE_TO_INDEX: dict[str, int] = {uci: i for i, uci in enumerate(MOVE_VOCABULARY)}

assert len(MOVE_VOCABULARY) == N_MOVES, (
    f"move vocabulary is {len(MOVE_VOCABULARY)}, expected {N_MOVES}"
)

# ── Packed board record ───────────────────────────────────────────────────

#: Layout, little-endian, one record per position:
#:   bytes  0..31  64 squares at 4 bits each; square 0 (a1) is the low nibble
#:                 of byte 0. Piece codes: 0 empty, 1-6 white P N B R Q K,
#:                 7-12 black p n b r q k.
#:   byte   32     bit 0 side to move (1 = white), bits 1-4 castling KQkq
#:   byte   33     en passant file 0-7, or 255 for none
#:   byte   34     halfmove clock, saturated at 255
#:   byte   35     reserved, keeps the record even-width
BOARD_RECORD_BYTES = 36

_NO_EN_PASSANT = 255

_CASTLING_BITS = (
    (chess.BB_H1, chess.WHITE, 1),  # white kingside
    (chess.BB_A1, chess.WHITE, 2),  # white queenside
    (chess.BB_H8, chess.BLACK, 3),  # black kingside
    (chess.BB_A8, chess.BLACK, 4),  # black queenside
)


def piece_code(piece: chess.Piece | None) -> int:
    """0 for an empty square, 1-6 for white pieces, 7-12 for black."""
    if piece is None:
        return 0
    return piece.piece_type + (0 if piece.color == chess.WHITE else 6)


def pack_board(board: chess.Board) -> bytes:
    """Encode a position into one fixed-width record.

    Stored as it stands, in the colour it was played. Canonicalising to the
    side to move happens at load time, over a whole K-ply window at once, so
    the stored corpus stays faithful to the source.
    """
    record = bytearray(BOARD_RECORD_BYTES)

    for square in chess.SQUARES:
        code = piece_code(board.piece_at(square))
        if code == 0:
            continue
        index, high = divmod(square, 2)
        record[index] |= code << 4 if high else code

    flags = 1 if board.turn == chess.WHITE else 0
    for mask, colour, bit in _CASTLING_BITS:
        if board.castling_rights & mask and board.king(colour) is not None:
            flags |= 1 << bit
    record[32] = flags

    record[33] = chess.square_file(board.ep_square) if board.ep_square else _NO_EN_PASSANT
    record[34] = min(board.halfmove_clock, 255)
    return bytes(record)


def unpack_board(record: bytes) -> chess.Board:
    """Rebuild a position from a packed record.

    Exists so the encoding can be round-tripped in tests. The move counters
    are not fully recoverable: fullmove number is derivable from the ply index
    and is not stored.
    """
    if len(record) != BOARD_RECORD_BYTES:
        raise ValueError(f"expected {BOARD_RECORD_BYTES} bytes, got {len(record)}")

    board = chess.Board.empty()
    for square in chess.SQUARES:
        index, high = divmod(square, 2)
        code = (record[index] >> 4) if high else (record[index] & 0x0F)
        if code == 0:
            continue
        colour = chess.WHITE if code <= 6 else chess.BLACK
        board.set_piece_at(square, chess.Piece(code - (0 if colour else 6), colour))

    flags = record[32]
    board.turn = chess.WHITE if flags & 1 else chess.BLACK

    rights = chess.BB_EMPTY
    for mask, _colour, bit in _CASTLING_BITS:
        if flags & (1 << bit):
            rights |= mask
    board.castling_rights = rights

    if record[33] != _NO_EN_PASSANT:
        rank = 5 if board.turn == chess.WHITE else 2
        board.ep_square = chess.square(record[33], rank)

    board.halfmove_clock = record[34]
    return board

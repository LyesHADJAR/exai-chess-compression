"""Decoding packed records and cutting windows from a corpus."""

import chess
import numpy as np
import pytest
import torch

from chessdt.constants import CASTLING_ROOK_CODES, ELO_CENTRE, ELO_SCALE, EN_PASSANT_CODE
from chessdt.data import Corpus, decode_records
from chessdt.encoding import BOARD_RECORD_BYTES, MOVE_TO_INDEX, pack_board, piece_code
from chessdt.splits import TEST, TRAIN


def decode(board: chess.Board):
    record = torch.frombuffer(bytearray(pack_board(board)), dtype=torch.uint8)
    codes, side = decode_records(record[None])
    return codes[0].tolist(), int(side[0])


# ── Decoding ──────────────────────────────────────────────────────────────


def test_pieces_land_on_their_squares():
    board = chess.Board("8/8/8/3k4/8/2N5/8/4K3 w - - 0 1")
    codes, side = decode(board)
    assert side == 1
    for square in chess.SQUARES:
        assert codes[square] == piece_code(board.piece_at(square))


def test_side_to_move_is_read_from_the_record():
    assert decode(chess.Board())[1] == 1
    after = chess.Board()
    after.push_san("e4")
    assert decode(after)[1] == 0


def test_rooks_with_castling_rights_get_their_own_code():
    codes, _ = decode(chess.Board())
    white, black = CASTLING_ROOK_CODES
    assert [codes[chess.A1], codes[chess.H1]] == [white, white]
    assert [codes[chess.A8], codes[chess.H8]] == [black, black]


def test_a_rook_without_the_right_is_an_ordinary_rook():
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w Kq - 0 1")
    codes, _ = decode(board)
    white, black = CASTLING_ROOK_CODES
    assert codes[chess.H1] == white and codes[chess.A1] == piece_code(board.piece_at(chess.A1))
    assert codes[chess.A8] == black and codes[chess.H8] == piece_code(board.piece_at(chess.H8))


@pytest.mark.parametrize(
    ("fen", "target"),
    [
        ("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3", chess.D6),
        ("rnbqkbnr/pppp1ppp/8/8/3Pp3/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 3", chess.D3),
    ],
)
def test_the_en_passant_target_is_marked_on_its_square(fen, target):
    codes, _ = decode(chess.Board(fen))
    assert codes[target] == EN_PASSANT_CODE
    assert codes.count(EN_PASSANT_CODE) == 1


def test_no_en_passant_marks_nothing():
    assert EN_PASSANT_CODE not in decode(chess.Board())[0]


def test_decoding_does_not_modify_the_stored_records():
    records = torch.frombuffer(bytearray(pack_board(chess.Board())), dtype=torch.uint8)[None]
    before = records.clone()
    decode_records(records)
    assert torch.equal(records, before)


# ── Windows ───────────────────────────────────────────────────────────────


def write_corpus(directory, games):
    """games: list of (sans, result, white_elo, black_elo, split)."""
    records, moves, lengths, labels = bytearray(), [], [], []
    for sans, _result, _white, _black, split in games:
        board = chess.Board()
        for san in sans:
            move = board.parse_san(san)
            records += pack_board(board)
            moves.append(MOVE_TO_INDEX[move.uci()])
            board.push(move)
        lengths.append(len(sans))
        labels += [split] * len(sans)

    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    (directory / "positions.bin").write_bytes(bytes(records))
    (directory / "moves.bin").write_bytes(np.asarray(moves, dtype=np.uint16).tobytes())
    np.save(directory / "split_labels.npy", np.asarray(labels, dtype=np.uint8))
    np.save(directory / "game_splits.npy", np.asarray([g[4] for g in games], dtype=np.uint8))
    np.savez(
        directory / "games.npz",
        game_id=np.asarray([f"g{i}" for i in range(len(games))], dtype="S12"),
        offset=offsets,
        length=np.asarray(lengths, dtype=np.int32),
        result=np.asarray([g[1] for g in games], dtype=np.int8),
        white_elo=np.asarray([g[2] for g in games], dtype=np.int16),
        black_elo=np.asarray([g[3] for g in games], dtype=np.int16),
        record_bytes=np.int32(BOARD_RECORD_BYTES),
    )


@pytest.fixture
def corpus(tmp_path):
    write_corpus(
        tmp_path,
        [
            (["e4", "e5", "Nf3", "Nc6", "Bb5"], 1, 2400, 1900, int(TRAIN)),
            (["d4", "d5", "c4"], -1, 2000, 2200, int(TEST)),
        ],
    )
    return Corpus(tmp_path)


def test_a_window_stops_being_valid_at_the_end_of_its_game(corpus):
    batch = corpus.window(torch.tensor([3]), plies=4)
    assert batch.valid[0].tolist() == [True, True, False, False]


def test_outcome_is_seen_from_the_player_to_move(corpus):
    """White won game 0, so white's plies read +1 and black's read -1."""
    batch = corpus.window(torch.tensor([0]), plies=5)
    assert batch.cond[0, :, 0].tolist() == [1.0, -1.0, 1.0, -1.0, 1.0]


def test_rating_is_that_of_the_player_to_move(corpus):
    batch = corpus.window(torch.tensor([0]), plies=2)
    expected = [(2400 - ELO_CENTRE) / ELO_SCALE, (1900 - ELO_CENTRE) / ELO_SCALE]
    assert batch.cond[0, :, 1].tolist() == pytest.approx(expected)


def test_moves_and_labels_follow_the_positions(corpus):
    batch = corpus.window(torch.tensor([0]), plies=2)
    assert batch.moves[0].tolist() == [MOVE_TO_INDEX["e2e4"], MOVE_TO_INDEX["e7e5"]]
    assert batch.labels[0].tolist() == [int(TRAIN), int(TRAIN)]


def test_training_samples_come_only_from_training_games(corpus):
    batch = corpus.sample_train(64, 3, torch.Generator().manual_seed(0))
    assert (batch.labels[:, 0] == TRAIN).all()


def test_split_starts_tile_each_game_from_its_first_ply(corpus):
    assert corpus.split_starts(int(TRAIN), plies=2).tolist() == [0, 2, 4]
    assert corpus.split_starts(int(TEST), plies=2).tolist() == [5, 7]


def test_an_inconsistent_corpus_is_refused(tmp_path):
    write_corpus(tmp_path, [(["e4", "e5"], 0, 2000, 2000, int(TRAIN))])
    np.save(tmp_path / "split_labels.npy", np.zeros(99, dtype=np.uint8))
    with pytest.raises(ValueError, match="inconsistent"):
        Corpus(tmp_path)

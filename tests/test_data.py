"""Decoding packed records and cutting windows from a corpus."""

import chess
import numpy as np
import pytest
import torch

from chessdt.constants import CASTLING_ROOK_CODES, ELO_CENTRE, ELO_SCALE, EN_PASSANT_CODE
from chessdt.data import Corpus, decode_records, tactical_fraction_at
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


# ── Tactical sampling ─────────────────────────────────────────────────────


def tactics_file(path, positions, splits):
    np.savez(
        path,
        position=np.asarray(positions, dtype=np.int64),
        split=np.asarray(splits, dtype=np.uint8),
        offset=np.arange(len(positions) + 1, dtype=np.int64),
        moves=np.zeros(len(positions), dtype=np.uint16),
    )


def test_tactical_windows_contain_a_tactical_position(corpus, tmp_path):
    tactics_file(tmp_path / "t.npz", [2], [int(TRAIN)])  # a position in game 0
    assert corpus.load_tactics(tmp_path / "t.npz") == 1

    generator = torch.Generator().manual_seed(0)
    for _ in range(20):
        batch = corpus.sample_train(8, 3, generator, tactical_fraction=1.0)
        # Position 2 is ply 2 of game 0; its window starts at 0, 1 or 2.
        assert batch.moves.shape == (8, 3)
    starts = set()
    for _ in range(40):
        s = corpus.sample_train(4, 3, generator, tactical_fraction=1.0).moves[:, 0]
        starts.update(s.tolist())
    # Starting at ply 0, 1 or 2 puts a different move first, so all three occur.
    assert len(starts) == 3


def test_a_tactical_window_never_starts_before_its_game(corpus, tmp_path):
    tactics_file(tmp_path / "t.npz", [0], [int(TRAIN)])  # the very first position
    corpus.load_tactics(tmp_path / "t.npz")
    generator = torch.Generator().manual_seed(1)
    for _ in range(30):
        batch = corpus.sample_train(6, 4, generator, tactical_fraction=1.0)
        assert batch.valid[:, 0].all(), "a window left its game"
        assert (batch.moves[:, 0] == MOVE_TO_INDEX["e2e4"]).all()


def test_held_out_tactical_positions_are_never_loaded(corpus, tmp_path):
    tactics_file(tmp_path / "t.npz", [0, 5, 6], [int(TRAIN), int(TEST), int(TEST)])
    assert corpus.load_tactics(tmp_path / "t.npz") == 1


def test_a_file_with_only_held_out_positions_is_refused(corpus, tmp_path):
    tactics_file(tmp_path / "t.npz", [5], [int(TEST)])
    with pytest.raises(ValueError, match="--splits train"):
        corpus.load_tactics(tmp_path / "t.npz")


def test_tactical_sampling_needs_the_positions_first(corpus):
    with pytest.raises(ValueError, match="load_tactics"):
        corpus.sample_train(4, 3, torch.Generator().manual_seed(0), tactical_fraction=0.5)


def test_without_a_fraction_sampling_is_unchanged(corpus):
    a = corpus.sample_train(6, 3, torch.Generator().manual_seed(3))
    b = corpus.sample_train(6, 3, torch.Generator().manual_seed(3), tactical_fraction=0.0)
    assert torch.equal(a.moves, b.moves)


# ── The tactical schedule ─────────────────────────────────────────────────


def test_without_annealing_the_fraction_is_constant():
    assert [tactical_fraction_at(s, 100, 0.4) for s in (0, 50, 99)] == [0.4, 0.4, 0.4]


def test_annealing_falls_in_a_straight_line_to_zero():
    values = [tactical_fraction_at(s, 100, 0.5, anneal=0.8) for s in (0, 20, 40, 60, 80)]
    assert values == pytest.approx([0.5, 0.375, 0.25, 0.125, 0.0])


def test_annealing_stays_at_zero_afterwards_and_never_goes_negative():
    assert tactical_fraction_at(95, 100, 0.5, anneal=0.8) == 0.0
    assert tactical_fraction_at(10**6, 100, 0.5, anneal=0.8) == 0.0


def test_a_window_with_no_tactical_share_needs_no_tactics_loaded(corpus):
    # Once annealed to zero the sampler must not demand the tactical positions.
    batch = corpus.sample_train(4, 3, torch.Generator().manual_seed(0), tactical_fraction=0.0)
    assert batch.moves.shape == (4, 3)


def test_several_tactic_sets_are_mixed_evenly(corpus, tmp_path):
    # Set A holds one position, set B holds four: an even mix still picks A half the time.
    tactics_file(tmp_path / "a.npz", [0], [int(TRAIN)])
    tactics_file(tmp_path / "b.npz", [1, 2, 3, 4], [int(TRAIN)] * 4)
    assert corpus.load_tactics([tmp_path / "a.npz", tmp_path / "b.npz"]) == 5

    generator = torch.Generator().manual_seed(0)
    first = torch.cat(
        [
            corpus.sample_train(32, 1, generator, tactical_fraction=1.0).moves[:, 0]
            for _ in range(40)
        ]
    )
    from_a = (first == MOVE_TO_INDEX["e2e4"]).float().mean().item()  # only position 0 plays e2e4
    assert 0.42 < from_a < 0.58


def test_every_position_of_every_set_is_reachable(corpus, tmp_path):
    tactics_file(tmp_path / "a.npz", [0, 1], [int(TRAIN)] * 2)
    tactics_file(tmp_path / "b.npz", [3, 4], [int(TRAIN)] * 2)
    corpus.load_tactics([tmp_path / "a.npz", tmp_path / "b.npz"])
    generator = torch.Generator().manual_seed(1)
    seen = set()
    for _ in range(60):
        seen.update(
            corpus.sample_train(8, 1, generator, tactical_fraction=1.0).moves[:, 0].tolist()
        )
    assert len(seen) == 4  # moves of positions 0, 1, 3 and 4 are all different


def test_one_file_still_loads_as_a_single_path(corpus, tmp_path):
    tactics_file(tmp_path / "a.npz", [0], [int(TRAIN)])
    assert corpus.load_tactics(str(tmp_path / "a.npz")) == 1

"""
Settled parameters of the challenge.

Everything here has been agreed and is depended on elsewhere. Values still
under discussion live in the design docs, not in this file. Changing a number
here changes the released artifact, so `tests/test_budget.py` guards the ones
that determine its size.
"""

from dataclasses import dataclass

# ── Encoding ──────────────────────────────────────────────────────────────

#: FEN flattened to a fixed-length token sequence, following DeepMind's
#: searchless_chess convention: 64 squares, side to move, 4 castling rights,
#: en passant target, and the two move counters.
POSITION_TOKENS = 77

#: Every UCI move that can occur in legal chess, lexicographically sorted.
#: Generated rather than hardcoded; this is the expected length.
N_MOVES = 1968

# ── Reference dump ────────────────────────────────────────────────────────

#: Figures read off database.lichess.org on 2026-10-29. The game count is
#: unfiltered: every rating and time control, bullet included.
DUMP_MONTH = "2026-09"
DUMP_GAMES = 89_616_462
DUMP_SIZE_GB = 29.2
DUMP_URL = f"https://database.lichess.org/standard/lichess_db_standard_rated_{DUMP_MONTH}.pgn.zst"


# ── Collection filters ────────────────────────────────────────────────────

MIN_ELO = 1800  # both players
MIN_CLOCK_SECONDS = 180  # initial clock; excludes bullet
MIN_PLIES = 20
TARGET_GAMES = 300_000

# ── Splits ────────────────────────────────────────────────────────────────

#: Assigned by hashing the game ID, so the split is reproducible from the
#: dump alone and no split file has to be stored.
SPLIT_FRACTIONS = {"train": 0.90, "val": 0.05, "test": 0.05}


@dataclass(frozen=True)
class ModelConfig:
    """Reference architecture.

    A board encoder compresses 64 square tokens into one vector per ply; the
    Decision Transformer backbone then runs over the 3K-token interleaved
    stream of return, state and action.
    """

    # Board encoder
    enc_dim: int = 256
    enc_layers: int = 3
    enc_heads: int = 8
    enc_ff: int = 768

    # Decision Transformer backbone
    dim: int = 384
    layers: int = 4
    heads: int = 8
    ff: int = 1024

    # Sequence
    context_plies: int = 16

    dropout: float = 0.1

    @property
    def stream_tokens(self) -> int:
        """Length of the interleaved [R, s, a] stream."""
        return self.context_plies * 3

    def parameter_count(self) -> int:
        """Exact parameter count for this configuration.

        Kept in sync with the model by `tests/test_budget.py`, which also
        guards the released artifact size.
        """

        def block(dim: int, ff: int) -> int:
            attn = 4 * dim * dim  # qkv + output projection, no bias
            swiglu = 3 * dim * ff  # w1, w3, w2
            norms = 2 * 2 * dim  # two LayerNorms, weight and bias
            return attn + swiglu + norms

        encoder = self.enc_layers * block(self.enc_dim, self.enc_ff)
        backbone = self.layers * block(self.dim, self.ff)

        piece_embed = 13 * self.enc_dim
        square_embed = 64 * self.enc_dim
        projection = self.enc_dim * self.dim + self.dim
        move_embed = N_MOVES * self.dim
        policy_head = self.dim * N_MOVES + N_MOVES
        return_embed = self.dim + self.dim
        timestep_embed = (self.context_plies + 1) * self.dim
        final_norm = 2 * self.dim

        return (
            encoder
            + backbone
            + piece_embed
            + square_embed
            + projection
            + move_embed
            + policy_head
            + return_embed
            + timestep_embed
            + final_norm
        )

    def artifact_bytes(self, bytes_per_param: int = 4) -> int:
        """Size of the released fp32 checkpoint."""
        return self.parameter_count() * bytes_per_param


REFERENCE = ModelConfig()

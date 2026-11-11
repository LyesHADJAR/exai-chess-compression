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


def dump_url(month: str) -> str:
    """The monthly standard-games dump for `month`, written like 2026-09."""
    return f"https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"


DUMP_URL = dump_url(DUMP_MONTH)


#: What a square can hold as the model sees it. The twelve pieces and the
#: empty square, plus three codes that fold the rest of the position into the
#: board itself: an empty square that is the en passant target, and a rook
#: that still carries its castling right, one code per colour.
SQUARE_CODES = 16
EN_PASSANT_CODE = 13
CASTLING_ROOK_CODES = (14, 15)  # white, black

#: Conditioning channels on the return token: outcome-to-go for the player to
#: move, and that player's rating. Kept separate so each can be ablated alone.
CONDITION_DIM = 2
ELO_CENTRE = 2000.0
ELO_SCALE = 400.0

#: How a puzzle is conditioned. It has no game around it, so the convention is
#: fixed: the solver is winning and is a strong player.
PUZZLE_OUTCOME = 1.0
PUZZLE_ELO = 2400.0


# ── Collection filters ────────────────────────────────────────────────────

MIN_ELO = 1800  # both players
MIN_CLOCK_SECONDS = 180  # initial clock; excludes bullet
MIN_PLIES = 20
TARGET_GAMES = 300_000

# ── Splits ────────────────────────────────────────────────────────────────

#: Assigned by hashing the game ID, so the split is reproducible from the
#: dump alone and no split file has to be stored.
SPLIT_FRACTIONS = {"train": 0.90, "val": 0.05, "test": 0.05}


#: Offsets a square can differ from another by on the board: 15 each way.
REL_OFFSETS = 15 * 15
#: Distinct files, ranks and the two diagonals a square sits on.
GEOMETRY_ROWS = 8 + 8 + 15 + 15


@dataclass(frozen=True)
class ModelConfig:
    """Architecture of the chess Decision Transformer.

    A board encoder reads the 64 squares of a position; the Decision
    Transformer backbone then runs over the interleaved stream of condition,
    position and move. Every option below defaults to the reference model, so a
    config that sets none of them is exactly v1.
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

    # How much of the board reaches the backbone. The reference pools all 64
    # squares into one vector per ply. More vectors keep more of the position;
    # full_detail keeps every square.
    summary_tokens: int = 1
    full_detail: bool = False

    #: Board geometry in the encoder: squares are embedded as the sum of their
    #: file, rank and both diagonals, and attention gets a learned bias by the
    #: offset between two squares.
    geometry: bool = False

    #: Share of training steps in which the conditioning is replaced by a
    #: learned "unknown", so the model learns what the conditioning does.
    cond_dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.summary_tokens < 1:
            raise ValueError("summary_tokens must be at least 1")
        if not 0.0 <= self.cond_dropout < 1.0:
            raise ValueError("cond_dropout must be in [0, 1)")

    @property
    def state_tokens(self) -> int:
        """How many tokens each position occupies in the backbone's stream."""
        return 64 if self.full_detail else self.summary_tokens

    @property
    def stream_tokens(self) -> int:
        """Length of the interleaved [condition, position..., move] stream."""
        return self.context_plies * (self.state_tokens + 2)

    def parameter_count(self) -> int:
        """Exact parameter count for this configuration.

        `tests/test_model.py` asserts the built model matches this to the
        parameter for every option, and `tests/test_budget.py` guards the
        artifact size.
        """

        def block(dim: int, ff: int) -> int:
            attn = 4 * dim * dim  # qkv + output projection, no bias
            swiglu = 3 * dim * ff  # w1, w3, w2
            norms = 2 * 2 * dim  # two LayerNorms, weight and bias
            return attn + swiglu + norms

        encoder = self.enc_layers * block(self.enc_dim, self.enc_ff)
        backbone = self.layers * block(self.dim, self.ff)

        square_code_embed = SQUARE_CODES * self.enc_dim
        square_embed = (GEOMETRY_ROWS if self.geometry else 64) * self.enc_dim
        relative_bias = self.enc_layers * self.enc_heads * REL_OFFSETS if self.geometry else 0
        side_embed = 2 * self.enc_dim
        encoder_norm = 2 * self.enc_dim
        pooling = 0
        if self.summary_tokens > 1 and not self.full_detail:
            pooling = self.summary_tokens * self.enc_dim + 4 * self.enc_dim**2
        projection = self.enc_dim * self.dim + self.dim
        condition_embed = CONDITION_DIM * self.dim + self.dim
        condition_null = self.dim if self.cond_dropout > 0 else 0
        move_embed = N_MOVES * self.dim
        ply_embed = self.context_plies * self.dim
        final_norm = 2 * self.dim
        policy_head = self.dim * N_MOVES + N_MOVES

        return (
            encoder
            + backbone
            + square_code_embed
            + square_embed
            + relative_bias
            + side_embed
            + encoder_norm
            + pooling
            + projection
            + condition_embed
            + condition_null
            + move_embed
            + ply_embed
            + final_norm
            + policy_head
        )

    def artifact_bytes(self, bytes_per_param: int = 4) -> int:
        """Size of the released fp32 checkpoint."""
        return self.parameter_count() * bytes_per_param


REFERENCE = ModelConfig()


#: The size ladder. v1 is the reference; the others spend the room a 100 MiB
#: artifact allows. Each stays under the cap with every option switched on.
PRESETS = {
    "v1": REFERENCE,
    "m18": ModelConfig(enc_dim=320, enc_layers=4, enc_ff=960, dim=384, layers=6, ff=1024),
    "m25": ModelConfig(enc_dim=320, enc_layers=5, enc_ff=960, dim=512, layers=5, ff=1344),
}

#: The largest artifact we will ship, in bytes of fp32 weights.
SIZE_CAP_BYTES = 100 * 1024**2

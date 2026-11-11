"""
The reference chess Decision Transformer.

Two stages. A board encoder reads the 64 squares of one position with
bidirectional attention and pools them into a single vector. The backbone is a
causal transformer over the interleaved stream

    [c_1, s_1, a_1, c_2, s_2, a_2, ..., c_K, s_K, a_K]

where c is the conditioning token (outcome-to-go and rating of the player to
move), s the encoded position and a the move played from it. A position can
occupy one token (the reference), several pooled ones, or all 64 squares;
`ModelConfig.state_tokens` says which, and the stream grows to match.

Module and parameter names are load-bearing: released checkpoints and every
compressed submission are keyed on them.
"""

import math
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import CONDITION_DIM, N_MOVES, REL_OFFSETS, SQUARE_CODES, ModelConfig


class SwiGLU(nn.Module):
    """Gated feed-forward: ``(silu(W1 x) * W3 x) W2``."""

    def __init__(self, dim: int, ff: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(dim, ff, bias=False)
        self.w3 = nn.Linear(dim, ff, bias=False)
        self.w2 = nn.Linear(ff, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


def relative_square_index() -> torch.Tensor:
    """For every pair of squares, which (rank offset, file offset) separates them."""
    rank, file = torch.arange(64) // 8, torch.arange(64) % 8
    d_rank = rank[None, :] - rank[:, None] + 7
    d_file = file[None, :] - file[:, None] + 7
    return d_rank * 15 + d_file


class Attention(nn.Module):
    """Multi-head self-attention, causal, bidirectional or under an explicit mask.

    With `relative_bias` each head also learns a bias for every offset between
    two squares, so "same file", "same diagonal" or "a knight's move away" is
    one shared number rather than something rediscovered per pair of squares.
    """

    def __init__(
        self, dim: int, heads: int, dropout: float, causal: bool, relative_bias: bool = False
    ):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.dropout = dropout
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.relative = nn.Parameter(torch.zeros(heads, REL_OFFSETS)) if relative_bias else None
        if relative_bias:
            self.register_buffer("relative_index", relative_square_index(), persistent=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, tokens, dim = x.shape
        q, k, v = (
            t.view(batch, tokens, self.heads, dim // self.heads).transpose(1, 2)
            for t in self.qkv(x).split(dim, dim=-1)
        )
        if self.relative is not None:
            mask = self.relative[:, self.relative_index].to(q.dtype)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            is_causal=self.causal and mask is None,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.proj(out.transpose(1, 2).reshape(batch, tokens, dim))


class Block(nn.Module):
    """Pre-LN transformer block."""

    def __init__(
        self,
        dim: int,
        heads: int,
        ff: int,
        dropout: float,
        causal: bool,
        relative_bias: bool = False,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout, causal, relative_bias)
        self.ffn = SwiGLU(dim, ff, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask)
        return x + self.ffn(self.ln2(x))


class AttentionPool(nn.Module):
    """Summarise the squares into k vectors, each by attending to what it needs.

    Mean pooling gives every square the same say and one vector has to carry
    the whole position. Here each of k learned queries picks its own mixture of
    squares, so different vectors can hold different parts of the position.
    """

    def __init__(self, dim: int, heads: int, k: int):
        super().__init__()
        self.heads = heads
        self.queries = nn.Parameter(torch.zeros(k, dim))
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, 2 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, squares, dim = x.shape
        split = lambda t: t.view(batch, -1, self.heads, dim // self.heads).transpose(1, 2)  # noqa: E731
        q = split(self.q(self.queries).expand(batch, -1, -1))
        k, v = (split(t) for t in self.kv(x).chunk(2, dim=-1))
        pooled = F.scaled_dot_product_attention(q, k, v)
        return self.out(pooled.transpose(1, 2).reshape(batch, -1, dim))


class BoardEncoder(nn.Module):
    """One position in, one or more vectors out.

    Each square is a token: what stands on it plus where it is, with the side
    to move added to every square. Castling rights and the en passant target
    arrive as square codes, so the whole position is in the 64 tokens.

    What comes out depends on the config: the mean of the squares (the
    reference), k attention-pooled vectors, or every square.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.code_embed = nn.Embedding(SQUARE_CODES, cfg.enc_dim)
        if cfg.geometry:
            # A square is the sum of its file, rank and the two diagonals through it.
            self.file_embed = nn.Embedding(8, cfg.enc_dim)
            self.rank_embed = nn.Embedding(8, cfg.enc_dim)
            self.diag_embed = nn.Embedding(15, cfg.enc_dim)
            self.anti_embed = nn.Embedding(15, cfg.enc_dim)
            squares = torch.arange(64)
            rank, file = squares // 8, squares % 8
            self.register_buffer("files", file, persistent=False)
            self.register_buffer("ranks", rank, persistent=False)
            self.register_buffer("diags", rank - file + 7, persistent=False)
            self.register_buffer("antis", rank + file, persistent=False)
        else:
            self.square_embed = nn.Embedding(64, cfg.enc_dim)
            self.register_buffer("squares", torch.arange(64), persistent=False)
        self.side_embed = nn.Embedding(2, cfg.enc_dim)
        self.blocks = nn.ModuleList(
            Block(cfg.enc_dim, cfg.enc_heads, cfg.enc_ff, cfg.dropout, False, cfg.geometry)
            for _ in range(cfg.enc_layers)
        )
        self.norm = nn.LayerNorm(cfg.enc_dim)
        self.pool = (
            AttentionPool(cfg.enc_dim, cfg.enc_heads, cfg.summary_tokens)
            if cfg.summary_tokens > 1 and not cfg.full_detail
            else None
        )
        self.proj = nn.Linear(cfg.enc_dim, cfg.dim)

    def square_position(self) -> torch.Tensor:
        if not self.cfg.geometry:
            return self.square_embed(self.squares)
        return (
            self.file_embed(self.files)
            + self.rank_embed(self.ranks)
            + self.diag_embed(self.diags)
            + self.anti_embed(self.antis)
        )

    def forward(self, codes: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        """codes (N, 64) and side (N,), both long. Returns (N, state_tokens, dim)."""
        x = self.code_embed(codes) + self.square_position() + self.side_embed(side)[:, None]
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        if self.cfg.full_detail:
            return self.proj(x)
        if self.pool is not None:
            return self.proj(self.pool(x))
        return self.proj(x.mean(dim=1)).unsqueeze(1)


def ply_block_mask(plies: int, state_tokens: int) -> torch.Tensor:
    """Which tokens of the stream may attend to which, when a position has several.

    Within a ply the condition and the position tokens see one another in both
    directions, so every square can attend to every other. The move token sees
    its own ply and everything before. Nothing sees the move of its own ply,
    which is what keeps the answer out of reach of the tokens that predict it,
    and nothing sees a later ply.
    """
    width = state_tokens + 2
    index = torch.arange(plies * width)
    ply, offset = index // width, index % width
    earlier = ply[None, :] < ply[:, None]
    same = ply[None, :] == ply[:, None]
    visible = (offset[None, :] <= state_tokens) | (index[None, :] == index[:, None])
    return earlier | (same & visible)


class ChessDT(nn.Module):
    """Decision Transformer over K plies of (condition, position, move).

    The move at ply t is read from the hidden state at the **last position
    token** of ply t, never the move token. Under the attention mask that token
    has seen the conditioning and position of ply t and everything before, but
    not the move it is being asked for; the move token is the embedding of that
    move, and reading there hands the model its own answer.
    `tests/test_model.py` holds the test that pins this down, for every option.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.state_tokens = cfg.state_tokens
        #: Offset of the readout token inside each ply's group of tokens: the
        #: last position token, which sits just before the move token.
        self.readout_offset = self.state_tokens
        self.encoder = BoardEncoder(cfg)
        self.cond_embed = nn.Linear(CONDITION_DIM, cfg.dim)
        self.cond_null = nn.Parameter(torch.randn(cfg.dim) * 0.02) if cfg.cond_dropout > 0 else None
        self.move_embed = nn.Embedding(N_MOVES, cfg.dim)
        self.ply_embed = nn.Embedding(cfg.context_plies, cfg.dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            Block(cfg.dim, cfg.heads, cfg.ff, cfg.dropout, causal=True) for _ in range(cfg.layers)
        )
        self.norm = nn.LayerNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, N_MOVES)
        self.register_buffer("plies", torch.arange(cfg.context_plies), persistent=False)
        if self.state_tokens > 1:
            mask = ply_block_mask(cfg.context_plies, self.state_tokens)
            self.register_buffer("stream_mask", mask, persistent=False)
        else:
            self.stream_mask = None  # plain causal attention, as in the reference
        self._init_weights()

    def _init_weights(self) -> None:
        depth = self.cfg.enc_layers + self.cfg.layers
        for name, param in self.named_parameters():
            if param.dim() < 2:
                if name.endswith("bias"):
                    nn.init.zeros_(param)
                continue
            if name.endswith("relative"):
                continue  # starts at zero: no geometric preference until it is learned
            scaled = name.endswith(("attn.proj.weight", "ffn.w2.weight"))
            nn.init.normal_(param, std=0.02 / math.sqrt(2 * depth) if scaled else 0.02)

    def forward(
        self,
        codes: torch.Tensor,  # (B, K, 64) long   square codes
        side: torch.Tensor,  # (B, K)     long   1 where white is to move
        cond: torch.Tensor | None,  # (B, K, 2) float, or None for "unknown"
        moves: torch.Tensor,  # (B, K)     long   move played at each ply
    ) -> torch.Tensor:  # (B, K, N_MOVES) logits for the move at each ply
        batch, plies = side.shape
        if plies > self.cfg.context_plies:
            raise ValueError(f"{plies} plies exceeds the context of {self.cfg.context_plies}")

        state = self.encoder(codes.reshape(batch * plies, 64), side.reshape(-1))
        state = state.view(batch, plies, self.state_tokens, -1)
        ply = self.ply_embed(self.plies[:plies])

        if cond is None:
            if self.cond_null is None:
                raise ValueError("this model was built without conditioning dropout")
            condition = self.cond_null.expand(batch, plies, -1)
        else:
            condition = self.cond_embed(cond)
            if self.training and self.cond_null is not None and self.cfg.cond_dropout > 0:
                dropped = torch.rand(batch, plies, 1, device=side.device) < self.cfg.cond_dropout
                condition = torch.where(dropped, self.cond_null, condition)

        stream = torch.cat(
            [
                (condition + ply)[:, :, None],
                state + ply[None, :, None],
                (self.move_embed(moves) + ply)[:, :, None],
            ],
            dim=2,
        ).reshape(batch, plies * (self.state_tokens + 2), -1)

        mask = None
        if self.stream_mask is not None:
            mask = self.stream_mask[: stream.shape[1], : stream.shape[1]]

        x = self.drop(stream)
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)

        return self.head(x[:, self.readout_offset :: self.state_tokens + 2])


def save_checkpoint(path, model: ChessDT, **extra) -> None:
    """Write weights with the config as plain data.

    The config goes in as a dict, not a pickled object, so a checkpoint loads
    with ``weights_only=True`` anywhere, with no class to import first.
    """
    torch.save({"model": model.state_dict(), "config": asdict(model.cfg), **extra}, path)


def load_checkpoint(path, device: str = "cpu") -> tuple[ChessDT, dict]:
    """Load a checkpoint into a model in eval mode. Returns the model and the rest."""
    payload = torch.load(path, map_location=device, weights_only=True)
    model = ChessDT(ModelConfig(**payload["config"]))
    model.load_state_dict(payload["model"])
    return model.to(device).eval(), payload

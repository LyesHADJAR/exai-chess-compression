"""
The reference chess Decision Transformer.

Two stages. A board encoder reads the 64 squares of one position with
bidirectional attention and pools them into a single vector. The backbone is a
causal transformer over the interleaved stream

    [c_1, s_1, a_1, c_2, s_2, a_2, ..., c_K, s_K, a_K]

where c is the conditioning token (outcome-to-go and rating of the player to
move), s the encoded position and a the move played from it.

Module and parameter names are load-bearing: released checkpoints and every
compressed submission are keyed on them.
"""

import math
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import CONDITION_DIM, N_MOVES, SQUARE_CODES, ModelConfig


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


class Attention(nn.Module):
    """Multi-head self-attention, causal or bidirectional."""

    def __init__(self, dim: int, heads: int, dropout: float, causal: bool):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.dropout = dropout
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        q, k, v = (
            t.view(batch, tokens, self.heads, dim // self.heads).transpose(1, 2)
            for t in self.qkv(x).split(dim, dim=-1)
        )
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=self.causal, dropout_p=self.dropout if self.training else 0.0
        )
        return self.proj(out.transpose(1, 2).reshape(batch, tokens, dim))


class Block(nn.Module):
    """Pre-LN transformer block."""

    def __init__(self, dim: int, heads: int, ff: int, dropout: float, causal: bool):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout, causal)
        self.ffn = SwiGLU(dim, ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class BoardEncoder(nn.Module):
    """One position in, one vector out.

    Each square is a token: what stands on it plus where it is, with the side
    to move added to every square. Castling rights and the en passant target
    arrive as square codes, so the whole position is in the 64 tokens.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.code_embed = nn.Embedding(SQUARE_CODES, cfg.enc_dim)
        self.square_embed = nn.Embedding(64, cfg.enc_dim)
        self.side_embed = nn.Embedding(2, cfg.enc_dim)
        self.blocks = nn.ModuleList(
            Block(cfg.enc_dim, cfg.enc_heads, cfg.enc_ff, cfg.dropout, causal=False)
            for _ in range(cfg.enc_layers)
        )
        self.norm = nn.LayerNorm(cfg.enc_dim)
        self.proj = nn.Linear(cfg.enc_dim, cfg.dim)
        self.register_buffer("squares", torch.arange(64), persistent=False)

    def forward(self, codes: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        """codes (N, 64) and side (N,), both long. Returns (N, dim)."""
        x = (
            self.code_embed(codes)
            + self.square_embed(self.squares)
            + self.side_embed(side)[:, None]
        )
        for block in self.blocks:
            x = block(x)
        return self.proj(self.norm(x).mean(dim=1))


class ChessDT(nn.Module):
    """Decision Transformer over K plies of (condition, position, move).

    The move at ply t is read from the hidden state at the **position** token
    of ply t, never the move token. Under a causal mask the position token has
    seen the conditioning and position of ply t and everything before, but not
    the move it is being asked for; the move token is the embedding of that
    move, and reading there hands the model its own answer.
    `tests/test_model.py` holds the test that pins this down.
    """

    #: Offset of the position token inside each (condition, position, move) triple.
    READOUT_OFFSET = 1

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = BoardEncoder(cfg)
        self.cond_embed = nn.Linear(CONDITION_DIM, cfg.dim)
        self.move_embed = nn.Embedding(N_MOVES, cfg.dim)
        self.ply_embed = nn.Embedding(cfg.context_plies, cfg.dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            Block(cfg.dim, cfg.heads, cfg.ff, cfg.dropout, causal=True) for _ in range(cfg.layers)
        )
        self.norm = nn.LayerNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, N_MOVES)
        self.register_buffer("plies", torch.arange(cfg.context_plies), persistent=False)
        self._init_weights()

    def _init_weights(self) -> None:
        depth = self.cfg.enc_layers + self.cfg.layers
        for name, param in self.named_parameters():
            if param.dim() < 2:
                if name.endswith("bias"):
                    nn.init.zeros_(param)
                continue
            scaled = name.endswith(("attn.proj.weight", "ffn.w2.weight"))
            nn.init.normal_(param, std=0.02 / math.sqrt(2 * depth) if scaled else 0.02)

    def forward(
        self,
        codes: torch.Tensor,  # (B, K, 64) long   square codes
        side: torch.Tensor,  # (B, K)     long   1 where white is to move
        cond: torch.Tensor,  # (B, K, 2)  float  outcome-to-go, normalised rating
        moves: torch.Tensor,  # (B, K)     long   move played at each ply
    ) -> torch.Tensor:  # (B, K, N_MOVES) logits for the move at each ply
        batch, plies = side.shape
        if plies > self.cfg.context_plies:
            raise ValueError(f"{plies} plies exceeds the context of {self.cfg.context_plies}")

        state = self.encoder(codes.reshape(batch * plies, 64), side.reshape(-1))
        state = state.view(batch, plies, -1)
        ply = self.ply_embed(self.plies[:plies])

        stream = torch.stack(
            [self.cond_embed(cond) + ply, state + ply, self.move_embed(moves) + ply], dim=2
        ).view(batch, 3 * plies, -1)

        x = self.drop(stream)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        return self.head(x[:, self.READOUT_OFFSET :: 3])


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

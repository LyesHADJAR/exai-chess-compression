"""
Serving windows of the encoded corpus to the model.

The whole corpus is about a gigabyte, so it is loaded onto the training device
once and batches are cut from it by indexing. There is no DataLoader and no
worker pool: a batch is a handful of gathers on tensors that already live next
to the model.

A window is K consecutive plies starting anywhere in a game. Windows are
left-aligned and a window that runs off the end of its game is simply marked
invalid past that point. Under a causal mask no valid ply can see a later one,
so the overrun needs no attention mask, only exclusion from the loss.

The model is asked for the move at every ply of a window, not just the last,
so a single window trains it with zero plies of history at its first position
and K-1 at its last. That is what lets the same model be evaluated on a puzzle,
which arrives with no history at all.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .constants import (
    CASTLING_ROOK_CODES,
    ELO_CENTRE,
    ELO_SCALE,
    EN_PASSANT_CODE,
)
from .encoding import BOARD_RECORD_BYTES
from .splits import TRAIN

#: Home squares of the rooks, in the bit order the record stores castling rights.
_CASTLING_SQUARES = ((1, 7, 0), (2, 0, 0), (3, 63, 1), (4, 56, 1))  # bit, square, colour


def decode_records(records: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Packed records to model inputs.

    `records` is (..., BOARD_RECORD_BYTES) uint8. Returns square codes
    (..., 64) and side to move (...), both long. Castling rights and the en
    passant target are folded into the square codes here.
    """
    packed = records[..., :32].long()
    codes = torch.stack([packed & 0x0F, packed >> 4], dim=-1).flatten(-2)

    flags = records[..., 32].long()
    side = flags & 1

    for bit, square, colour in _CASTLING_SQUARES:
        has_right = ((flags >> bit) & 1).bool()
        codes[..., square] = torch.where(has_right, CASTLING_ROOK_CODES[colour], codes[..., square])

    ep_file = records[..., 33].long()
    has_ep = ep_file != 255
    # The target square sits behind the pawn that just moved: rank 6 when
    # white is to capture, rank 3 when black is.
    ep_square = torch.where(side == 1, 40, 16) + ep_file.clamp(max=7)
    ep_square = torch.where(has_ep, ep_square, 0).unsqueeze(-1)
    current = codes.gather(-1, ep_square)
    codes.scatter_(-1, ep_square, torch.where(has_ep.unsqueeze(-1), EN_PASSANT_CODE, current))

    return codes, side


def condition(outcome: torch.Tensor, elo: torch.Tensor) -> torch.Tensor:
    """The conditioning vector: outcome-to-go and normalised rating."""
    return torch.stack([outcome.float(), (elo.float() - ELO_CENTRE) / ELO_SCALE], dim=-1)


@dataclass
class Batch:
    codes: torch.Tensor  # (B, K, 64) long
    side: torch.Tensor  # (B, K) long
    cond: torch.Tensor  # (B, K, 2) float
    moves: torch.Tensor  # (B, K) long
    valid: torch.Tensor  # (B, K) bool — inside the window's own game
    labels: torch.Tensor  # (B, K) uint8 — split label of each position

    def inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.codes, self.side, self.cond, self.moves


class Corpus:
    """The encoded game corpus, resident on one device."""

    def __init__(self, directory: str | Path, device: str = "cpu"):
        directory = Path(directory)
        meta = np.load(directory / "games.npz")
        records = np.fromfile(directory / "positions.bin", dtype=np.uint8)
        records = records.reshape(-1, BOARD_RECORD_BYTES)
        moves = np.fromfile(directory / "moves.bin", dtype=np.uint16)
        labels = np.load(directory / "split_labels.npy")

        lengths = meta["length"].astype(np.int64)
        offsets = meta["offset"].astype(np.int64)
        if not (len(records) == len(moves) == len(labels) == int(lengths.sum())):
            raise ValueError(f"corpus at {directory} is inconsistent; rebuild it")

        white_to_move = (records[:, 32] & 1).astype(bool)
        result = np.repeat(meta["result"].astype(np.int8), lengths)
        outcome = np.where(white_to_move, result, -result).astype(np.int8)
        elo = np.where(
            white_to_move,
            np.repeat(meta["white_elo"], lengths),
            np.repeat(meta["black_elo"], lengths),
        ).astype(np.int16)

        self.device = device
        self.records = torch.from_numpy(records).to(device)
        self.moves = torch.from_numpy(moves.astype(np.int16)).to(device)
        self.labels = torch.from_numpy(labels).to(device)
        self.outcome = torch.from_numpy(outcome).to(device)
        self.elo = torch.from_numpy(elo).to(device)
        self.game_end = torch.from_numpy(np.repeat(offsets + lengths, lengths)).to(device)

        self.train_positions = torch.from_numpy(np.flatnonzero(labels == TRAIN)).to(device)
        self._offsets, self._lengths = offsets, lengths
        self._game_splits = np.load(directory / "game_splits.npy")

    def __len__(self) -> int:
        return len(self.records)

    def window(self, starts: torch.Tensor, plies: int) -> Batch:
        """Cut one window of `plies` plies from each start position."""
        index = starts[:, None] + torch.arange(plies, device=self.device)
        valid = index < self.game_end[starts][:, None]
        index = index.clamp(max=len(self) - 1)

        codes, side = decode_records(self.records[index])
        return Batch(
            codes=codes,
            side=side,
            cond=condition(self.outcome[index], self.elo[index]),
            moves=self.moves[index].long(),
            valid=valid,
            labels=self.labels[index],
        )

    def sample_train(self, batch: int, plies: int, generator: torch.Generator) -> Batch:
        """A random batch of training windows."""
        pick = torch.randint(
            len(self.train_positions), (batch,), generator=generator, device=self.device
        )
        return self.window(self.train_positions[pick], plies)

    def split_starts(self, split: int, plies: int, limit: int | None = None, seed: int = 0):
        """Window starts that tile every game of a split, end to end.

        A fixed, deterministic protocol: each game is cut into consecutive
        windows from its first ply. `limit` keeps a seeded subsample, for
        cheap periodic evaluation during training.
        """
        games = np.flatnonzero(self._game_splits == split)
        starts = np.concatenate(
            [np.arange(self._offsets[g], self._offsets[g] + self._lengths[g], plies) for g in games]
        )
        if limit is not None and len(starts) > limit:
            starts = np.sort(np.random.default_rng(seed).choice(starts, limit, replace=False))
        return torch.from_numpy(starts).to(self.device)

"""Neural typed-decision network of Primus Decision 0.1; each ensemble member is one instance.

    state tokens ──► Encoder (BiGRU | S4D state-space) ──► token states H (L, D)
    question ──► schema embedding + bag-of-words(instructions) ──► query q
    attention pooling(H, q) ──► context c
    option k ──► option embedding (schema, k) + bag-of-words(option text) ──► o_k
    logit_k = MLP([c, o_k, c ⊙ o_k])  ──► softmax over the K options *provided*

No cardinality is hard-coded: a question with K options yields a K-way softmax. Types are
handled by the same head; ``noul`` is a 2-option question whose second option is "true", and
``score`` questions additionally report the expected level.

No transformer, no attention between tokens, no pretrained encoder.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import Case, Decision, tokenize

PAD, UNK_BASE = 0, 1  # ids 1..1+n_hash are hashed OOV buckets; real vocab starts after


@dataclass
class Vocab:
    itos: list[str] = field(default_factory=list)
    stoi: dict[str, int] = field(default_factory=dict)
    n_hash: int = 2048

    @property
    def size(self) -> int:
        return 1 + self.n_hash + len(self.itos)

    def encode(self, text: str, max_len: int) -> list[int]:
        ids = []
        for tok in tokenize(text)[:max_len]:
            i = self.stoi.get(tok)
            if i is None:
                h = int(hashlib.blake2b(tok.encode(), digest_size=4).hexdigest(), 16) % self.n_hash
                ids.append(UNK_BASE + h)
            else:
                ids.append(UNK_BASE + self.n_hash + i)
        return ids or [UNK_BASE]

    @classmethod
    def from_json(cls, text: str) -> "Vocab":
        o = json.loads(text)
        v = cls(n_hash=o["n_hash"])
        v.itos = o["itos"]
        v.stoi = {w: i for i, w in enumerate(v.itos)}
        return v


class BiGRUEncoder(nn.Module):
    """Bidirectional GRU encoder (member1 of the released ensemble)."""

    def __init__(self, d_in: int, d_model: int, n_layers: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(d_in, d_model // 2, num_layers=n_layers, batch_first=True, bidirectional=True,
                          dropout=dropout if n_layers > 1 else 0.0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        return self.norm(out)


class S4DLayer(nn.Module):
    """Diagonal state-space layer (S4D, Gu et al. 2022) computed as an FFT convolution.

    Continuous system per channel d with N complex diagonal states:
        h'(t) = A h(t) + B x(t),  y(t) = Re(C h(t)) + D x(t)
    discretised with step Δ_d (ZOH on the diagonal). The SSM kernel K[d, l] = Re(C A_bar^l B_bar)
    is materialised for the sequence length and applied as a causal convolution. Linear
    time-invariant, so there is no token-to-token attention and no transformer component.
    """

    def __init__(self, d_model: int, n_state: int = 16, dt_min: float = 1e-3, dt_max: float = 1e-1):
        super().__init__()
        self.d_model, self.n_state = d_model, n_state
        # S4D-Lin initialisation: A_n = -1/2 + i π n
        log_dt = torch.rand(d_model) * (np.log(dt_max) - np.log(dt_min)) + np.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)
        self.log_a_real = nn.Parameter(torch.log(0.5 * torch.ones(d_model, n_state)))
        self.a_imag = nn.Parameter(np.pi * torch.arange(n_state).float().repeat(d_model, 1))
        c = torch.randn(d_model, n_state, 2) / np.sqrt(n_state)
        self.c = nn.Parameter(c)
        self.d_skip = nn.Parameter(torch.randn(d_model))

    def kernel(self, length: int) -> torch.Tensor:
        dt = torch.exp(self.log_dt).unsqueeze(-1)  # (D,1)
        a = -torch.exp(self.log_a_real) + 1j * self.a_imag  # (D,N)
        c = torch.view_as_complex(self.c)  # (D,N)
        dt_a = dt * a  # (D,N)
        # ZOH discretisation: B_bar = (exp(dtA) - 1)/A * B with B = 1
        b_bar = (torch.exp(dt_a) - 1.0) / a
        pos = torch.arange(length, device=dt.device)
        # K[d,l] = 2 * Re( sum_n c_n * b_bar_n * exp(dt a_n l) )
        exponent = dt_a.unsqueeze(-1) * pos  # (D,N,L)
        k = 2 * torch.einsum("dn,dnl->dl", c * b_bar, torch.exp(exponent)).real
        return k  # (D,L)

    def forward(self, x):  # x: (B,L,D)
        bsz, length, _ = x.shape
        k = self.kernel(length)  # (D,L)
        n_fft = 2 * length
        xf = torch.fft.rfft(x.transpose(1, 2), n=n_fft)  # (B,D,F)
        kf = torch.fft.rfft(k, n=n_fft)  # (D,F)
        y = torch.fft.irfft(xf * kf.unsqueeze(0), n=n_fft)[..., :length]  # (B,D,L)
        y = y + self.d_skip.view(1, -1, 1) * x.transpose(1, 2)
        return y.transpose(1, 2)


class S4DBlock(nn.Module):
    def __init__(self, d_model: int, n_state: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = S4DLayer(d_model, n_state)
        self.out = nn.Linear(d_model, 2 * d_model)  # GLU
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        z = self.norm(x)
        z = self.ssm(z * mask.unsqueeze(-1))
        z = F.glu(self.out(F.gelu(z)), dim=-1)
        return x + self.drop(z)


class S4DEncoder(nn.Module):
    """Bidirectional stack of diagonal state-space blocks (member0 of the released ensemble)."""

    def __init__(self, d_in: int, d_model: int, n_layers: int, dropout: float, n_state: int = 16):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.fwd = nn.ModuleList([S4DBlock(d_model, n_state, dropout) for _ in range(n_layers)])
        self.bwd = nn.ModuleList([S4DBlock(d_model, n_state, dropout) for _ in range(n_layers)])
        self.merge = nn.Linear(2 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, lengths):
        mask = (torch.arange(x.shape[1], device=x.device).unsqueeze(0) < lengths.unsqueeze(1)).float()
        h = self.proj(x)
        f = h
        for blk in self.fwd:
            f = blk(f, mask)
        # backward direction: reverse each sequence within its own length
        idx = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], -1)
        rev = (lengths.unsqueeze(1) - 1 - idx).clamp(min=0)
        b = torch.gather(h, 1, rev.unsqueeze(-1).expand(-1, -1, h.shape[-1]))
        for blk in self.bwd:
            b = blk(b, mask)
        b = torch.gather(b, 1, rev.unsqueeze(-1).expand(-1, -1, b.shape[-1]))
        return self.norm(self.merge(torch.cat([f, b], dim=-1)))


@dataclass
class NetConfig:
    encoder: str = "gru"  # gru | s4d
    vocab_size: int = 0
    d_emb: int = 128
    d_model: int = 192
    n_layers: int = 2
    n_state: int = 16
    n_heads: int = 4
    dropout: float = 0.3
    max_len: int = 768
    lsa_dims: int = 0  # >0: a dense LSA case vector is merged into the pooled context
    derived_relations: bool = True  # state text includes derived-relation sentences (data.py)
    schemas: list[str] = field(default_factory=list)
    schema_options: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def from_json(cls, text: str) -> "NetConfig":
        return cls(**json.loads(text))


class TypedDecisionNet(nn.Module):
    def __init__(self, cfg: NetConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_emb, padding_idx=PAD)
        self.emb_drop = nn.Dropout(cfg.dropout)
        if cfg.encoder == "gru":
            self.encoder = BiGRUEncoder(cfg.d_emb, d, cfg.n_layers, cfg.dropout)
        elif cfg.encoder == "s4d":
            self.encoder = S4DEncoder(cfg.d_emb, d, cfg.n_layers, cfg.dropout, cfg.n_state)
        else:
            raise ValueError(cfg.encoder)
        self.schema_emb = nn.Embedding(len(cfg.schemas), d)
        n_opt_slots = sum(len(v) for v in cfg.schema_options.values())
        self.option_emb = nn.Embedding(n_opt_slots, d)
        self.text_proj = nn.Linear(cfg.d_emb, d)
        self.query = nn.Linear(2 * d, cfg.n_heads * d)
        self.key = nn.Linear(d, cfg.n_heads * d)
        self.pool_out = nn.Linear(cfg.n_heads * d, d)
        self.lsa_proj = nn.Linear(cfg.lsa_dims, d) if cfg.lsa_dims > 0 else None
        self.ctx_merge = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(cfg.dropout)) if cfg.lsa_dims > 0 else None
        self.head = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(d, 1))

    def bag(self, ids: torch.Tensor) -> torch.Tensor:
        """Mean of token embeddings (ids padded with PAD)."""
        e = self.emb(ids)
        m = (ids != PAD).float().unsqueeze(-1)
        return self.text_proj((e * m).sum(1) / m.sum(1).clamp(min=1.0))

    def encode_states(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.emb_drop(self.emb(tokens))
        return self.encoder(x, lengths)  # (B,L,D)

    def pool(self, h: torch.Tensor, lengths: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Multi-head additive attention pooling of token states with a question query."""
        bsz, length, d = h.shape
        nh = self.cfg.n_heads
        keys = self.key(h).view(bsz, length, nh, d)
        queries = self.query(q).view(bsz, 1, nh, d)
        scores = (torch.tanh(keys + queries)).sum(-1) / np.sqrt(d)  # (B,L,nh)
        mask = torch.arange(length, device=h.device).unsqueeze(0) < lengths.unsqueeze(1)
        scores = scores.masked_fill(~mask.unsqueeze(-1), -1e4)
        attn = torch.softmax(scores, dim=1)  # over tokens
        pooled = torch.einsum("blh,bld->bhd", attn, h).reshape(bsz, nh * d)
        return self.pool_out(pooled)

    def forward(self, batch: dict) -> torch.Tensor:
        """Log-probabilities of shape (M, Kmax): one row per decision, masked to its K options."""
        h = self.encode_states(batch["tokens"], batch["lengths"])
        case_idx = batch["dec_case"]  # (M,) index of the case each decision belongs to
        schema_ids = batch["dec_schema"]  # (M,)
        instr = self.bag(batch["dec_instr"])  # (M,D)
        q = torch.cat([self.schema_emb(schema_ids), instr], dim=-1)
        h_sel = h[case_idx]  # (M,L,D)
        len_sel = batch["lengths"][case_idx]
        c = self.pool(h_sel, len_sel, q)  # (M,D)
        if self.lsa_proj is not None:
            lsa = self.lsa_proj(batch["case_feats"][case_idx])
            c = self.ctx_merge(torch.cat([c, lsa], dim=-1))
        opt_slots = batch["opt_slot"]  # (M,Kmax)
        opt_mask = batch["opt_mask"]  # (M,Kmax) bool
        o = self.option_emb(opt_slots) + self.bag(batch["opt_text"].view(-1, batch["opt_text"].shape[-1])).view(
            opt_slots.shape[0], opt_slots.shape[1], -1)
        ce = c.unsqueeze(1).expand(-1, o.shape[1], -1)
        logits = self.head(torch.cat([ce, o, ce * o], dim=-1)).squeeze(-1)  # (M,Kmax)
        logits = logits.masked_fill(~opt_mask, -1e4)
        return F.log_softmax(logits, dim=-1)


class Batcher:
    def __init__(self, vocab: Vocab, cfg: NetConfig, device="cpu", featurizer=None):
        self.vocab, self.cfg, self.device = vocab, cfg, device
        self.featurizer = featurizer
        self.schema_index = {s: i for i, s in enumerate(cfg.schemas)}
        self.option_slot = {}
        slot = 0
        for s in cfg.schemas:
            for k in cfg.schema_options[s]:
                self.option_slot[(s, k)] = slot
                slot += 1
        self._cache: dict[str, list[int]] = {}

    def _ids(self, text: str, max_len: int) -> list[int]:
        key = f"{max_len}:{text}"
        if key not in self._cache:
            if len(self._cache) >= 4096:  # bounded: a long-running bridge sees an unbounded stream of distinct states
                self._cache.clear()
            self._cache[key] = self.vocab.encode(text, max_len)
        return self._cache[key]

    def make(self, cases: list[Case]):
        tok_lists = [self._ids(c.state_text, self.cfg.max_len) for c in cases]
        lengths = torch.tensor([len(t) for t in tok_lists])
        tokens = torch.full((len(cases), int(lengths.max())), PAD, dtype=torch.long)
        for i, t in enumerate(tok_lists):
            tokens[i, : len(t)] = torch.tensor(t)
        decisions: list[Decision] = []
        dec_case, dec_schema, instr_ids, opt_slots, opt_texts = [], [], [], [], []
        for ci, c in enumerate(cases):
            for d in c.decisions:
                decisions.append(d)
                dec_case.append(ci)
                if d.schema_id not in self.schema_index:
                    raise ValueError(f"unknown question schema {d.schema_id!r}: questions must use one of the workflow/question-id pairs listed under 'schemas' in the model config")
                dec_schema.append(self.schema_index[d.schema_id])
                instr_ids.append(self._ids(d.instructions, 64))
                opt_slots.append([self.option_slot[(d.schema_id, k)] for k in d.option_keys])
                opt_texts.append([self._ids(t, 48) for t in d.option_texts])
        kmax = max(len(s) for s in opt_slots)
        tmax = max(len(t) for ts in opt_texts for t in ts)
        imax = max(len(t) for t in instr_ids)
        m = len(decisions)
        instr = torch.full((m, imax), PAD, dtype=torch.long)
        slots = torch.zeros((m, kmax), dtype=torch.long)
        omask = torch.zeros((m, kmax), dtype=torch.bool)
        otext = torch.full((m, kmax, tmax), PAD, dtype=torch.long)
        for i in range(m):
            instr[i, : len(instr_ids[i])] = torch.tensor(instr_ids[i])
            for k, s in enumerate(opt_slots[i]):
                slots[i, k] = s
                omask[i, k] = True
                t = opt_texts[i][k]
                otext[i, k, : len(t)] = torch.tensor(t)
        batch = {"tokens": tokens, "lengths": lengths, "dec_case": torch.tensor(dec_case), "dec_schema": torch.tensor(dec_schema),
                 "dec_instr": instr, "opt_slot": slots, "opt_mask": omask, "opt_text": otext}
        if self.cfg.lsa_dims > 0:
            if self.featurizer is None:
                raise ValueError("config has lsa_dims > 0 but no featurizer was provided")
            batch["case_feats"] = torch.from_numpy(self.featurizer.transform(cases))
        return decisions, {k: v.to(self.device) for k, v in batch.items()}

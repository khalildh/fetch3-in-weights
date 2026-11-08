"""Integer-only forward pass of the ternary transformer: the exact reference (the spec).

Every value at run time is an integer; fixed-point scales are powers of two (`Qf` means
value = int / 2^f) or an integer multiplier followed by a rounding right shift. Floats
appear only in `quantize_model` / `tables`, which build the integer model offline from the
trained float checkpoint (weights, cos/sin and SiLU tables, exp polynomial coefficients),
and in `float_forward`, the float model used to measure the gap. docs/integer_spec.md is
the prose version of this file; tern/intops.py (interpreter) and tern/rt_int.c (compiled)
must agree with it bit for bit.

Formats (see FORMATS):
  residual stream x, norm inputs       int32 Q16
  BitLinear input                      int8 q + per-vector unit a (int, Q24): value = q * a
  ternary weight                       trits + per-tensor scale (int, Q24)
  q, k, v, gate, up (BitLinear out)    int16 Q11 (saturating)
  o, down (BitLinear out)              int32 Q16 (added to the residual, saturating)
  norm gains                           int32 Q16
  RoPE cos / sin table                 int16 Q14
  attention scores                     int32 Q16 (q.k / sqrt(head_dim))
  exp, softmax probabilities           Q30 (int64), Q15
  SiLU                                 int16 Q11 -> int16 Q11 lookup table (65536 entries)
  embedding / tied head                int8 + per-row scale (int, Q32)
  logits                               int32 Q16

Rounding everywhere is `rshr` (add half, arithmetic shift right: round half up), and
saturation is to the stated width. Integer division is floor division of non-negative
numbers. All intermediate products are bounded below 2^63 (docs/integer_spec.md).

Arrays are numpy int64; functions accept a leading batch of rows (`...`) where it makes
sense so the same code runs one token (generation) or a whole sequence (validation loss).
Large matrix products may use float64 BLAS as an *exact* integer engine (`_mm`): every
operand and partial sum is an integer below 2^53, which float64 represents exactly in any
summation order; `exact=True` uses int64 instead, and the tests check both agree.
"""

import math

import numpy as np

# ------------------------------------------------------------------ formats

F_X = 16      # residual stream, norm inputs, o / down outputs, attention output (int32)
F_A = 24      # unit of an int8 BitLinear input (per vector)
F_W = 24      # per-tensor ternary weight scale
F_H = 11      # q, k, v, gate, up (int16)
F_G = 16      # norm gains
F_ROPE = 14   # cos / sin
F_S = 16      # attention scores
F_EXP = 30    # exp(r), r in (-ln 2, 0]
F_P = 15      # softmax probabilities
F_E = 32      # per-row embedding scale
F_L = 16      # logits
F_ISQ = 15    # 1 / sqrt(head_dim)
EPS = 1e-5    # RMSNorm epsilon (as trained)
EPS2 = round(EPS * 2 ** (2 * F_X))  # epsilon in squared residual units (Q32): 42950
NORM_BITS = 24  # sum of squares is taken over x >> s with |x >> s| < 2^24
RSQ_FRAC = 7    # extra fraction bits of the integer RMS
P_MAX = 1024    # RoPE table positions
C_MAX = 1 << 44  # BitLinear multiplier bound (|acc| < 2^19, so |acc * c| < 2^63)
ACC_MAX = 1 << 19
MUL_SHIFT = 2 * F_H - F_X  # silu(gate) * up: Q22 -> Q16
LN2_S = round(math.log(2) * 2**F_S)  # 45426
EXP_DEGREE = 3

FORMATS = dict(F_X=F_X, F_A=F_A, F_W=F_W, F_H=F_H, F_G=F_G, F_ROPE=F_ROPE, F_S=F_S, F_EXP=F_EXP, F_P=F_P,
               F_E=F_E, F_L=F_L, F_ISQ=F_ISQ, EPS2=EPS2, NORM_BITS=NORM_BITS, RSQ_FRAC=RSQ_FRAC,
               MUL_SHIFT=MUL_SHIFT, LN2_S=LN2_S)

I64 = np.int64


def _exp_coeffs(degree=EXP_DEGREE):
    """Integer (Q30) coefficients of a polynomial for exp(r), r in [-ln 2, 0], highest first.
    Built offline: least squares on Chebyshev nodes (close to minimax). Fixed constants of
    the spec, duplicated in tern/rt_int.c (checked by tests/test_int_forward.py)."""
    n = 64
    t = np.cos((2 * np.arange(n) + 1) * np.pi / (2 * n))
    r = (t - 1) * math.log(2) / 2
    c = np.polyfit(r, np.exp(r), degree)
    return [int(round(v * 2**F_EXP)) for v in c]


EXP_COEFFS = [127494503, 515997870, 1070942615, 1073682383]  # == _exp_coeffs(3), pinned

# ------------------------------------------------------------------ primitives


def rshr(v, s):
    """Round-half-up arithmetic right shift: (v + 2^(s-1)) >> s, and v for s = 0. s may be an array."""
    v = np.asarray(v, dtype=I64)
    s = np.asarray(s, dtype=I64)
    half = np.where(s > 0, np.left_shift(I64(1), np.maximum(s - 1, 0)), 0)
    return np.right_shift(v + half, s)


def sat(v, bits):
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return np.clip(np.asarray(v, dtype=I64), lo, hi)


def bitlen(a):
    """Number of bits of each non-negative int64 (0 for 0), integer-only."""
    a = np.asarray(a, dtype=I64).copy()
    n = np.zeros(a.shape, dtype=I64)
    for k in (32, 16, 8, 4, 2, 1):
        big = a >= (I64(1) << k)
        n += np.where(big, k, 0)
        a = np.where(big, a >> k, a)
    return n + (a > 0)


def isqrt(a):
    """floor(sqrt(a)) of non-negative int64s < 2^62: integer Newton from above."""
    a = np.asarray(a, dtype=I64)
    x = np.left_shift(I64(1), (bitlen(a) + 1) // 2)  # >= sqrt(a)
    x = np.where(a == 0, 0, x)
    while True:
        y = np.where(x > 0, (x + a // np.maximum(x, 1)) >> 1, 0)
        done = y >= x
        if done.all():
            return x
        x = np.where(done, x, y)


def _mm(a, b, exact=False):
    """a @ b for integer arrays whose products and partial sums stay below 2^53 in magnitude."""
    if exact:
        return np.matmul(a.astype(I64), b.astype(I64))
    return np.matmul(a.astype(np.float64), b.astype(np.float64)).astype(I64)


# ------------------------------------------------------------------ operations


def inorm(x, g):
    """Fused RMSNorm + absmax int8 quantization (the input of every BitLinear and the head).

    x: (..., n) int32 Q16, g: (n,) int32 Q16 gains. Returns (q, a): q (..., n) int8 values,
    a (...,) Q24 so that rmsnorm(x) * g ~= q * a / 2^24. The 1/rms factor is scale-only, so
    q depends on v = x * g alone; 1/rms enters a (one integer division per vector)."""
    x = np.asarray(x, dtype=I64)
    n = x.shape[-1]
    v = sat(rshr(x * np.asarray(g, dtype=I64), F_G), 32)
    m = np.abs(v).max(axis=-1)
    # RMS: sum of squares of x >> s, with s making |x >> s| < 2^24 (no overflow for n < 2^14)
    s = np.maximum(bitlen(np.abs(x).max(axis=-1)) - NORM_BITS, 0)
    zx = rshr(x, s[..., None])
    ms = (zx * zx).sum(axis=-1) // n + (EPS2 >> (2 * s))
    r = isqrt(ms << (2 * RSQ_FRAC))  # rms * 2^7, in units of 2^(s - 16)
    # a = (m / 2^16) / (127 * rms) * 2^24 = m * 2^31 / (127 * r * 2^s)
    a = np.where(m > 0, (m << (F_A + RSQ_FRAC)) // ((127 * r) << s), 0)
    R = np.where(m > 0, (I64(127) << 32) // np.maximum(m, 1), 0)  # 127 / m in Q32
    q = np.sign(v) * ((np.abs(v) * R[..., None] + (I64(1) << 31)) >> 32)
    return q, a


def irescale(acc, c, shift, bits):
    """BitLinear output: sat_bits(rshr(acc * c, shift)), c = W * a (Q48), 0 <= c < 2^44."""
    acc = np.asarray(acc, dtype=I64)
    c = np.asarray(c, dtype=I64)
    if np.any(c < 0) or np.any(c >= C_MAX):
        raise ValueError("irescale: multiplier out of range")
    if np.any(np.abs(acc) >= ACC_MAX):
        raise ValueError("irescale: accumulator out of range")
    c = c[..., None] if c.ndim else c
    return sat(rshr(acc * c, shift), bits)


def bitlinear(codes, W, shift, bits, q, a, exact=False):
    """int8 q (..., n) x ternary codes (m, n) -> int32 acc -> rescaled output."""
    acc = _mm(q, np.asarray(codes).T, exact)
    return irescale(acc, np.asarray(W, dtype=I64) * a, shift, bits)


def irope(x, pos, n_heads, table):
    """Rotary embedding on Q11 int16 (..., D) at positions pos (...,), rotating halves per head.
    table: (P_MAX, hd/2, 2) int16 Q14 cos / sin."""
    x = np.asarray(x, dtype=I64)
    D = x.shape[-1]
    hd = D // n_heads
    half = hd // 2
    v = x.reshape(*x.shape[:-1], n_heads, hd)
    t = np.asarray(table, dtype=I64).reshape(-1, half, 2)[np.asarray(pos)]
    c, s = t[..., None, :, 0], t[..., None, :, 1]
    x1, x2 = v[..., :half], v[..., half:2 * half]
    o = np.concatenate([rshr(x1 * c - x2 * s, F_ROPE), rshr(x1 * s + x2 * c, F_ROPE)], axis=-1)
    return sat(o, 16).reshape(x.shape)


def iexp(d):
    """exp(d / 2^16) in Q30 for d <= 0 (int): d = r - z ln2, exp = poly(r) >> z (rounded)."""
    d = np.asarray(d, dtype=I64)
    z = (-d) // LN2_S
    r = d + z * LN2_S  # in (-ln2, 0] (Q16)
    y = np.full(d.shape, EXP_COEFFS[0], dtype=I64)
    for c in EXP_COEFFS[1:]:
        y = rshr(y * r, F_S) + c
    return np.where(z > 40, 0, rshr(y, np.minimum(z, 40)))


def isoftmax(s, mask=None):
    """Softmax over the last axis of Q16 scores -> Q15 probabilities (int)."""
    s = np.asarray(s, dtype=I64)
    if mask is not None:
        s = np.where(mask, s, np.iinfo(I64).min // 4)
    mx = s.max(axis=-1, keepdims=True)
    e = iexp(s - mx)
    if mask is not None:
        e = np.where(mask, e, 0)
    S = e.sum(axis=-1, keepdims=True)  # >= 2^30 (the max contributes exp(0))
    return ((e << F_P) + (S >> 1)) // S


def iscores(q, k, hd, exact=False):
    """(..., hd) q and k Q11 -> q.k / sqrt(hd) in Q16. Dot products are < 2^36."""
    isq = math.isqrt((1 << (2 * F_ISQ)) // hd)  # floor(2^15 / sqrt(hd)), integer-only (4096 for hd = 64)
    return rshr(_mm(q, np.swapaxes(k, -1, -2), exact) * isq, 2 * F_H + F_ISQ - F_S)


def iattend(q, K, V, n_heads, exact=False):
    """One query (D,) over cached keys / values (T, D), all Q11 -> attention output (D,) Q16."""
    q, K, V = (np.asarray(a, dtype=I64) for a in (q, K, V))
    T, D = K.shape
    hd = D // n_heads
    qh = q.reshape(n_heads, 1, hd)
    kh = K.reshape(T, n_heads, hd).transpose(1, 0, 2)
    vh = V.reshape(T, n_heads, hd).transpose(1, 0, 2)
    p = isoftmax(iscores(qh, kh, hd, exact))  # (H, 1, T)
    o = _mm(p, vh, exact)  # (H, 1, hd), Q26, |o| < 2^31
    return sat(rshr(o, F_P + F_H - F_X), 32).reshape(D)


def iattend_seq(Q, K, V, n_heads, exact=False):
    """Causal attention over a whole sequence (T, D): row t equals iattend(Q[t], K[:t+1], V[:t+1])."""
    Q, K, V = (np.asarray(a, dtype=I64) for a in (Q, K, V))
    T, D = Q.shape
    hd = D // n_heads
    qh, kh, vh = (a.reshape(T, n_heads, hd).transpose(1, 0, 2) for a in (Q, K, V))
    mask = np.tril(np.ones((T, T), dtype=bool))
    p = isoftmax(iscores(qh, kh, hd, exact), mask)
    o = _mm(p, vh, exact)
    return sat(rshr(o, F_P + F_H - F_X), 32).transpose(1, 0, 2).reshape(T, D)


def isilu(x, table):
    return np.asarray(table, dtype=I64)[sat(x, 16) + 32768]


def imul(a, b, shift):
    return sat(rshr(np.asarray(a, dtype=I64) * np.asarray(b, dtype=I64), shift), 32)


def iadd(a, b):
    return sat(np.asarray(a, dtype=I64) + np.asarray(b, dtype=I64), 32)


def iembed(E8, Escale, tokens):
    return sat(rshr(E8[tokens].astype(I64) * Escale[tokens][..., None], F_E - F_X), 32)


def ihead(E8, Escale, q, a, exact=False):
    """Tied int8 head: logits (..., V) Q16 = (E8 . q) * e_row * a."""
    dot = _mm(q, E8.T, exact)  # |dot| < 2^24
    c = rshr(Escale.astype(I64) * np.asarray(a, dtype=I64)[..., None], F_A)  # Q32
    return sat(rshr(dot * c, F_E - F_L), 32)


# ------------------------------------------------------------------ the integer model

LAYER_LINEARS = {"q": "attn.q_proj", "k": "attn.k_proj", "v": "attn.v_proj", "o": "attn.o_proj",
                 "gate": "mlp.gate_proj", "up": "mlp.up_proj", "down": "mlp.down_proj"}
LAYER_NORMS = {"attn_norm": "attn_norm", "attn_subln": "attn.subln", "mlp_norm": "mlp_norm",
               "mlp_subln": "mlp.subln"}
OUT_FORMAT = {"q": (F_H, 16), "k": (F_H, 16), "v": (F_H, 16), "gate": (F_H, 16), "up": (F_H, 16),
              "o": (F_X, 32), "down": (F_X, 32)}  # (fraction bits, saturation width)


def tables(head_dim):
    """Offline-built tables: RoPE cos/sin (P_MAX, hd/2, 2) Q14 and SiLU over int16 Q11."""
    half = head_dim // 2
    theta = np.arange(P_MAX)[:, None] * 10000.0 ** (-np.arange(half) * 2.0 / head_dim)
    rope = np.stack([np.round(np.cos(theta) * 2**F_ROPE), np.round(np.sin(theta) * 2**F_ROPE)], -1)
    xs = np.arange(-32768, 32768) / 2**F_H
    silu = np.round(xs / (1 + np.exp(-xs)) * 2**F_H)
    return rope.astype(np.int16), sat(silu, 16).astype(np.int16)


def quantize_model(z, embed_mode="row"):
    """Integer model (dict of int arrays) from a float export (phase0/check_forward.py npz).
    embed_mode: "row" (int8 + per-row scale) or "tensor" (one scale)."""
    E = np.asarray(z["embed.weight"], dtype=np.float64)
    amax = np.abs(E).max(axis=1) if embed_mode == "row" else np.full(E.shape[0], np.abs(E).max())
    e = np.maximum(amax, 1e-12) / 127
    n_heads = int(z["n_heads"])
    rope, silu = tables(E.shape[1] // n_heads)
    m = {"n_layers": np.array(int(z["n_layers"])), "n_heads": np.array(n_heads),
         "embed.q": np.clip(np.round(E / e[:, None]), -127, 127).astype(np.int8),
         "embed.scale": np.round(e * 2**F_E).astype(I64),
         "norm": np.round(np.asarray(z["norm.weight"], np.float64) * 2**F_G).astype(np.int32),
         "rope": rope.reshape(-1), "silu": silu}  # rope flattened: (pos * hd/2 + i) * 2 + {0: cos, 1: sin}
    for i in range(int(z["n_layers"])):
        for k, name in LAYER_NORMS.items():
            m[f"layers.{i}.{k}"] = np.round(np.asarray(z[f"layers.{i}.{name}.weight"], np.float64) * 2**F_G).astype(np.int32)
        for k, name in LAYER_LINEARS.items():
            frac, bits = OUT_FORMAT[k]
            m[f"layers.{i}.{k}.codes"] = np.asarray(z[f"layers.{i}.{name}.codes"]).astype(np.int8)
            m[f"layers.{i}.{k}.scale"] = np.array(round(float(z[f"layers.{i}.{name}.scale"]) * 2**F_W), dtype=I64)
            m[f"layers.{i}.{k}.shift"] = np.array(F_W + F_A - frac, dtype=I64)
            m[f"layers.{i}.{k}.bits"] = np.array(bits, dtype=I64)
    return m


class IntModel:
    def __init__(self, m):
        self.m = {k: np.asarray(v) for k, v in m.items()}
        self.n_layers, self.n_heads = int(self.m["n_layers"]), int(self.m["n_heads"])
        self.E8 = self.m["embed.q"]
        self.Escale = self.m["embed.scale"].astype(I64)
        self.D = self.E8.shape[1]

    def __getitem__(self, k):
        return self.m[k]

    def linear(self, i, k, q, a, exact=False):
        p = f"layers.{i}.{k}"
        return bitlinear(self.m[p + ".codes"], self.m[p + ".scale"], int(self.m[p + ".shift"]),
                         int(self.m[p + ".bits"]), q, a, exact)


def forward_seq(M, tokens, exact=False):
    """Integer logits (T, V) Q16 for a whole sequence (causal); row t equals step(t)."""
    tokens = np.asarray(tokens)
    T = len(tokens)
    pos = np.arange(T)
    x = iembed(M.E8, M.Escale, tokens)
    for i in range(M.n_layers):
        L = lambda k: M[f"layers.{i}.{k}"]  # noqa: E731
        q8, a = inorm(x, L("attn_norm"))
        Q = irope(M.linear(i, "q", q8, a, exact), pos, M.n_heads, M["rope"])
        K = irope(M.linear(i, "k", q8, a, exact), pos, M.n_heads, M["rope"])
        V = M.linear(i, "v", q8, a, exact)
        att = iattend_seq(Q, K, V, M.n_heads, exact)
        x = iadd(x, M.linear(i, "o", *inorm(att, L("attn_subln")), exact))
        q8, a = inorm(x, L("mlp_norm"))
        g = imul(isilu(M.linear(i, "gate", q8, a, exact), M["silu"]), M.linear(i, "up", q8, a, exact), MUL_SHIFT)
        x = iadd(x, M.linear(i, "down", *inorm(g, L("mlp_subln")), exact))
    return ihead(M.E8, M.Escale, *inorm(x, M["norm"]), exact)


class Decoder:
    """One token at a time with a KV cache (what examples/forward_int.tern does). steer: activation steering
    (lens/steer.py), None or {layer i: (D,) int64}: added (saturated) to the residual after layer i while it is set."""

    def __init__(self, M):
        self.M = M
        self.K = [[] for _ in range(M.n_layers)]
        self.V = [[] for _ in range(M.n_layers)]
        self.pos = 0
        self.steer = None

    def step(self, token, exact=True):
        M, pos = self.M, self.pos
        x = iembed(M.E8, M.Escale, np.array(token))
        for i in range(M.n_layers):
            L = lambda k: M[f"layers.{i}.{k}"]  # noqa: E731
            q8, a = inorm(x, L("attn_norm"))
            self.K[i].append(irope(M.linear(i, "k", q8, a, exact), pos, M.n_heads, M["rope"]))
            self.V[i].append(M.linear(i, "v", q8, a, exact))
            q = irope(M.linear(i, "q", q8, a, exact), pos, M.n_heads, M["rope"])
            att = iattend(q, np.stack(self.K[i]), np.stack(self.V[i]), M.n_heads, exact)
            x = iadd(x, M.linear(i, "o", *inorm(att, L("attn_subln")), exact))
            q8, a = inorm(x, L("mlp_norm"))
            g = imul(isilu(M.linear(i, "gate", q8, a, exact), M["silu"]), M.linear(i, "up", q8, a, exact), MUL_SHIFT)
            x = iadd(x, M.linear(i, "down", *inorm(g, L("mlp_subln")), exact))
            if self.steer is not None and i in self.steer:
                x = iadd(x, self.steer[i])
        self.pos += 1
        return ihead(M.E8, M.Escale, *inorm(x, M["norm"]), exact)


def generate(M, prompt, n_new):
    """Greedy decode: (tokens (n_new,), logits (n_new, V)) where logits[j] produced tokens[j]."""
    dec = Decoder(M)
    for t in prompt:
        logits = dec.step(int(t))
    out, seen = [], []
    while len(out) < n_new:
        seen.append(logits)
        out.append(int(np.argmax(logits)))  # first maximum, like tern's argmax
        if len(out) < n_new:
            logits = dec.step(out[-1])
    return np.array(out), np.stack(seen)


# ------------------------------------------------------------------ float model (measurement only)


def float_forward(z, tokens):
    """Float64 logits (T, V) of the trained model (same math as tern/tensors.py and MLX)."""
    E = np.asarray(z["embed.weight"], dtype=np.float64)
    H = int(z["n_heads"])
    T = len(tokens)

    def rms(x, w):
        return x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS) * w

    def bl(name, x):
        s = 127.0 / np.maximum(np.abs(x).max(-1, keepdims=True), EPS)
        return (np.clip(np.round(x * s), -128, 127) @ z[name + ".codes"].astype(np.float64).T) * float(z[name + ".scale"]) / s

    def rope(x):
        hd = x.shape[-1]
        half = hd // 2
        th = np.arange(T)[:, None] * 10000.0 ** (-np.arange(half) * 2.0 / hd)
        c, s = np.cos(th)[:, None], np.sin(th)[:, None]
        return np.concatenate([x[..., :half] * c - x[..., half:] * s, x[..., :half] * s + x[..., half:] * c], -1)

    x = E[np.asarray(tokens)]
    mask = np.triu(np.full((T, T), -np.inf), 1)
    for i in range(int(z["n_layers"])):
        p = f"layers.{i}"
        h = rms(x, z[p + ".attn_norm.weight"])
        q, k, v = (bl(f"{p}.attn.{n}_proj", h).reshape(T, H, -1) for n in "qkv")
        q, k = rope(q), rope(k)
        sc = np.einsum("thd,shd->hts", q, k) / np.sqrt(q.shape[-1]) + mask
        w = np.exp(sc - sc.max(-1, keepdims=True))
        w /= w.sum(-1, keepdims=True)
        a = np.einsum("hts,shd->thd", w, v).reshape(T, -1)
        x = x + bl(p + ".attn.o_proj", rms(a, z[p + ".attn.subln.weight"]))
        h = rms(x, z[p + ".mlp_norm.weight"])
        g = bl(p + ".mlp.gate_proj", h)
        x = x + bl(p + ".mlp.down_proj", rms(g / (1 + np.exp(-g)) * bl(p + ".mlp.up_proj", h), z[p + ".mlp.subln.weight"]))
    return rms(x, z["norm.weight"]) @ E.T


def cross_entropy(logits, targets):
    """Mean cross-entropy (nats) of float logits; for measurement only."""
    lg = np.asarray(logits, dtype=np.float64)
    lg = lg - lg.max(-1, keepdims=True)
    lp = lg - np.log(np.exp(lg).sum(-1, keepdims=True))
    return float(-lp[np.arange(len(targets)), targets].mean())

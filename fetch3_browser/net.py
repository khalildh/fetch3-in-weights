"""The compiled filesystem net (examples/wire/fs_net.tern, exported) run on token streams, with the model's integer
softmax (tern/intref.py iscores / isoftmax / iexp) for every attention head (docs/tfs.md).

Python because it is the glue to the integer models' Python decoders and the chat engine; the net itself is built,
and its silence proved, in tern.

    net = Net.load(dir)                    # a directory exported by fs_net.tern
    out = net.forward(tokens)              # teacher forcing: per position the output code, fire bit, named wires
    dec = net.decoder(); dec.step(tok)     # incremental, for generation (KV caches per head)

Every read is checked exact: the winning key has probability exactly 2^15 (every other key's iexp is 0), so the head's
output is exactly 32 x the winner's value. Where a head's read is not used (an abs head whose target is past the
current position), it may be inexact; the cases never read it there (proved in fs_net.tern: the gates are 0).
"""

import subprocess
import sys
from pathlib import Path

import numpy as np

from . import integer as R
from ._paths import ROOT

from . import ref

I64 = np.int64
B = 1 << 24          # the readout's scale (Q16 logits): a forced token leads every other token by >= 2B


def export(out_dir, v0, nb=None):
    """Build and export the net for base vocabulary v0 (runs fs_net.tern in the interpreter, a few seconds); nb: a
    wider code than the block's own (a merged vocabulary that goes past it, docs/tfs_fetch3.md)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([sys.executable, "-m", "tern", str(ROOT / "examples/wire/fs_net.tern"), str(v0), str(out_dir),
                        *([str(nb)] if nb else [])],
                       capture_output=True, text=True, cwd=ROOT, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-4000:])
    return r.stdout


def _sparse(flat, k):
    a = np.asarray(flat, I64).reshape(-1, k)
    return [a[:, i] for i in range(k)]


class Head:
    def __init__(self, name, depth, q, k, v, o, tgt):
        self.name, self.depth = name, depth
        self.q, self.k = q, k                   # (coord, wire, coef) arrays
        self.nq = int(max(q[0].max(), k[0].max())) + 1
        m = 0
        while 4 ** m < self.nq:
            m += 1
        self.hd = 4 ** m                        # padded so that 1/sqrt(hd) is a power of two: scores exact
        self.scale = 1 << (6 + m)               # q * 2^(6+m): rshr(q.k * isq, 21) = the score in units
        self.v, self.o = v, o                   # value wires, output wires
        self.tgt = tgt                          # abs heads: the digit wires of the target

    def form(self, terms, vals):
        """(T, hd) from (coord, wire, coef) terms over vals (n_wires, T)."""
        c, w, a = terms
        out = np.zeros((self.hd, vals.shape[1]), I64)
        np.add.at(out, c, a[:, None] * vals[w])
        return out.T


class Net:
    def __init__(self, d):
        d = Path(d)
        meta = np.load(d / "meta.npy")
        (self.n, self.nb, self.v0, self.S, nh, self.VB, self.VSTRIDE, self.TREAL, self.GAP, self.NF,
         self.NS) = (int(x) for x in meta)
        self.lay = ref.Layout(self.v0, self.nb)
        assert self.lay.nb == self.nb and self.S == ref.S
        self.fnames = (d / "feat.txt").read_text().split()
        self.fw = np.load(d / "feat_w.npy")
        self.pnames = (d / "posf.txt").read_text().split()
        self.pw = np.load(d / "posf_w.npy")
        self.cw, self.cv = np.load(d / "const_w.npy"), np.load(d / "const_v.npy")
        self.op_w, self.op_d = np.load(d / "op_w.npy"), np.load(d / "op_d.npy")
        self.op_l = _sparse(np.load(d / "op_l.npy"), 3)
        self.op_g = _sparse(np.load(d / "op_g.npy"), 3)
        names = (d / "heads.txt").read_text().split()
        hd = np.load(d / "head_d.npy")
        hq, hk = _sparse(np.load(d / "head_q.npy"), 4), _sparse(np.load(d / "head_k.npy"), 4)
        hv, ho = _sparse(np.load(d / "head_v.npy"), 3), _sparse(np.load(d / "head_o.npy"), 3)
        ht = np.load(d / "head_t.npy").reshape(-1, 4)
        tg = {int(r[0]): r[1:] for r in ht}
        self.heads = []
        for h in range(nh):
            sel = lambda a, m: [x[a[0] == h] for x in a[1:]][:m]  # noqa: E731
            self.heads.append(Head(names[h], int(hd[h]), sel(hq, 3), sel(hk, 3), sel(hv, 2)[1], sel(ho, 2)[1],
                                   tg.get(h)))
        ow = np.load(d / "out_w.npy")
        self.out_w, self.notfire_w, self.fire_w = ow[:self.nb], int(ow[self.nb]), int(ow[self.nb + 1])
        self.names = dict(zip((d / "names.txt").read_text().split(), (int(x) for x in np.load(d / "names_w.npy"))))
        self.depth = int(max(self.op_d.max(), hd.max()))
        # ops by depth: (op rows, left (row, wire, coef), gate (row, wire, coef)) with rows local to the depth
        self.by_depth = []
        for dd in range(self.depth + 1):
            rows = np.nonzero(self.op_d == dd)[0]
            loc = np.full(len(self.op_w), -1, I64)
            loc[rows] = np.arange(len(rows))
            ml, mg = np.isin(self.op_l[0], rows), np.isin(self.op_g[0], rows)
            self.by_depth.append((self.op_w[rows], (loc[self.op_l[0][ml]], self.op_l[1][ml], self.op_l[2][ml]),
                                  (loc[self.op_g[0][mg]], self.op_g[1][mg], self.op_g[2][mg]),
                                  [h for h in self.heads if h.depth == dd]))
        self.ftable = self._features()
        self.readout = self._readout()

    @staticmethod
    def load(d):
        return Net(d)

    # ---- tables -----------------------------------------------------------------------------------------------

    def _features(self):
        """The token embedding: (V, n_feat) int64, by feature name (fs_net.tern new_graph)."""
        lay, V = self.lay, self.lay.V
        F = np.zeros((V, len(self.fnames)), I64)
        col = {n: i for i, n in enumerate(self.fnames)}
        ids = np.arange(V)
        for b in range(self.nb):
            F[:, col[f"code{b}"]] = 2 * ((ids >> b) & 1) - 1
        for t in range(lay.v0, V):
            k, kx = lay.kind(t), lay.kidx(t)
            flags = {"MARK"}
            if k in ref.CALLS:
                flags.add("CALL")
            if k in ("text", "mount"):
                flags.add("TEXT")
            if k in ("new", "ok"):
                flags.add("ACK")
                F[t, col[f"ACKOH{kx}"]] = 1
            if k == "new":
                F[t, col[f"NEWOH{kx}"]] = 1
            if k == "del":
                F[t, col[f"DELOH{kx}"]] = 1
            flags.add(k.upper())
            for f in flags:
                F[t, col[f]] = 1
            F[t, col["KIDX"]] = kx
        return F

    def pos_features(self, j):
        """The position table's rows for positions j (array): pos, bos, and per offset o the base-32 digits of
        max(j + o, 0) with their squares and the sum of squares."""
        j = np.asarray(j, I64)
        out = np.zeros((len(j), len(self.pnames)), I64)
        for i, n in enumerate(self.pnames):
            if n == "pos":
                out[:, i] = j
            elif n == "bos":
                out[:, i] = j == 0
            else:
                o, dg = n[1:].split("_")
                o = -int(o[1:]) if o.startswith("m") else int(o)
                x = np.maximum(j + o, 0)
                a, b, c = x >> 10, (x >> 5) & 31, x & 31
                out[:, i] = {"a": a, "aa": a * a, "b": b, "bb": b * b, "c": c, "cc": c * c,
                             "ss": a * a + b * b + c * c}[dg]
        return out

    def _readout(self):
        """(V, nb + 1) trits: the code of every token, and -1 on the reserved tokens for the mask column."""
        V = self.lay.V
        ids = np.arange(V)
        Rm = np.zeros((V, self.nb + 1), I64)
        for b in range(self.nb):
            Rm[:, b] = 2 * ((ids >> b) & 1) - 1
        Rm[[t for t in ids if self.lay.reserved(t)], self.nb] = -1
        return Rm

    def bump(self, out_code, notfire):
        """The logit change (V,) Q16 for one position: B * readout @ [code, nb (1 - fire)]."""
        return B * (self.readout @ np.concatenate([out_code, [notfire]]))

    # ---- attention: the model's integer softmax ------------------------------------------------------------

    def attend(self, h, Q, K, V, mask):
        """Q (Tq, hd), K (Tk, hd), V (Tk, nv) -> (Tq, nv) values read, and a (Tq,) flag: the read is exact."""
        s = R.iscores(Q * h.scale, K, h.hd, exact=True)                    # (Tq, Tk) Q16
        p = R.isoftmax(s, mask)
        o = R.sat(R.rshr(R._mm(p, V, exact=True), R.F_P + R.F_H - R.F_X), 32)
        exact = (p.max(axis=1) == 1 << R.F_P) & np.all(o % 32 == 0, axis=1)
        return R.rshr(o, 5), exact

    # ---- teacher-forced forward ---------------------------------------------------------------------------

    def inputs(self, toks):
        toks = np.asarray(toks, I64)
        T = len(toks)
        assert T <= self.TREAL, f"tfs: {T} positions (at most {self.TREAL})"
        vals = np.zeros((self.n, T), I64)
        vals[self.fw] = self.ftable[toks].T
        vals[self.pw] = self.pos_features(np.arange(T)).T
        vals[self.cw] = self.cv[:, None]
        return vals

    def ops(self, dd, vals):
        w, (lr, lw, lc), (gr, gw, gc), _ = self.by_depth[dd]
        if not len(w):
            return
        T = vals.shape[1]
        left = np.zeros((len(w), T), I64)
        gate = np.zeros((len(w), T), I64)
        np.add.at(left, lr, lc[:, None] * vals[lw])
        np.add.at(gate, gr, gc[:, None] * vals[gw])
        vals[w] = left * np.maximum(gate, 0)

    def forward(self, toks, virtual=None):
        """toks: a transcript (BOS first). virtual: extra (position, token) keys of the copy head (mounted files'
        text, disk.py). Returns dict: code (T, nb), fire (T,), notfire (T,), vals (n, T), exact {head: (T,)}."""
        vals = self.inputs(toks)
        T = vals.shape[1]
        causal = np.tril(np.ones((T, T), dtype=bool))
        exact = {}
        for dd in range(1, self.depth + 1):
            for h in self.by_depth[dd][3]:
                Q, K = h.form(h.q, vals), h.form(h.k, vals)
                Vv = vals[h.v].T
                mask = causal
                if h.name == "copy" and virtual:
                    K2, V2 = self.virtual_kv(h, virtual)
                    K, Vv = np.concatenate([K, K2]), np.concatenate([Vv, V2])
                    mask = np.concatenate([causal, np.ones((T, len(K2)), dtype=bool)], 1)
                vals[h.o], exact[h.name] = (a.T if i == 0 else a for i, a in enumerate(self.attend(h, Q, K, Vv, mask)))
            self.ops(dd, vals)
        return dict(code=vals[self.out_w].T, fire=vals[self.fire_w], notfire=vals[self.notfire_w], vals=vals,
                    exact=exact)

    def virtual_kv(self, h, virtual):
        """Keys / values of the copy head for virtual (position, token) entries."""
        pos = np.array([p for p, _ in virtual], I64)
        tok = np.array([t for _, t in virtual], I64)
        vals = np.zeros((self.n, len(pos)), I64)
        vals[self.fw] = self.ftable[tok].T
        vals[self.pw] = self.pos_features(pos).T
        vals[self.cw] = self.cv[:, None]
        return h.form(h.k, vals), vals[h.v].T

    def target(self, h, vals):
        """An abs head's target position at every position (from its digit wires)."""
        a, b, c = (vals[w] for w in h.tgt)
        return 1024 * a + 32 * b + c

    def decode_code(self, code):
        """The token id an output code names (codes are +-1 bits)."""
        return int(sum(1 << b for b in range(self.nb) if code[b] > 0))

    def decoder(self):
        return Decoder(self)


class Decoder:
    """Incremental forward: one position per step, every head's keys / values cached. step(tok) returns
    (code (nb,), fire, notfire) at that position; .vals holds every wire of every position so far."""

    def __init__(self, net, virtual=None):
        self.net = net
        self.cols = []
        self.tokens = []
        self.K = {h.name: [] for h in net.heads}
        self.V = {h.name: [] for h in net.heads}
        self.inexact = []                          # (position, head) reads that were not exact
        self._vver = 0
        self.virtual = []
        if virtual:
            self.add_virtual(virtual)
        self.ports = {}                            # head name -> an in-forward I/O op (tfs/io_ops.py Port); {}: off
        # KV eviction (docs/tfs_fetch_compact.md): the caches hold rows for the live positions only (kpos: the
        # position of every row); evict(ps) drops positions' rows (and their wire columns) from every head
        self.nk = 0
        self.kpos = []
        self.evicted = set()
        self.check = False                         # check mode: keep the evicted rows and recompute every later
        self._ev = {}                              # position with them, asserting the outputs are the same
        self.ev_hits = {}                          # head -> later reads whose result the evicted rows would change
        self.ev_checked = 0

    @property
    def virtual(self):
        return self._virtual

    @virtual.setter
    def virtual(self, entries):
        self._virtual = entries
        self._vent = []                            # the raw entries (for the other heads that read them, vheads)
        self._vver += 1

    def add_virtual(self, entries):
        """Page in copy-head entries (position, token): a mounted file's text (disk.py)."""
        h = next(h for h in self.net.heads if h.name == "copy")
        K2, V2 = self.net.virtual_kv(h, entries)
        self._virtual += [(K2[i], V2[i]) for i in range(len(K2))]
        self._vent = getattr(self, "_vent", []) + list(entries)
        self._vver += 1

    @property
    def pos(self):
        return len(self.cols)

    def step(self, tok):
        net = self.net
        t = len(self.cols)
        v = self._col(tok, t)
        full = self._col(tok, t) if self.check and self._ev else None
        for dd in range(1, net.depth + 1):
            for h in net.by_depth[dd][3]:
                k = h.form(h.k, v)[0]
                K, Vv = self._push(h.name, k, v[h.v, 0])
                if self.ports and h.name in self.ports:   # in-forward I/O: an address in the port's region is read
                    io = self.ports[h.name].serve(self, h, v, t, tok)   # through its capability, not the cache
                    if io is not None:
                        v[h.o, 0] = io
                        if full is not None:
                            full[h.o, 0] = io
                        continue
                if h.name == "copy" and self.virtual:
                    vk, vv = self._virtual_arrays()
                    K = np.concatenate([K, vk])
                    Vv = np.concatenate([Vv, vv])
                elif h.name in getattr(net, "vheads", ()) and self.virtual:   # another reader of the same entries
                    vk, vv = self._virtual_of(h)
                    K = np.concatenate([K, vk])
                    Vv = np.concatenate([Vv, vv])
                o, ex = net.attend(h, h.form(h.q, v), K, Vv, None)
                v[h.o, 0] = o[0]
                if not ex[0]:
                    self.inexact.append((t, h.name))
                if full is not None:              # the same read with the evicted rows back (their own column)
                    ek, evv = self._ev[h.name]
                    o2, _ = net.attend(h, h.form(h.q, full), np.concatenate([K, ek]), np.concatenate([Vv, evv]), None)
                    full[h.o, 0] = o2[0]
                    if not np.array_equal(o2[0], o[0]):
                        self.ev_hits[h.name] = self.ev_hits.get(h.name, 0) + 1
            net.ops(dd, v)
            if full is not None:
                net.ops(dd, full)
        if full is not None:
            self.ev_checked += 1
            ws = np.concatenate([net.out_w, [net.fire_w, net.notfire_w], getattr(net, "deny_w", []),
                                 [net.active_w] if hasattr(net, "active_w") else []]).astype(I64)
            if not np.array_equal(v[ws, 0], full[ws, 0]):
                raise AssertionError(f"eviction: position {t} reads the evicted region (outputs differ; heads whose "
                                     f"reads changed so far: {self.ev_hits})")
        self.cols.append(v[:, 0])
        self.tokens.append(int(tok))
        self.kpos.append(t)
        self.nk += 1
        return v[net.out_w, 0], int(v[net.fire_w, 0]), int(v[net.notfire_w, 0])

    def evict(self, positions):
        """Drop positions (earlier ones) from every head's cache and their wire columns. Position numbers stay: later
        positions keep theirs (the nets address the context by absolute position). Safe only where nothing reads them
        again; in check mode (self.check) every later step is recomputed with the evicted rows and must give the same
        outputs (code, fire, notfire, a mask's deny / active), else AssertionError."""
        ps = set(int(p) for p in positions) - self.evicted
        if not ps:
            return 0
        assert max(ps) < self.pos
        keep = np.array([i for i, q in enumerate(self.kpos) if q not in ps], I64)
        drop = np.array([i for i, q in enumerate(self.kpos) if q in ps], I64)
        for name in list(self.K):
            K, Vv = self.K[name], self.V[name]
            if isinstance(K, list):
                if not K:
                    continue
                K, Vv = np.stack(K), np.stack(Vv)
            if self.check and len(drop):
                ek, ev = self._ev.get(name, (np.zeros((0, K.shape[1]), I64), np.zeros((0, Vv.shape[1]), I64)))
                self._ev[name] = (np.concatenate([ek, K[drop]]), np.concatenate([ev, Vv[drop]]))
            n = len(keep)
            K2 = np.zeros((max(64, 2 * n), K.shape[1]), I64)
            V2 = np.zeros((K2.shape[0], Vv.shape[1]), I64)
            K2[:n], V2[:n] = K[keep], Vv[keep]
            self.K[name], self.V[name] = K2, V2
        self.kpos = [self.kpos[i] for i in keep]
        self.nk = len(keep)
        for q in ps:
            self.cols[q] = None
        self.evicted |= ps
        return len(drop)

    def kv_bytes(self):
        """The caches' live rows, in bytes (int64 keys and values)."""
        return sum(self.nk * (K.shape[1] + self.V[name].shape[1]) * 8 for name, K in self.K.items()
                   if not isinstance(K, list))

    def _virtual_of(self, h):
        """Another head's keys / values for the virtual entries (net.vheads: e.g. fetch3's tape readers)."""
        key = (self._vver, h.name)
        c = getattr(self, "_vcache", {})
        if key not in c:
            c = {k: v for k, v in c.items() if k[0] == self._vver}
            c[key] = self.net.virtual_kv(h, self._vent)
            self._vcache = c
        return c[key]

    def _virtual_arrays(self):
        """The copy head's virtual keys / values stacked, kept until the entries change (a new page, a new tape)."""
        key = self._vver
        if getattr(self, "_vkey", None) != key:
            self._vkey = key
            self._varr = (np.stack([x for x, _ in self.virtual]), np.stack([y for _, y in self.virtual]))
        return self._varr

    def _push(self, name, k, val):
        """Append a head's key and value at this position; (keys, values) of every position so far. The caches are
        arrays that grow by doubling (re-stacking lists at every step made a long decode quadratic)."""
        t = self.nk                                 # the row: live positions so far (eviction drops rows)
        K, Vv = self.K[name], self.V[name]
        if isinstance(K, list):                     # first use (or a list from before): an array with room
            K0, V0 = (np.stack(K), np.stack(Vv)) if K else (np.zeros((0, len(k)), I64), np.zeros((0, len(val)), I64))
            K = np.zeros((max(64, 2 * len(K0)), len(k)), I64)
            Vv = np.zeros((K.shape[0], len(val)), I64)
            K[:len(K0)], Vv[:len(V0)] = K0, V0
        if t >= K.shape[0]:
            K = np.concatenate([K, np.zeros_like(K)])
            Vv = np.concatenate([Vv, np.zeros_like(Vv)])
        K[t], Vv[t] = k, val
        self.K[name], self.V[name] = K, Vv
        return K[:t + 1], Vv[:t + 1]

    def _col(self, tok, t):
        net = self.net
        assert t < net.TREAL
        v = np.zeros((net.n, 1), I64)
        v[net.fw, 0] = net.ftable[tok]
        v[net.pw, 0] = net.pos_features([t])[0]
        v[net.cw, 0] = net.cv
        return v

    def wire(self, name, t=-1):
        return int(self.cols[t][self.net.names[name]])

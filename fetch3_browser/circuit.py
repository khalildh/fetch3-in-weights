"""A compiled tool's net (exported by examples/wire/toolnet.tern) run on token streams: tfs/net.py's runner, for any
tool (docs/tfs_tools.md). The files are fs_net.tern's; only meta.npy is generic (n, nb, lo, S, heads, TREAL, GAP) and
tool.txt names the kind. The token embedding is the tool's (`features(fnames, V)`: (V, n_feat) int64); the position
table, the heads through the integer softmax, the ReGLU levels, the incremental decoder and the exactness checks are
tfs/net.py's own (inherited).

    net = Circuit(dir, features)              # features: fnames, V -> table (the tool's, e.g. tfs/calc.py)
    out = net.forward(tokens)                 # per position: code (T, nb), fire, notfire, vals, exact {head: (T,)}
    dec = net.decoder(); dec.step(tok)        # incremental
"""

from pathlib import Path

import numpy as np

from . import net as N

I64 = np.int64


class Circuit(N.Net):
    def __init__(self, d, features, V):
        d = Path(d)
        meta = np.load(d / "meta.npy")
        self.n, self.nb, self.lo, self.S, nh, self.TREAL, self.GAP = (int(x) for x in meta)
        self.kind = (d / "tool.txt").read_text().strip()
        self.V = V
        assert (V - 1).bit_length() == self.nb, f"{self.kind} net: {self.nb} code bits, vocabulary {V}"
        self.fnames = (d / "feat.txt").read_text().split()
        self.fw = np.load(d / "feat_w.npy")
        self.pnames = (d / "posf.txt").read_text().split()
        self.pw = np.load(d / "posf_w.npy")
        self.cw, self.cv = np.load(d / "const_w.npy"), np.load(d / "const_v.npy")
        self.op_w, self.op_d = np.load(d / "op_w.npy"), np.load(d / "op_d.npy")
        self.op_l = N._sparse(np.load(d / "op_l.npy"), 3)
        self.op_g = N._sparse(np.load(d / "op_g.npy"), 3)
        names = (d / "heads.txt").read_text().split()
        hd = np.load(d / "head_d.npy")
        hq, hk = N._sparse(np.load(d / "head_q.npy"), 4), N._sparse(np.load(d / "head_k.npy"), 4)
        hv, ho = N._sparse(np.load(d / "head_v.npy"), 3), N._sparse(np.load(d / "head_o.npy"), 3)
        ht = np.load(d / "head_t.npy").reshape(-1, 4)
        tg = {int(r[0]): r[1:] for r in ht}
        self.heads = []
        for h in range(nh):
            sel = lambda a, m: [x[a[0] == h] for x in a[1:]][:m]  # noqa: E731
            self.heads.append(N.Head(names[h], int(hd[h]), sel(hq, 3), sel(hk, 3), sel(hv, 2)[1], sel(ho, 2)[1],
                                     tg.get(h)))
        ow = np.load(d / "out_w.npy")
        self.out_w, self.notfire_w, self.fire_w = ow[:self.nb], int(ow[self.nb]), int(ow[self.nb + 1])
        self.names = dict(zip((d / "names.txt").read_text().split(), (int(x) for x in np.load(d / "names_w.npy"))))
        self.depth = int(max(self.op_d.max(), hd.max()))
        self.by_depth = []
        for dd in range(self.depth + 1):
            rows = np.nonzero(self.op_d == dd)[0]
            loc = np.full(len(self.op_w), -1, I64)
            loc[rows] = np.arange(len(rows))
            ml, mg = np.isin(self.op_l[0], rows), np.isin(self.op_g[0], rows)
            self.by_depth.append((self.op_w[rows], (loc[self.op_l[0][ml]], self.op_l[1][ml], self.op_l[2][ml]),
                                  (loc[self.op_g[0][mg]], self.op_g[1][mg], self.op_g[2][mg]),
                                  [h for h in self.heads if h.depth == dd]))
        self.ftable = np.asarray(features(self.fnames, V), I64)
        assert self.ftable.shape == (V, len(self.fnames))

    def stats(self):
        return dict(wires=self.n, ops=len(self.op_w), heads=len(self.heads), levels=self.depth,
                    trits=int(len(self.op_l[0]) + len(self.op_g[0])))


def code_columns(V, nb):
    """(V, nb) +-1: every token's code (the readout's rows)."""
    ids = np.arange(V)
    return np.stack([2 * ((ids >> b) & 1) - 1 for b in range(nb)], 1).astype(I64)

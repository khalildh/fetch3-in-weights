"""The fetch tool (docs/tfs_fetch.md): internet access for the integer model as a compiled tool. Its meaning (the spec
the compiled net examples/wire/fetch_net.tern is checked against), the token layout, the net's token embedding, the
net run with the runtime's paging, and the tool's entry for tfs/tools.py. The runtime that fetches (HTTP, caps, the
allowlist again, text, tokens, the cache) is tfs/fetch_rt.py.

Tokens, ids LO .. LO + 53:

    LO + 0, 1      <|fetch|> <|fetch_end|>                                     the model writes these
    LO + 2..47     <|url_c|> for c in CHARS (a-z 0-9 . - _ ~ / : ? = & %)      the model writes these
    LO + 48..51    <|fetch_data|> <|fetch_done|> <|fetch_err|> <|fetch_deny|>   reserved: only forced
    LO + 52, 53    <|fetch_fail|> <|fetch_cut|>                                reserved: only paged in by the runtime

    <|fetch|> URL <|fetch_end|>  ->  <|fetch_err|>                          malformed (not 1..64 characters, not
                                                                            http(s)://, a non-URL token, no open call)
                                 ->  <|fetch_deny|>                         the origin is not on the allowlist
                                 ->  <|fetch_data|> PAGE <|fetch_done|>     PAGE: the runtime's page, copied exactly

The spec is positional, as the calculator's: at a <|fetch_end|> the response's first token; while the latest fetch
marker is <|fetch_data|> at d, position p copies page[p - d], and <|fetch_done|> at p - d = len(page). The page is
the runtime's (paged into the copy head's virtual entries VB .. VB + len, then <|fetch_end|> as the terminator), so
the spec takes it from a callback: `pages(url) -> token ids`, called when <|fetch_data|> follows an allowed call.
"""

import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from ._paths import ROOT

from . import circuit as CI

I64 = np.int64
CHARS = "abcdefghijklmnopqrstuvwxyz0123456789.-_~/:?=&%"
NAMES = ["fetch", "fetch_end"] + [f"url_{c}" for c in CHARS] + ["fetch_data", "fetch_done", "fetch_err", "fetch_deny",
                                                                 "fetch_fail", "fetch_cut"]
S = len(NAMES)                 # 54
EMIT = 2 + len(CHARS)          # the first reserved token
UMAX = 64                      # URL characters at most
OMAX = 48                      # an allowed origin's characters at most
VB = 16384                     # the page's first virtual position
PMAX = 16383 - 1               # page tokens at most (VB + PMAX + 1 < 32768: the copy head's digits)
NET = ROOT / "examples/wire/fetch_net.tern"
ORIGIN = re.compile(r"^(https?)://([a-z0-9-]+(\.[a-z0-9-]+)*)(:[0-9]{1,5})?$")


def check_origin(o):
    """An allowlist entry: scheme://host[:port], lower case, at most OMAX characters."""
    if not isinstance(o, str) or len(o) > OMAX or not ORIGIN.match(o):
        raise ValueError(f"fetch: not an origin (scheme://host[:port], lower case, <= {OMAX} characters): {o!r}")
    return o


def parse_allow(s):
    """'https://a.org,http://127.0.0.1:8000' (or a list) -> validated, distinct origins; '' / '-' / None -> []."""
    if s is None or s in ("", "-"):
        return []
    items = s.split(",") if isinstance(s, str) else list(s)
    out = []
    for o in items:
        o = check_origin(o.strip())
        if o not in out:
            out.append(o)
    return out


class Layout:
    def __init__(self, lo):
        self.lo = lo
        self.FETCH, self.END = lo, lo + 1
        self.chars = {c: lo + 2 + i for i, c in enumerate(CHARS)}
        self.char_of = {t: c for c, t in self.chars.items()}
        (self.DATA, self.DONE, self.ERR, self.DENY, self.FAIL, self.CUT) = range(lo + EMIT, lo + S)
        self.markers = {self.FETCH, self.END, self.DATA, self.DONE, self.ERR, self.DENY}

    def name(self, t):
        return f"<|{NAMES[t - self.lo]}|>"

    def tokens(self):
        return [self.name(self.lo + i) for i in range(S)]

    def reserved(self):
        return range(self.lo + EMIT, self.lo + S)

    def url(self, s):
        """A URL's character tokens (KeyError on a character outside CHARS)."""
        return [self.chars[c] for c in s]

    def call(self, s):
        return [self.FETCH, *self.url(s), self.END]

    def text(self, toks):
        """Character tokens -> the URL string, or None if a token is not a character."""
        if any(t not in self.char_of for t in toks):
            return None
        return "".join(self.char_of[t] for t in toks)


def verdict(url, allow):
    """'err' | 'deny' | 'ok' for a URL string (None: a non-character token) and the allowlist (the net's checks)."""
    if url is None or not 1 <= len(url) <= UMAX:
        return "err"
    if not (url.startswith("https://") or url.startswith("http://")):
        return "err"
    for o in allow:
        if url == o or url.startswith(o + "/"):
            return "ok"
    return "deny"


class Spec:
    """feed(t): the token at the next position; returns the token the fetch tool forces after it, or None.
    pages(url) -> the page's token ids (the runtime's), asked when <|fetch_data|> follows an allowed call."""

    def __init__(self, lay, allow, pages):
        self.lay, self.allow, self.pages = lay, list(allow), pages
        self.toks = []
        self.mark = None                 # (position, token) of the latest fetch marker
        self.pending = None              # the URL of an allowed call whose <|fetch_data|> comes next
        self.page = []
        self.fetched = []                # the URLs fetched, in order

    def feed(self, t):
        lay, p = self.lay, len(self.toks)
        self.toks.append(t)
        if t in lay.markers:
            prev = self.mark
            self.mark = (p, t)
            if t == lay.DATA and self.pending is not None and p > 0 and self.toks[p - 1] == lay.END:
                self.page = list(self.pages(self.pending))
                self.fetched.append(self.pending)
            self.pending = None
            if t == lay.END:
                if prev is None or prev[1] != lay.FETCH:
                    return lay.ERR
                url = lay.text(self.toks[prev[0] + 1:p])
                v = verdict(url, self.allow)
                if v == "ok":
                    self.pending = url
                return {"err": lay.ERR, "deny": lay.DENY, "ok": lay.DATA}[v]
        if self.mark is not None and self.mark[1] == lay.DATA:
            o = p - self.mark[0]
            if o < len(self.page):
                return self.page[o]
            if o == len(self.page):
                return lay.DONE
        return None


def forced(lay, allow, pages, toks):
    sp = Spec(lay, allow, pages)
    return [sp.feed(t) for t in toks]


def call_url(lay, toks):
    """The URL (string) of the call that the <|fetch_end|> at the end of toks closes (None if it is not open)."""
    e = len(toks) - 1
    for j in range(e - 1, -1, -1):
        if toks[j] in lay.markers:
            return lay.text(toks[j + 1:e]) if toks[j] == lay.FETCH else None
    return None


def virtual_page(lay, page):
    """The copy head's entries for a page: (position, token), then <|fetch_end|> as the terminator."""
    page = [int(t) for t in page]
    assert len(page) <= PMAX, f"a page holds at most {PMAX} tokens"
    return [(VB + i, t) for i, t in enumerate(page)] + [(VB + len(page), lay.END)]


def features(lay):
    """The net's token embedding for vocabulary V (fetch_net.tern's feature names)."""
    def table(fnames, V):
        F = np.zeros((V, len(fnames)), I64)
        col = {n: i for i, n in enumerate(fnames)}
        ids = np.arange(V)
        for n, i in col.items():
            if n.startswith("code"):
                F[:, i] = 2 * ((ids >> int(n[4:])) & 1) - 1
        for t in lay.markers:
            F[t, col["MARK"]] = 1
        F[lay.FETCH, col["FETCH"]] = 1
        F[lay.END, col["FEND"]] = 1
        F[lay.DATA, col["DATA"]] = 1
        for c, t in lay.chars.items():
            F[t, col["ISCHAR"]], F[t, col["CVAL"]] = 1, CHARS.index(c)
        return F
    return table


def export(out_dir, lay, V, allow):
    """Build and export the net (runs fetch_net.tern in the interpreter, a few seconds)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    allow = parse_allow(allow)
    r = subprocess.run([sys.executable, "-m", "tern", str(NET), str(lay.lo), str(V), ",".join(allow) or "-",
                        str(out_dir)], capture_output=True, text=True, cwd=ROOT, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-4000:])
    (out_dir / "allow.txt").write_text("".join(o + "\n" for o in allow))
    return r.stdout


class FetchNet(CI.Circuit):
    """The exported net (tfs/circuit.py's runner) with the runtime's page: one page at a time in the copy head's
    virtual entries, replaced by the next fetch (`page(dec, entries)`); `forward(toks, pages=[(d, entries)])` gives
    each position the page of its latest <|fetch_data|> (what the decoder had paged in by then)."""

    def __init__(self, d, lay, V):
        super().__init__(d, features(lay), V)
        self.lay = lay
        p = Path(d) / "allow.txt"
        self.allow = p.read_text().split() if p.exists() else []

    def page(self, dec, entries):
        dec.virtual = []
        dec.add_virtual(entries)

    def forward(self, toks, pages=None):
        if not pages:
            return super().forward(toks)
        T = len(toks)
        out = None
        starts = [d for d, _ in pages] + [T]
        # positions before the first page see none; [d_i, d_{i+1}) see page i (the copy head only reads it there)
        segs = [(0, starts[0], None)] + [(starts[i], starts[i + 1], pages[i][1]) for i in range(len(pages))]
        for lo, hi, ent in segs:
            if lo >= hi:
                continue
            f = super().forward(toks, virtual=ent)
            if out is None:
                out = {k: (v.copy() if k != "exact" else {h: x.copy() for h, x in v.items()}) for k, v in f.items()}
                continue
            out["code"][lo:hi] = f["code"][lo:hi]
            for k in ("fire", "notfire"):
                out[k][lo:hi] = f[k][lo:hi]
            out["vals"][:, lo:hi] = f["vals"][:, lo:hi]
            for h in out["exact"]:
                out["exact"][h][lo:hi] = f["exact"][h][lo:hi]
        return out


class FetchTool:
    """tfs/tools.py's entry: the block, the spec, the net. allow: the origins compiled into the net (tools.json)."""
    kind, priority, dirname = "fetch", 6, "fetch"
    S = S

    @staticmethod
    def size(**_):
        return S

    def __init__(self, lo, V, allow=None, **_):
        self.lo, self.V = lo, V
        self.lay = Layout(lo)
        self.allow = parse_allow(allow)
        self.net = None
        self.pages = lambda url: [self.lay.FAIL]        # the spec's pages: the runtime sets it

    def config(self):
        return {"allow": self.allow}

    def tokens(self):
        return self.lay.tokens()

    def reserved(self):
        return self.lay.reserved()

    def export(self, d):
        return export(d, self.lay, self.V, self.allow)

    def load(self, d):
        self.net = FetchNet(d, self.lay, self.V)
        if self.net.allow != self.allow:
            raise ValueError(f"{d}: the net allows {self.net.allow}, tools.json {self.allow}")
        return self.net

    def spec(self):
        return Spec(self.lay, self.allow, self.pages)


def session(lay, allow, pages, rng, n_calls=6, vocab=range(1, 20), bad=0.3, follow=True):
    """A random transcript: ordinary tokens and fetch calls, some allowed, some denied, some malformed (empty, too long,
    no scheme, a base token or a stray marker inside, a stray <|fetch_end|>); every forced token followed. Returns
    (tokens, [(position of <|fetch_data|>, url)] of the fetches)."""
    vocab = list(vocab)
    sp = Spec(lay, allow, pages)
    toks, nxt, fetches = [], None, []

    def emit(t):
        nonlocal nxt
        if t == lay.DATA and toks and toks[-1] == lay.END:
            fetches.append((len(toks), call_url(lay, toks)))
        toks.append(t)
        nxt = sp.feed(t)

    def put(ts):
        for t in ts:
            while follow and nxt is not None:
                emit(nxt)
            emit(t)

    def word(n):
        return "".join(CHARS[int(i)] for i in rng.integers(0, 36, n))

    others = ["https://evil.test", "http://" + word(5) + ".test", "https://" + word(3) + ".org:8080"]
    put([0])
    for _ in range(n_calls):
        put([int(x) for x in rng.choice(vocab, int(rng.integers(0, 3)))])
        r = rng.random()
        o = (allow[int(rng.integers(len(allow)))] if allow and r < 0.6 else others[int(rng.integers(len(others)))])
        url = o + ("" if rng.random() < 0.2 else "/" + word(int(rng.integers(1, 12))))
        if rng.random() < 0.15:
            url = o + word(int(rng.integers(1, 3)))      # the origin's host extended: another host, denied
        url = url[:UMAX]
        call = lay.call(url)
        if rng.random() < bad:
            k = int(rng.integers(7))
            if k == 0:
                call = [lay.FETCH, lay.END]
            elif k == 1:
                call = lay.call((url + "/" + "a" * UMAX)[:UMAX + 1 + int(rng.integers(0, 3))])
            elif k == 2:
                call = lay.call(url.replace("://", ":/", 1))
            elif k == 3:
                i = int(rng.integers(1, len(call) - 1))
                call = call[:i] + [int(rng.choice(vocab))] + call[i:]
            elif k == 4:
                call = [lay.END]
            elif k == 5:
                call = [lay.FETCH] + call
            else:
                call = lay.call(url.replace("http", "ftp", 1))
        put(call)
    put([int(x) for x in rng.choice(vocab, 2)])
    while follow and nxt is not None:
        emit(nxt)
    return toks, fetches

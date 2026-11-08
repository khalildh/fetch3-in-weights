"""Fetch v3 (docs/tfs_fetch3.md): a light text-mode browser in the weights. v2 (tfs/fetch2.py) put HTTP into the net;
v3 adds, in the same compiled net (examples/wire/fetch3_net.tern, a superset of v2's):

1. **HTML into structure**: a state machine over the response's bytes whose state is carried in the token each byte
   becomes (docs/tfs_json.md's trick): tags, quoted attribute values (a `>` inside quotes stays inside), comments,
   <script> / <style> contents skipped, entities, block tags as line breaks, headings, and **numbered links**: an
   <a href="..."> whose value resolves to a URL becomes the token <|ln_k|> (shown "[k]"), k = 1, 2, .. per page;
2. **following links**: the model writes <|follow|> <|ln_k|>; the net looks up link k on the current page (the
   latest response), resolves its href against the page's URL (absolute, root-relative /x, relative x) and forces
   the call <|fetch|> URL <|fetch_end|>, which then goes through v2's machinery (the allowlist, the request);
3. **masking navigation** (the companion part fetch3_mask, on when start points are given): inside a <|fetch|> call
   the model can write only an allowlisted start point, and after <|follow|> only a link number of the current page;
4. **paging**: a page segment holds at most PAGE positions; then the net forces <|h_more|>, the byte offset as three
   base-32 digits and <|fetch_cut|>, and the model may write <|next|> for the next segment (back = earlier pages,
   still in the context).

Tokens (S = 81 ids from LO, after v1's and v2's blocks, which v3 needs):

    LO + 0 .. 12   <|s_skip|> <|s_lt|> <|s_tag|> <|s_dq|> <|s_sq|> <|s_heq|> <|s_hdq|> <|s_hsq|> <|s_hdf|> <|s_hsf|>
                   <|s_cmt|> <|s_scr|> <|s_sty|>        a body byte that is not text, by the parser's state
    LO + 13        <|x_head|>                           a heading starts (shown "\\n# ")
    LO + 14        <|h_more|>                           the page segment is full (paging)
    LO + 15 .. 46  <|pg_0|> .. <|pg_31|>                the byte offset where the segment stopped (3 digits)
    LO + 47, 48    <|follow|> <|next|>                  the model writes these
    LO + 49 .. 80  <|ln_1|> .. <|ln_32|>                links: forced in pages, written by the model after <|follow|>

Reserved (only forced): LO + 0 .. 46. The spec here is the meaning the net is checked against (tfs/fetch3_fuzz.py).
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from ._paths import ROOT

from . import circuit as CI
from . import fetch as F
from . import fetch2 as F2

I64 = np.int64
K = 32                    # links per page
NPG = 32                  # page-offset digit tokens (base 32)
PAGE = 1024               # positions per page segment (default; the merge's / runtime's `page`)
STATES = ["s_skip", "s_lt", "s_tag", "s_dq", "s_sq", "s_heq", "s_hdq", "s_hsq", "s_hdf", "s_hsf", "s_cmt", "s_scr",
          "s_sty"]
NAMES = STATES + ["x_head", "h_more"] + [f"pg_{i}" for i in range(NPG)] + ["follow", "next"] + \
    [f"ln_{k}" for k in range(1, K + 1)]
S = len(NAMES)            # 81
EMIT = len(STATES) + 2 + NPG          # 47: the first token the model writes
# page compaction (docs/tfs_fetch_compact.md): four reserved tokens after the links, the compact block's markers
CNAMES = ["pc_url", "pc_links", "pc_text", "pc_end"]
SC = S + len(CNAMES)      # 85: the block with compaction
(ST_DATA, ST_LT, ST_TAG, ST_DQ, ST_SQ, ST_HEQ, ST_HDQ, ST_HSQ, ST_HDF, ST_HSF, ST_CMT, ST_SCR, ST_STY) = range(1, 14)
MK3 = {"follow": 17, "next": 18, "h_more": 19, "pc_url": 20, "pc_links": 21, "pc_text": 22, "pc_end": 23}
UMAX = F.UMAX
WS = F2.WS
DELIM = (9, 10, 13, 32, 47, 62)       # after a tag name: whitespace, '/', '>'
BLOCK = ["p", "br", "hr", "li", "ul", "ol", "dl", "dt", "dd", "tr", "div", "pre", "nav", "table", "title", "form",
         "main", "body", "header", "footer", "section", "article", "aside", "h1", "h2", "h3", "h4", "h5", "h6"]
NET = ROOT / "examples/wire/fetch3_net.tern"
URLB = frozenset(ord(c) for c in F2.URLCH)     # an href's bytes: v2's URL characters (upper case included)


class Layout(F2.Layout):
    """lo3: v3's first id; lo2: v2's (tfs/fetch2.py Layout), fetch_lo: v1's; compact: page compaction, its four
    tokens after the links (docs/tfs_fetch_compact.md)."""

    def __init__(self, lo3, lo2, fetch_lo, words=(), agent=F2.AGENT, compact=False):
        super().__init__(lo2, fetch_lo, words, agent)
        self.lo3 = lo3
        self.compact = bool(compact)
        self.S3 = SC if compact else S
        self.PCU, self.PCL, self.PCT, self.PCE = (lo3 + S + i for i in range(len(CNAMES)))
        for i, n in enumerate(STATES):
            setattr(self, n.upper(), lo3 + i)
        self.HEAD, self.MORE = lo3 + 13, lo3 + 14
        self.PG = [lo3 + 15 + i for i in range(NPG)]
        self.FOLLOW, self.NEXT = lo3 + EMIT, lo3 + EMIT + 1
        self.LN = [lo3 + EMIT + 2 + k for k in range(K)]          # LN[k - 1] = <|ln_k|>
        self.ln_of = {t: k + 1 for k, t in enumerate(self.LN)}
        self.pg_of = {t: i for i, t in enumerate(self.PG)}
        self.mk.update({self.FOLLOW: MK3["follow"], self.NEXT: MK3["next"], self.MORE: MK3["h_more"]})
        if compact:
            self.mk.update({self.PCU: MK3["pc_url"], self.PCL: MK3["pc_links"], self.PCT: MK3["pc_text"],
                            self.PCE: MK3["pc_end"]})
        st = {self.S_SKIP: ST_DATA, self.HEAD: ST_DATA, self.S_LT: ST_LT, self.S_TAG: ST_TAG, self.S_DQ: ST_DQ,
              self.S_SQ: ST_SQ, self.S_HEQ: ST_HEQ, self.S_HDQ: ST_HDQ, self.S_HSQ: ST_HSQ, self.S_HDF: ST_HDF,
              self.S_HSF: ST_HSF, self.S_CMT: ST_CMT, self.S_SCR: ST_SCR, self.S_STY: ST_STY}
        for t in self.LN:
            st[t] = ST_TAG
        for t in self.byte_of:
            st[t] = ST_DATA
        for t in self.word_of:
            st[t] = ST_DATA
        self.st = st
        self.rawb = {self.SKIP, *[lo3 + i for i in range(len(STATES))]}      # tokens that stand for their raw byte

    def name(self, t):
        if self.lo3 <= t < self.lo3 + self.S3:
            return f"<|{(NAMES + CNAMES)[t - self.lo3]}|>"
        return super().name(t)

    def tokens(self):
        return [f"<|{n}|>" for n in (NAMES + CNAMES)[:self.S3]]

    def reserved(self):
        return list(range(self.lo3, self.lo3 + EMIT)) + (self.pcs() if self.compact else [])

    def pcs(self):
        """The compact block's markers."""
        return [self.PCU, self.PCL, self.PCT, self.PCE]

    def textish(self, t):
        """A token of a page's text (the compact block copies these): TEXT bytes, words, links, headings."""
        return t in self.byte_of or t in self.word_of or t in self.ln_of or t == self.HEAD

    def show(self, toks, words=None):
        """A transcript as text: v2's rendering, links "[k]", headings "# ", the parser's skips "·" (DATA) or
        omitted (markup), page digits as one number."""
        out = []
        for t in toks:
            if t in self.ln_of:
                out.append(f"[{self.ln_of[t]}]")
            elif t == self.HEAD:
                out.append("\n# ")
            elif t == self.S_SKIP:
                out.append("·")
            elif self.lo3 <= t < self.lo3 + len(STATES):
                out.append("")
            elif t in self.pg_of:
                out.append(f"{self.pg_of[t]}.")
            elif t in (self.MORE, self.FOLLOW, self.NEXT):
                out.append(self.name(t))
            elif self.compact and t in self.pcs():
                out.append({self.PCU: "\n[page ", self.PCL: "]\n", self.PCT: "\n", self.PCE: "\n[end]\n"}[t])
            else:
                out.append(super().show([t], words))
        return "".join(out)

    def page_text(self, toks):
        """What a reader of the page sees: TEXT bytes, words (they stand on their last character, the earlier ones
        are TEXT), links "[k]", headings."""
        out = []
        for t in toks:
            if t in self.byte_of:
                out.append(chr(self.byte_of[t]))
            elif t in self.word_of:
                out.append(self.word_of[t][-1])
            elif t in self.ln_of:
                out.append(f"[{self.ln_of[t]}]")
            elif t == self.HEAD:
                out.append("\n# ")
        return "".join(out)


# ---- links ------------------------------------------------------------------------------------------------------

def host_end(url):
    j = url.find("/", 8)
    return j if j >= 0 else len(url)


def resolve(base, href):
    """An href (its part before '#', every character a URL character) against the page's URL (its fragment dropped):
    None for '//host' (network-path references are not links). The net's rule (no dot segments: they pass through to
    the server)."""
    if href.startswith("https://") or href.startswith("http://"):
        return href
    if href.startswith("//"):
        return None
    base = F2.base_url(base)
    he = host_end(base)
    if href.startswith("/"):
        return base[:he] + href
    lsl = None
    for j in range(8, len(base)):
        if base[j] == "?":
            break
        if base[j] == "/":
            lsl = j
    if lsl is None:
        return base + "/" + href
    return base[:lsl + 1] + href


def _ci(x, c):
    c = ord(c)
    return x == c or (97 <= c <= 122 and x == c - 32)


def tagch(b):
    return 97 <= b <= 122 or 65 <= b <= 90 or b in (47, 33, 63)


# ---- the spec ---------------------------------------------------------------------------------------------------

class Spec:
    """feed(t) -> the token the v3 net forces after t, or None. allow: the origins; respond(scheme, target, request
    bytes) -> (bytes, terminator), asked at <|h_send|> (the runtime's pipe); page: positions per page segment."""

    def __init__(self, lay, allow, respond, page=PAGE):
        self.lay, self.allow, self.respond, self.page = lay, list(allow), respond, page
        self.toks = []
        self.mark = self.prev_mark = None
        self.cur = None               # the URL being fetched (the call's, then each Location's)
        self.nred = 0
        self.blk = self.blk_at = None
        self.d = self.D = None        # the latest <|h_send|>; the effective send (d plus the paging gaps)
        self.Dat = []                 # D at every position
        self.r = None
        self.h = None
        self.info = None
        self.chunk = None             # (virtual position, size) of the latest <|h_chunk|>
        self.base = None              # the URL of the current page's request
        self.L, self.V = {}, {}       # the latest position / value of each tracked token class
        self.lnpos = {}               # link k -> its latest position in a page
        self.links = {}               # position of a page's <|ln_k|> -> the resolved URL
        self.fol = None               # (position, URL) of the latest valid follow
        self.fetched = []             # (scheme, target, request) of every send
        self.sent = []                # (URL, how: call / follow / redirect) of every send
        self.how = None
        self.follows = []             # (k, URL) of every valid follow
        self.tpos = []                # positions of the pages' text tokens (compaction)
        self.cb = None                # (position of its <|fetch_done|>, tokens) of the latest compact block
        self.tab = {}                 # link k -> (position of its <|ln_k|> in the latest table, URL)
        self.blocks = []              # (done position, block tokens) of every compact block

    def nlinks(self):
        """The current page's link count (what a follow may name)."""
        if self.lay.compact:
            n = self.cb[2] if self.cb is not None else 0
            return n if n and self.tab[1][0] > self.last("hend") else 0
        return self.V["ln"] if self.last("ln") > self.last("hend") else 0

    def tape(self, i):
        if self.r is None or i < 0:
            return F2.NONE
        return self.r[i] if i < len(self.r) else F2.NONE

    def last(self, k):
        return self.L.get(k, -1)

    def rb(self, q):
        """The raw byte the token at q stands for (the window's r[i - 1] there)."""
        if self.Dat[q] is None:
            return F2.NONE
        return self.tape(q - 1 - self.Dat[q])

    # -- the feed ------------------------------------------------------------------------------------------------
    def feed(self, t):
        lay, f, p = self.lay, self.lay.f, len(self.toks)
        self.toks.append(t)
        is_mark = t in lay.mk
        if is_mark:
            self.prev_mark, self.mark = self.mark, (p, t)
        mk = self.mark[1] if self.mark else None
        valid_next = False
        if t == lay.NEXT:
            valid_next = self._resume_ok(p)
            if valid_next:
                c = self.L["more"]
                i_res = sum(lay.pg_of[self.toks[c + 1 + j]] * 32 ** (2 - j) for j in range(3))
                self.D = p - i_res
                self.L["resume"] = p
        self.Dat.append(self.D)
        self._track(p, t, mk in (lay.HEND, lay.CHUNK, lay.NEXT))
        if lay.compact:
            r = self._compact(p, t, mk)
            if r is not False:
                return r
        if t == f.END:
            return self._verdict(p)
        if p >= 1 and self.toks[p - 1] == lay.FOLLOW and not is_mark:
            return self._follow(p, t)
        if self.fol is not None and mk == f.FETCH and self.mark[0] == self.fol[0] + 1:
            k, url = p - self.fol[0] - 1, self.fol[1]
            return lay.utok[url[k]] if k < len(url) else (f.END if k == len(url) else None)
        if mk in (lay.TCP, lay.TLS, lay.REQ) or mk in lay.REDIR:
            if mk in lay.REDIR and t == mk:
                self.blk = F2.block(lay, self.cur)
                self.blk_at = p
                return self.blk[0]
            k = p - self.blk_at
            return self.blk[k] if k < len(self.blk) else None
        if mk == lay.SEND:
            if t == lay.SEND:
                self._send(p)
            return self._head(p)
        if mk == lay.MORE:
            k = p - self.mark[0]
            if k == 3:
                return f.CUT
            ib = self.mark[0] - 1 - self.D
            return lay.PG[[ib >> 10, (ib >> 5) & 31, ib & 31][k]] if k < 3 else None
        if t == lay.NEXT:
            return self._body(p) if valid_next else f.ERR
        if mk in (lay.HEND, lay.CHUNK, lay.NEXT) or t == lay.HEND:
            return self._body(p)
        return None

    def _track(self, p, t, body):
        lay, L, V = self.lay, self.L, self.V
        st = lay.st.get(t, 0)
        if st and body:
            L["cont"], V["cont"] = p, st
        if t == lay.S_LT:
            L["lt"], V["lt"] = p, self._lt_kinds(p)
        elif t == lay.S_HEQ:
            L["heq"], V["heq"] = p, self._heq_kinds(p)
        elif t in (lay.S_HDQ, lay.S_HSQ):
            L["hval"] = p
            if self.rb(p) not in URLB:
                L["bad"] = p
        if t in (lay.SKIP, lay.CHUNK):            # chunk framing (a page break is not: an href may span one)
            L["frame"] = p
        if t == lay.MORE:
            L["more"] = p
        if t == lay.HEND:
            L["hend"] = p
        if body and t in lay.ln_of:
            L["ln"], V["ln"] = p, lay.ln_of[t]
            self.lnpos[lay.ln_of[t]] = p
        if body and lay.compact and lay.textish(t):
            self.tpos.append(p)
        if body and (t in lay.byte_of or t in lay.word_of or t == lay.HEAD or t in lay.ln_of):
            white = t == lay.HEAD or (t in lay.byte_of and lay.byte_of[t] in (10, 32))
            nl = t == lay.HEAD or (t in lay.byte_of and lay.byte_of[t] == 10)
            L["txt"], V["txt"] = p, (white, nl)

    def _compact(self, p, t, mk):
        """The compact page block (docs/tfs_fetch_compact.md), forced after a body's <|fetch_done|>:
        <|pc_url|> URL <|pc_links|> (<|ln_k|> URL_k)* <|pc_text|> TEXT <|pc_end|>; False: not in one."""
        lay, f = self.lay, self.lay.f
        if t == f.DONE and self.prev_mark is not None and self.prev_mark[1] in (lay.HEND, lay.CHUNK, lay.NEXT):
            h = self.last("hend")
            url = F2.base_url(self.base)
            blk = [lay.PCU] + [lay.utok[c] for c in url] + [lay.PCL]
            links = [self.links[q] for q in sorted(self.links) if q > h]
            for k, u in enumerate(links):
                self.tab[k + 1] = (p + 1 + len(blk), u)
                blk += [lay.LN[k]] + [lay.utok[c] for c in u]
            blk += [lay.PCT] + [self.toks[q] for q in self.tpos if q > h] + [lay.PCE]
            self.cb = (p, blk, len(links))
            self.blocks.append((p, blk))
            return blk[0]
        if mk in (lay.PCU, lay.PCL, lay.PCT, lay.PCE) and self.cb is not None:
            i = p - self.cb[0]
            return self.cb[1][i] if i < len(self.cb[1]) else None
        return False

    def _lt_kinds(self, q):
        iq = q - 1 - self.Dat[q]

        def b(k):
            return self.tape(iq + k)

        def nm(name, j0):
            return all(_ci(b(j0 + t), c) for t, c in enumerate(name)) and b(j0 + len(name)) in DELIM
        closing = b(1) == 47
        return dict(closing=closing, scr=nm("script", 1), sty=nm("style", 1), a=nm("a", 1),
                    head=any(nm(f"h{k}", 1) for k in range(1, 7)),
                    block=any(nm(n, 1) or (closing and nm(n, 2)) for n in BLOCK))

    def _heq_kinds(self, q):
        iq = q - 1 - self.Dat[q]
        v = [self.tape(iq + 2 + k) for k in range(8)]
        ab = v == list(b"https://") or v[:7] == list(b"http://")
        return dict(abs=ab, s1=v[0] == 47, s2=v[1] == 47)

    def _resume_ok(self, p):
        c = self.last("more")
        return (c >= 0 and self.d is not None and c > self.d and self.prev_mark is not None and
                self.prev_mark == (c + 4, self.lay.f.CUT))

    def _verdict(self, p):
        lay, f = self.lay, self.lay.f
        pm = self.prev_mark
        if pm is None or pm[1] != f.FETCH:
            return f.ERR
        url = lay.url_text(self.toks[pm[0] + 1:p])
        v = F2.verdict(url, self.allow)
        if v != "ok":
            return f.ERR if v == "err" else f.DENY
        self.cur, self.nred = url, 0
        self.how = "follow" if self.fol is not None and pm[0] == self.fol[0] + 1 else "call"
        self.blk = F2.block(lay, url)
        self.blk_at = p
        return self.blk[0]

    def _follow(self, p, t):
        lay, f = self.lay, self.lay.f
        k = lay.ln_of.get(t)
        if lay.compact:                   # the current page's table (the compact block), never the raw bytes
            e = self.tab.get(k) if k else None
            if e is None or e[0] <= self.last("hend"):
                return f.ERR
            self.fol = (p, e[1])
            self.follows.append((k, e[1]))
            return f.FETCH
        q = self.lnpos.get(k, -1) if k else -1
        if k is None or q <= self.last("hend") or (self.d or -1) > self.last("hend"):   # its href: on the tape
            return f.ERR
        self.fol = (p, self.links[q])
        self.follows.append((k, self.links[q]))
        return f.FETCH

    def _send(self, p):
        lay = self.lay
        b = self.toks[self.blk_at + 1:p + 1]
        sch = "http" if b[0] == lay.TCP else "https"
        ri = b.index(lay.REQ)
        target = "".join(chr(lay.byte_of[x]) for x in b[1:ri])
        req = bytes(lay.byte_of[x] for x in b[ri + 1:-1])
        body, term = self.respond(sch, target, req)
        body = list(body)[:F2.PMAX]
        self.fetched.append((sch, target, req))
        self.sent.append((self.cur, self.how))
        self.r = body + [term]
        self.d = self.D = p
        self.Dat[p] = p
        self.h, self.chunk = None, None
        self.base = self.cur

    def _head(self, p):
        lay, f = self.lay, self.lay.f
        i = p - self.d
        x = self.tape(i)
        if x >= 256:
            return f.FAIL
        if [self.tape(i - 3), self.tape(i - 2), self.tape(i - 1), x] == [13, 10, 13, 10]:
            return lay.HEND
        return lay.SKIP

    def _body(self, p):
        lay, f = self.lay, self.lay.f
        t = self.toks[p]
        G = self.D - self.d
        if t == lay.HEND and self.h is None:
            self.h = p
            self.info = F2.head_info(self.r, p - self.d - 1)
        if t == lay.CHUNK:
            i0 = p - self.D - 1
            m = 0
            while m < 6 and F2.is_hex(self.tape(i0 - 2 - m)):
                m += 1
            self.chunk = (p - G, sum(F2.hexval(self.tape(i0 - 2 - k)) * 16 ** k for k in range(m)))
        h = self.h
        st, mode, cl, loc = self.info
        vp = p - G
        i = p - self.D
        x = self.tape(i)
        if p == h:
            if F2.encoding_refused(self.r, p - self.d - 1) is not None:     # compressed: refused (v2's rule)
                return f.FAIL
            if st in F2.REDIRS:
                if loc is None or F2.verdict(loc, self.allow) != "ok" or self.nred >= F2.MAXREDIR:
                    return f.FAIL
                self.nred += 1
                self.cur = loc
                self.how = "redirect"
                return lay.REDIR[self.nred - 1]
            if st != 200:
                return f.FAIL
        term = {F2.EOF: f.FAIL, F2.CUT: f.CUT, F2.CFAIL: f.FAIL, F2.NONE: f.FAIL}
        if mode == "length":
            o = vp - h
            if o == cl:
                return f.DONE
            if x >= 256:
                return term[x]
            return self._content(p, i, x, last=(o == cl - 1))
        if mode == "eof":
            if x >= 256:
                return f.DONE if x == F2.EOF else term[x]
            return self._content(p, i, x, last=False)
        q1 = vp + 1
        if self.chunk is not None:
            k, n = self.chunk
            if q1 <= k + n:
                if x >= 256:
                    return term[x]
                return self._content(p, i, x, last=(q1 == k + n))
            if n == 0:
                if q1 == k + 1:
                    return lay.SKIP if x == 13 else (f.CUT if x == F2.CUT else f.FAIL)
                return f.DONE if x == 10 else (f.CUT if x == F2.CUT else f.FAIL)
            if q1 == k + n + 1:
                return lay.SKIP if x == 13 else (f.CUT if x == F2.CUT else f.FAIL)
            if q1 == k + n + 2:
                return lay.SKIP if x == 10 else (f.CUT if x == F2.CUT else f.FAIL)
            s = k + n + 3
        else:
            s = h + 1
        u = q1 - s
        if x >= 256:
            return term[x]
        if u >= 1 and self.tape(i - 1) == 13:
            return lay.CHUNK if x == 10 else f.FAIL
        if F2.is_hex(x) and u <= 5:
            return lay.SKIP
        if x == 13 and u >= 1:
            return lay.SKIP
        return f.FAIL

    # -- a content byte x = r[i] at position p + 1: the HTML state machine --------------------------------------
    def _content(self, p, i, x, last):
        lay, h = self.lay, self.h
        seg = self.last("resume") if self.last("resume") > h else h
        if p + 1 - seg > self.page:
            return lay.MORE
        S = self.V["cont"] if self.last("cont") > h else ST_DATA
        tp = self.tape
        if S == ST_DATA:
            return self._data(p, i, x, last)
        if S in (ST_LT, ST_TAG):
            if x == 62:
                return self._close()
            if x == 34:
                return lay.S_DQ
            if x == 39:
                return lay.S_SQ
            if x == 61:
                k = self.V.get("lt")
                if (k is not None and k["a"] and not k["closing"] and
                        all(_ci(tp(i - 4 + j), c) for j, c in enumerate("href")) and tp(i - 5) in WS and
                        not self.last("heq") > self.last("lt")):
                    return lay.S_HEQ
            return lay.S_TAG
        if S == ST_HEQ:
            return {34: lay.S_HDQ, 39: lay.S_HSQ}.get(x) or (self._close() if x == 62 else lay.S_TAG)
        if S == ST_DQ:
            return lay.S_TAG if x == 34 else lay.S_DQ
        if S == ST_SQ:
            return lay.S_TAG if x == 39 else lay.S_SQ
        if S in (ST_HDQ, ST_HDF, ST_HSQ, ST_HSF):
            q = 34 if S in (ST_HDQ, ST_HDF) else 39
            if x == q:
                return self._closing(p)
            if x == 35 and S in (ST_HDQ, ST_HSQ):
                return lay.S_HDF if S == ST_HDQ else lay.S_HSF
            return {ST_HDQ: lay.S_HDQ, ST_HDF: lay.S_HDF, ST_HSQ: lay.S_HSQ, ST_HSF: lay.S_HSF}[S]
        if S == ST_CMT:
            return lay.S_SKIP if (x == 62 and tp(i - 1) == 45 and tp(i - 2) == 45) else lay.S_CMT
        name = "/script" if S == ST_SCR else "/style"
        if x == 60 and all(_ci(tp(i + 1 + j), c) for j, c in enumerate(name)) and tp(i + 1 + len(name)) in DELIM:
            return lay.S_LT
        return lay.S_SCR if S == ST_SCR else lay.S_STY

    def _textish(self):
        """(exists after the head, white, newline-like) of the latest text token in this page."""
        if self.last("txt") > self.h:
            return True, self.V["txt"][0], self.V["txt"][1]
        return False, False, False

    def _close(self):
        lay, k = self.lay, self.V["lt"]
        if k["scr"]:
            return lay.S_SCR
        if k["sty"]:
            return lay.S_STY
        if k["head"]:
            return lay.HEAD
        if k["block"]:
            ex, _, nl = self._textish()
            return lay.tid[10] if ex and not nl else lay.S_SKIP
        return lay.S_SKIP

    def _closing(self, p):
        lay = self.lay
        hq = self.last("heq")
        tb = lambda q: q - 1 - self.Dat[q]  # noqa: E731   (the tape byte a token stands for)
        vs, lu = tb(hq) + 2, self.last("hval")
        n = tb(lu) - vs + 1 if lu > hq else 0
        bad = self.last("bad")
        if n < 1 or self.last("frame") > hq or (bad > hq and tb(bad) >= vs):
            return lay.S_TAG
        kd = self.V["heq"]
        if not kd["abs"] and kd["s1"] and kd["s2"]:
            return lay.S_TAG
        href = "".join(chr(self.tape(vs + j)) for j in range(n))
        url = resolve(self.base, href)
        if url is None or len(url) > UMAX:
            return lay.S_TAG
        count = self.V["ln"] if self.last("ln") > self.h else 0
        if count >= K:
            return lay.S_TAG
        self.links[p + 1] = url
        return lay.LN[count]

    def _data(self, p, i, x, last):
        lay, toks, h, tp = self.lay, self.toks, self.h, self.tape
        q = p + 1
        if x == 60 and tagch(tp(i + 1)):
            return lay.S_CMT if (tp(i + 1), tp(i + 2), tp(i + 3)) == (33, 45, 45) else lay.S_LT
        for m in range(1, 6):                         # an entity's tail: its head (a TEXT token) m back
            if q - m > h and toks[q - m] in lay.byte_of:
                for e, _ in F2.ENTS:
                    if len(e) > m and all(tp(i - m + k) == ord(e[k]) for k in range(len(e))):
                        return lay.S_SKIP
        for e, v in F2.ENTS:
            if all(tp(i + k) == ord(e[k]) for k in range(len(e))):
                return lay.tid[v]
        if x in WS:
            ex, white, _ = self._textish()
            return lay.tid[32] if ex and not white else lay.S_SKIP
        if x not in lay.tid:
            return lay.S_SKIP
        if chr(x) in F2.WORDCH and lay.words and (last or not (tp(i + 1) < 256 and chr(tp(i + 1)) in F2.WORDCH)):
            run = [chr(x)]
            j = p
            while j > h and toks[j] in lay.byte_of and chr(lay.byte_of[toks[j]]) in F2.WORDCH and len(run) <= F2.WMAX:
                run.append(chr(lay.byte_of[toks[j]]))
                j -= 1
            w = "".join(reversed(run))
            if w in lay.words:
                return lay.words[w]
        return lay.tid[x]


def forced(lay, allow, respond, toks, page=PAGE):
    sp = Spec(lay, allow, respond, page)
    return [sp.feed(t) for t in toks]


# ---- the mask part's meaning ---------------------------------------------------------------------------------

NU = len(F2.URLCH)        # 73 URL characters
NCLS = K + NU + 4         # 0 other, 1..K links, then the URL characters (v2's), <|fetch_end|>, <|follow|>, <|fetch|>
C_END, C_FOL, C_FETCH = K + NU + 1, K + NU + 2, K + NU + 3


def check_starts(starts, allow):
    out = []
    for s in starts or []:
        if F2.verdict(s, allow) != "ok" or len(s) > UMAX or any(c not in F2.URLCH for c in s):
            raise ValueError(f"fetch3: a start point must be an allowed URL of <= {UMAX} URL characters: {s!r}")
        if s not in out:
            out.append(s)
    return out


def classes(lay, V):
    """(V, NCLS) one-hot: every token's mask class."""
    Mk = np.zeros((V, NCLS), I64)
    cls = np.zeros(V, I64)
    for k, t in enumerate(lay.LN):
        cls[t] = 1 + k
    for i, c in enumerate(F2.URLCH):
        cls[lay.utok[c]] = K + 1 + i
    cls[lay.f.END], cls[lay.FOLLOW], cls[lay.f.FETCH] = C_END, C_FOL, C_FETCH
    Mk[np.arange(V), cls] = 1
    return Mk


class MaskSpec:
    """feed(t) -> (deny set of classes, active) for the next position. With no start points the mask is off."""

    def __init__(self, lay, starts):
        self.lay, self.starts = lay, list(starts)
        self.toks = []
        self.mark = None
        self.hend = -1
        self.ln = (-1, 0)

    def feed(self, t):
        lay, f, p = self.lay, self.lay.f, len(self.toks)
        self.toks.append(t)
        if t in lay.mk:
            self.mark = (p, t)
        mk = self.mark[1] if self.mark else None
        if t == lay.HEND:
            self.hend = p
        if t in lay.ln_of and mk in ((lay.PCL,) if lay.compact else (lay.HEND, lay.CHUNK, lay.NEXT)):
            self.ln = (p, lay.ln_of[t])        # compaction: the links of the table, not of the raw page
        if not self.starts:
            return set(), 0
        n = self.ln[1] if self.ln[0] > self.hend else 0
        in_call = mk == f.FETCH
        in_fol = t == lay.FOLLOW
        if in_fol:
            return set(range(NCLS)) - {k for k in range(1, n + 1)}, 1
        if in_call:
            m = p - self.mark[0]
            written = self.toks[self.mark[0] + 1:p + 1]
            ok = set()
            for s in self.starts:
                if len(s) >= m and [lay.utok[c] for c in s[:m]] == written:
                    ok.add(K + 1 + F2.URLCH.index(s[m]) if m < len(s) else C_END)
            return set(range(NCLS)) - ok, 1
        return ({C_FOL} if n == 0 else set()), 0


# ---- the net's embedding ---------------------------------------------------------------------------------------

def features(lay):
    """The token embedding for vocabulary V (fetch3_net.tern's feature names)."""
    base = F2.features(lay)

    def table(fnames, V):
        col = {n: i for i, n in enumerate(fnames)}
        Ft = base(fnames, V)                      # v2's columns (v3's markers are in lay.mk: MARK, MK)
        for t, s in lay.st.items():
            Ft[t, col["ST"]] = s
            Ft[t, col["ISST"]] = 1
        for k, t in enumerate(lay.LN):
            Ft[t, col["ISLN"]] = 1
            Ft[t, col["LNV"]] = k + 1
            Ft[t, col[f"LN{k + 1}"]] = 1
        for i, t in enumerate(lay.PG):
            Ft[t, col["PG"]] = i
        Ft[lay.FOLLOW, col["F_FOL"]] = 1
        Ft[lay.NEXT, col["F_NEXT"]] = 1
        Ft[lay.MORE, col["F_MORE"]] = 1
        Ft[lay.HEAD, col["ISHEAD"]] = 1
        for t in (lay.SKIP, lay.CHUNK, lay.MORE):
            Ft[t, col["F_FRAME"]] = 1
        if "F_CFRAME" in col:                     # chunk framing only (a page break inside an href is allowed)
            for t in (lay.SKIP, lay.CHUNK):
                Ft[t, col["F_CFRAME"]] = 1
        Ft[lay.S_LT, col["F_LT"]] = 1
        Ft[lay.S_HEQ, col["F_HEQ"]] = 1
        Ft[[lay.S_HDQ, lay.S_HSQ], col["F_HVAL"]] = 1
        for t in lay.rawb:
            Ft[t, col["RAWB"]] = 1
        for b, t in lay.tid.items():
            Ft[t, col["TEXTISH"]] = 1
            if b in (10, 32):
                Ft[t, col["ISWH"]] = 1
            if b == 10:
                Ft[t, col["ISNL"]] = 1
        for t in [*lay.word_of, *lay.LN, lay.HEAD]:
            Ft[t, col["TEXTISH"]] = 1
        Ft[lay.HEAD, col["ISWH"]] = Ft[lay.HEAD, col["ISNL"]] = 1
        for i, c in enumerate(F2.URLCH):
            Ft[lay.utok[c], col["CHI"]] = i + 1
        return Ft
    return table


F2_FEATS = {"MARK", "MK", "RFLAG", "RN", "USRC", "F_END", "F_SEND", "F_HEND", "F_CHUNK", "F_SCH", "F_TLS", "SKIP",
            "ISTEXT", "TBYTE", "ISWC", "WCI", "ISSP", "ISWORD", "ISUC", "TAPE"}


def export(out_dir, lay, V, allow, part="force", starts=(), page=PAGE):
    """Build the net (fetch3_net.tern in the interpreter); compaction is the layout's (lay.compact)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    allow = F.parse_allow(allow)
    words = ",".join(f"{w}:{t}" for w, t in sorted(lay.words.items())) or "-"
    (out_dir / "words.txt").write_text(words + "\n")
    r = subprocess.run([sys.executable, "-m", "tern", str(NET), str(lay.lo3), str(lay.lo), str(lay.f.lo), str(V),
                        ",".join(allow) or "-", str(out_dir / "words.txt"), part, ",".join(starts) or "-",
                        str(page) + ("c" if lay.compact else ""), lay.agent, str(out_dir)],
                       capture_output=True, text=True, cwd=ROOT, timeout=3600)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-4000:] + r.stdout[-2000:])
    (out_dir / "allow.txt").write_text("".join(o + "\n" for o in allow))
    (out_dir / "starts.txt").write_text("".join(o + "\n" for o in starts))
    F2.write_meta(out_dir, lay.agent)
    (out_dir / "v3version.txt").write_text(NET3_VERSION + "\n")
    (out_dir / "layout.json").write_text(json.dumps([lay.lo3, lay.lo, lay.f.lo, V, page, part] +
                                                    ([1] if lay.compact else [])) + "\n")
    return r.stdout


NET3_VERSION = "2"        # fetch3's own (v3version.txt): 2 = hrefs in tape bytes across page breaks, the '?' fix


def current3(d):
    p = Path(d) / "v3version.txt"
    return p.exists() and p.read_text().strip() == NET3_VERSION


class Fetch3Net(F2.Fetch2Net):
    """The exported force part: v2's runner (the tapes in the copy head's virtual entries) with v3's embedding."""

    def __init__(self, d, lay, V):
        CI.Circuit.__init__(self, d, features(lay), V)
        self.lay = lay
        p = Path(d) / "allow.txt"
        self.allow = p.read_text().split() if p.exists() else []
        self.agent = F2.read_agent(d)
        self.win = [self.fnames.index(f"W{o - WIN0}") for o in range(WIN0, WIN1 + 1)]
        self.wt = self.fnames.index("TAPE")
        self.vheads = {"copy"} | {h.name for h in self.heads if h.name.endswith("_tv")}   # the tape's readers

    def page(self, dec, r):
        dec.virtual = []
        dec.add_virtual(tape_entries(r))

    def forward(self, toks, tapes=None):
        """v2's teacher-forced forward (every tape's entries, each masked to its own positions), v3's window."""
        if not tapes:
            return CI.Circuit.forward(self, toks)
        T = len(toks)
        own = np.full(T, -1)
        for m, (d, _) in enumerate(tapes):
            own[d:] = m
        vals = self.inputs(toks)
        causal = np.tril(np.ones((T, T), dtype=bool))
        ents = [tape_entries(r) for _, r in tapes]
        exact = {}
        for dd in range(1, self.depth + 1):
            for h in self.by_depth[dd][3]:
                Q, K_ = h.form(h.q, vals), h.form(h.k, vals)
                Vv = vals[h.v].T
                mask = causal
                if h.name in self.vheads:
                    Ks, Vs, ms = [K_], [Vv], [causal]
                    for m, e in enumerate(ents):
                        K2, V2 = self.virtual_kv(h, e)
                        Ks.append(K2)
                        Vs.append(V2)
                        ms.append(np.repeat((own == m)[:, None], len(e), 1))
                    K_, Vv, mask = np.concatenate(Ks), np.concatenate(Vs), np.concatenate(ms, 1)
                vals[h.o], exact[h.name] = (a.T if i == 0 else a for i, a in enumerate(self.attend(h, Q, K_, Vv, mask)))
            self.ops(dd, vals)
        return dict(code=vals[self.out_w].T, fire=vals[self.fire_w], notfire=vals[self.notfire_w], vals=vals,
                    exact=exact)


WIN0, WIN1 = -28, 8


def tape_entries(r):
    """The copy head's virtual entries for a tape r: v2's, with v3's wider window (r[i - 28] .. r[i + 8])."""
    n = len(r) - 1
    ext = list(r) + [r[-1]]
    out = []
    for i in range(n + 2):
        w = []
        for o in range(WIN0, WIN1 + 1):
            j = i + o
            w.append(F2.NONE if j < 0 else (ext[j] if j <= n + 1 else ext[-1]))
        out.append((F2.VB + i, w))
    return out


class Fetch3Tool:
    """tfs/tools.py's entry (the force part). Config: allow, fetch_lo / fetch2_lo (the blocks v3 extends), words (the
    dictionary, as v2's), page (positions per page segment), starts (the mask part's start points; [] = off), agent
    (the request's User-Agent, as v2's), compact (page compaction: the block grows by four tokens, a compact page
    block after every body, follows read from it; docs/tfs_fetch_compact.md)."""
    kind, priority, dirname = "fetch3", 4.5, "fetch3"        # before v2 (5) and v1 (6): it answers all they answer
    part, mode = "force", "force"
    companions = ("fetch3_mask",)
    S = S

    @staticmethod
    def size(compact=False, **_):
        return SC if compact else S

    def __init__(self, lo, V, allow=None, fetch_lo=None, fetch2_lo=None, words=None, page=PAGE, starts=None,
                 agent=F2.AGENT, compact=False, **_):
        if fetch_lo is None or fetch2_lo is None:
            raise ValueError("fetch3 extends fetch and fetch2: merge with --tools fetch,fetch2,fetch3")
        self.lo, self.V = lo, V
        self.S = SC if compact else S
        self.lay = Layout(lo, fetch2_lo, fetch_lo, [tuple(x) for x in (words or [])], agent or F2.AGENT, compact)
        self.allow = F.parse_allow(allow)
        self.page = int(page)
        self.starts = check_starts(starts, self.allow)
        self.net = None
        self.respond = lambda sch, target, req: ([], F2.CFAIL)

    def config(self):
        c = {"allow": self.allow, "fetch_lo": self.lay.f.lo, "fetch2_lo": self.lay.lo, "page": self.page,
             "starts": self.starts, "words": sorted([w, t] for w, t in self.lay.words.items())}
        if self.lay.agent != F2.AGENT:
            c["agent"] = self.lay.agent
        if self.lay.compact:
            c["compact"] = True
        return c

    def echo_spans(self):
        """The echo's span (docs/tfs_echo.md): with compaction, the compact page's text, between <|pc_text|> and
        <|pc_end|> (the block's last token, so it is the latest result marker after a page); quoted (page text).
        Without it none (as before)."""
        if not self.lay.compact:
            return None
        from .tools import Span
        return [Span([self.lay.PCT], [self.lay.PCE], quotes=True)]

    def tokens(self):
        return self.lay.tokens()

    def reserved(self):
        return self.lay.reserved()

    def export(self, d):
        return export(d, self.lay, self.V, self.allow, self.part, self.starts, self.page)

    def load(self, d):
        if not F2.current(d, self.lay.agent) or not current3(d):
            d = F2.reexport(self, d)
        self.net = Fetch3Net(d, self.lay, self.V)
        if self.net.allow != self.allow:
            raise ValueError(f"{d}: the net allows {self.net.allow}, tools.json {self.allow}")
        return self.net

    def spec(self):
        return Spec(self.lay, self.allow, self.respond, self.page)


class Fetch3MaskTool(Fetch3Tool):
    """The mask part (tfs/tools.py's masking mode): no tokens of its own, never fires; with start points, inside a
    model-written <|fetch|> call every token that does not continue a start point is denied (B nb lower), and after
    <|follow|> every token but the current page's links; <|follow|> itself is denied while the page has no link."""
    kind, priority, dirname = "fetch3_mask", 8.5, "fetch3_mask"
    part, mode = "mask", "mask"
    companions = ()

    @staticmethod
    def size(**_):
        return 0

    def __init__(self, lo, V, **cfg):
        super().__init__(lo, V, **cfg)
        self.S = 0

    def echo_spans(self):
        return None

    def tokens(self):
        return []

    def reserved(self):
        return []

    def load(self, d):
        from .tools import MaskCircuit
        if not F2.current(d, self.lay.agent):           # an older mask (46 URL-character classes): again
            d = F2.reexport(self, d)
        self.net = MaskCircuit(d, features(self.lay), self.V)
        st = (Path(d) / "starts.txt")
        got = st.read_text().split() if st.exists() else []
        if got != self.starts:
            raise ValueError(f"{d}: the mask net's start points {got}, tools.json {self.starts}")
        return self.net

    def spec(self):
        from .tools import _NoForce
        return _NoForce()

    def mask_spec(self):
        return MaskSpec(self.lay, self.starts)

    def classes(self):
        return classes(self.lay, self.V)

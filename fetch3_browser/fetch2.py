"""Fetch v2 (docs/tfs_fetch2.md): the fetch tool with the protocol in the net. The model writes the same call as v1
(<|fetch|> URL <|fetch_end|>, tfs/fetch.py); the compiled net examples/wire/fetch2_net.tern then forces the HTTP request
(the connect target and the exact request bytes, built from the URL it checked), the runtime (tfs/fetch2_rt.py) sends
those bytes and pages the raw response bytes back, and the net parses the response as it copies it: status line,
headers (Content-Length, Transfer-Encoding: chunked, Location), redirects (re-checked against the allowlist, at most
three, the count carried in the tokens), the body's framing, and the body's text (tags stripped, entities decoded,
whitespace collapsed, known words as the model's own tokens). This module is the meaning (the spec the net is checked
against), the token layout, the net's embedding and the tool's entry.

Tokens: v2 extends v1's block (it needs the fetch tool, fetch_lo) with S = 62 ids from LO:

    LO + 0..51     <|b_XX|>   the text bytes that are not v1 URL characters (A-Z, space, \\t \\n \\r and the other
                              printable ASCII): with v1's 46 <|url_c|> tokens they make the 98 TEXT tokens
    LO + 52, 53    <|h_tcp|> <|h_tls|>     the connect line follows (plain TCP / a TLS tunnel)
    LO + 54        <|h_req|>               the request bytes follow
    LO + 55        <|h_send|>              the runtime sends the request and pages the response in
    LO + 56        <|h_skip|>              a response byte that is not text (head, markup, framing, collapsed space)
    LO + 57        <|h_hend|>              the response head's last byte (the LF of CRLF CRLF)
    LO + 58        <|h_chunk|>             a chunk-size line's LF (chunked transfer encoding)
    LO + 59..61    <|h_redir1|> .. <|h_redir3|>   a redirect followed (the count is in the token)

Every v2 token is reserved (only forced). The transcript of one fetch:

    <|fetch|> URL <|fetch_end|>                                      the model
      <|fetch_err|> | <|fetch_deny|>                                 malformed / not allowed (v1's rule)
      <|h_tcp|> HOST:PORT <|h_req|> REQUEST <|h_send|>               the request, every byte a TEXT token
      R0 R1 ... <|h_hend|> B0 B1 ...                                 one token per response byte, in order
      <|fetch_done|> | <|fetch_fail|> | <|fetch_cut|>                the end of the response (replaces a byte)
      | <|h_redirN|> <|h_tcp|> ... <|h_send|> ...                    a redirect: the next request, from Location

A response byte's token: <|h_skip|> (not text), <|h_hend|>, <|h_chunk|>, a TEXT token (the byte, or the decoded
entity, or a space for kept whitespace), or a word of the base vocabulary (the last character of a word that is in the
net's dictionary). The byte is always the raw byte at the same offset, so the transcript is a 1-to-1 reading of the
response: the raw bytes themselves are in the runtime's log (SHA-256).

The response: the runtime pages bytes r[0..n-1] and a terminator r[n] (EOF: the connection closed, CUT: a cap was hit,
CFAIL: no connection / refused) into the copy head's virtual entries; position d + 1 + i reads r[i], d the latest
<|h_send|>.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from ._paths import ROOT

from . import circuit as CI
from . import fetch as F

I64 = np.int64
EXTRA = [b for b in [9, 10, 13] + list(range(32, 127)) if chr(b) not in F.CHARS]          # 52 bytes
TEXT = sorted([9, 10, 13] + list(range(32, 127)))                                           # 98 bytes
MARKS = ["h_tcp", "h_tls", "h_req", "h_send", "h_skip", "h_hend", "h_chunk", "h_redir1", "h_redir2", "h_redir3"]
NAMES = [f"b_{b:02x}" for b in EXTRA] + MARKS
S = len(NAMES)                      # 62
NEXTRA = len(EXTRA)
EOF, CUT, CFAIL, NONE = 256, 257, 258, 300          # terminator values on the tape; NONE: before the start / past it
UMAX = F.UMAX                       # the call's URL: 1..64 characters (v1)
UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
URLCH = F.CHARS + UPPER + "#"       # v2's URL characters: v1's 46, upper case (so %XX escapes either case), '#'
URLEXTRA = UPPER + "#"              # the ones that are v2 TEXT tokens (<|b_XX|>), not v1 character tokens
LMAX = 63                           # a Location value: 1..63 characters, then CR
WMAX = 12                           # dictionary words: 1..12 characters of a-z 0-9
WIN = (-28, 5)                      # the tape's window: every entry carries r[i + o] for o in WIN[0] .. WIN[1]
NWIN = WIN[1] - WIN[0] + 1
VB = 16384
PMAX = 16383 - 2                    # response bytes at most (VB + n + 1 < 32768)
MAXREDIR = 3
WORDCH = "abcdefghijklmnopqrstuvwxyz0123456789"
WS = (9, 10, 13, 32)
REDIRS = (301, 302, 303, 307, 308)
ENTS = [("&amp;", 38), ("&lt;", 60), ("&gt;", 62), ("&quot;", 34), ("&#39;", 39), ("&apos;", 39), ("&nbsp;", 32)]
# The User-Agent the request sends, baked into the net at export (--fetch-agent, a tool config "agent"). A placeholder:
# the final string is not decided, ternary.dev is not owned yet, and some sites (Wikipedia) want a working contact
# (docs/tfs_fetch2.md "User-Agent").
AGENT = "tern-fetch/0.1 (+https://ternary.dev)"
NET_VERSION = "2"                   # the exported nets' behaviour (version.txt): 2 = encodings, URLs, the agent


def check_agent(agent):
    if not (1 <= len(agent) <= 120 and all(32 <= ord(c) <= 126 for c in agent)):
        raise ValueError(f"fetch2: a User-Agent is 1..120 printable ASCII characters: {agent!r}")
    return agent


def tail(agent=AGENT):
    return f"\r\nUser-Agent: {check_agent(agent)}\r\nAccept-Encoding: identity\r\nConnection: close\r\n\r\n"


TAIL = tail()
HTTP1 = " HTTP/1.1\r\nHost: "
NET = ROOT / "examples/wire/fetch2_net.tern"
# marker kinds (the net's MK feature)
MK = {"fetch": 1, "fetch_end": 2, "err": 3, "deny": 4, "done": 5, "fail": 6, "cut": 7, "h_tcp": 8, "h_tls": 9,
      "h_req": 10, "h_send": 11, "h_hend": 12, "h_chunk": 13, "h_redir1": 14, "h_redir2": 15, "h_redir3": 16}


class Layout:
    """lo: the first v2 id; flay: v1's layout (tfs/fetch.py Layout at fetch_lo)."""

    def __init__(self, lo, fetch_lo, words=(), agent=AGENT):
        self.lo = lo
        self.agent = check_agent(agent)
        self.f = F.Layout(fetch_lo)
        for i, n in enumerate(MARKS):
            setattr(self, n[2:].upper(), lo + NEXTRA + i)
        self.REDIR = [self.REDIR1, self.REDIR2, self.REDIR3]
        self.tid = {}                                     # byte -> TEXT token id
        for b in TEXT:
            c = chr(b)
            self.tid[b] = self.f.chars[c] if c in F.CHARS else lo + EXTRA.index(b)
        self.byte_of = {t: b for b, t in self.tid.items()}
        self.utok = {c: self.tid[ord(c)] for c in URLCH}  # a URL character -> its token (v1's, or a TEXT byte's)
        self.uchar = {t: c for c, t in self.utok.items()}
        self.upper_ids = [self.utok[c] for c in URLEXTRA]  # the URL characters a model writes as v2 TEXT tokens
        self.words = {}                                   # word (str) -> base id
        for w, t in words:
            if not (1 <= len(w) <= WMAX and all(c in WORDCH for c in w)):
                raise ValueError(f"fetch2: a dictionary word of 1..{WMAX} characters a-z 0-9: {w!r}")
            self.words[w] = int(t)
        self.word_of = {t: w for w, t in self.words.items()}
        f = self.f
        self.mk = {f.FETCH: 1, f.END: 2, f.ERR: 3, f.DENY: 4, f.DONE: 5, f.FAIL: 6, f.CUT: 7}
        for n in MARKS:
            if n != "h_skip":
                self.mk[getattr(self, n[2:].upper())] = MK[n]

    def name(self, t):
        if self.lo <= t < self.lo + S:
            return f"<|{NAMES[t - self.lo]}|>"
        return self.f.name(t)

    def tokens(self):
        return [f"<|{n}|>" for n in NAMES]

    def reserved(self):
        return range(self.lo, self.lo + S)

    def url_text(self, toks):
        """The URL a call's tokens spell (v2's URL characters), or None if one is not a URL character."""
        if any(t not in self.uchar for t in toks):
            return None
        return "".join(self.uchar[t] for t in toks)

    def call(self, url):
        """<|fetch|> URL <|fetch_end|> in v2's URL characters (upper case and '#' as TEXT tokens)."""
        return [self.f.FETCH, *[self.utok[c] for c in url], self.f.END]

    def text_ids(self, s):
        return [self.tid[b] for b in s.encode()]

    def show(self, toks, words=None):
        """A transcript as text (TEXT bytes as characters, markers by name, words in [brackets])."""
        out = []
        for t in toks:
            if t in self.byte_of:
                b = self.byte_of[t]
                out.append({9: "\\t", 10: "\\n", 13: "\\r"}.get(b, chr(b)))
            elif t in self.word_of:
                out.append(f"[{self.word_of[t]}]")
            elif t == self.SKIP:
                out.append("·")
            elif t in self.mk or self.lo <= t < self.lo + S or self.f.lo <= t < self.f.lo + F.S:
                out.append(self.name(t))
            else:
                out.append(f"<{t}>" if words is None else words(t))
        return "".join(out)


# ---- URLs (v1's rule, and the parts the request is built from) -------------------------------------------------

def base_url(url):
    """The URL without its fragment: what is checked against the allowlist and sent (the '#' and after never are)."""
    return url.split("#", 1)[0]


def verdict(url, allow):
    """v2's call / Location rule: 'err' | 'deny' | 'ok'. v1's (tfs/fetch.py verdict) over v2's URL characters (upper
    case, '#'), with the fragment dropped before the scheme and origin checks."""
    if url is None or not 1 <= len(url) <= UMAX or any(c not in URLCH for c in url):
        return "err"
    b = base_url(url)
    if not (b.startswith("https://") or b.startswith("http://")):
        return "err"
    for o in allow:
        if b == o or b.startswith(o + "/"):
            return "ok"
    return "deny"


def url_parts(url):
    """(scheme, authority, path, has_port): the request's pieces of an allowed URL (its fragment dropped)."""
    url = base_url(url)
    s0 = 8 if url.startswith("https://") else 7
    j = url.find("/", 8)
    he = j if j >= 0 else len(url)
    auth = url[s0:he]
    return url[:s0 - 3], auth, url[he:] or "/", ":" in auth


def connect_line(url):
    scheme, auth, _, hp = url_parts(url)
    return auth + ("" if hp else (":80" if scheme == "http" else ":443"))


def request(url, agent=AGENT):
    _, auth, path, _ = url_parts(url)
    return f"GET {path}{HTTP1}{auth}{tail(agent)}"


def block(lay, url):
    """The forced tokens after the call / redirect: the scheme marker, the connect line, <|h_req|>, the request,
    <|h_send|>."""
    sch = lay.TCP if url.startswith("http://") else lay.TLS
    return [sch, *lay.text_ids(connect_line(url)), lay.REQ, *lay.text_ids(request(url, lay.agent)), lay.SEND]


def location(head):
    """The latest Location value of a head (bytes) as the net reads it: '\\nlocation: ' (name in any case, one space),
    then 1..LMAX URL characters, then CR. None if there is none or it is not of that form."""
    best = None
    for m in re.finditer(rb"\n[Ll][Oo][Cc][Aa][Tt][Ii][Oo][Nn]: ", head):
        best = m.end()
    if best is None:
        return None
    j = best
    while j < len(head) and j - best < LMAX + 1 and chr(head[j]) in URLCH:
        j += 1
    if j - best < 1 or j - best > LMAX or j >= len(head) or head[j] != 13:
        return None
    return head[best:j].decode()


def _ci(s):
    return b"".join(b"[" + ch.upper().encode() + ch.lower().encode() + b"]" if ch.isalpha() else
                    re.escape(ch.encode()) for ch in s)


CL_RE = re.compile(rb"\n" + _ci("content-length:") + rb" ([0-9]{1,7})\r")
TE_RE = re.compile(rb"\n" + _ci("transfer-encoding:") + rb" " + _ci("chunked") + rb"\r")


def head_info(r, nh):
    """The head r[0 .. nh] (nh: the index of its last byte, the LF of CRLF CRLF): status, framing, Location."""
    head = bytes(r[:nh + 1])
    st = None
    if nh > 12 and head[:9] in (b"HTTP/1.1 ", b"HTTP/1.0 ") and all(48 <= c <= 57 for c in head[9:12]) and \
            head[12] in (32, 13):
        st = int(head[9:12])
    # the header flags fire at a line's last byte (CR) while the head is open: a match must end before nh - 2
    cl = None
    for m in CL_RE.finditer(head):
        if m.end() - 1 < nh:
            cl = int(m.group(1))
    te = any(m.end() - 1 < nh for m in TE_RE.finditer(head))
    return st, ("chunked" if te else ("length" if cl is not None else "eof")), cl, location(head)


CE_NAME = b"content-encoding:"


def encodings(r, nh):
    """The Content-Encoding headers of the head r[0 .. nh] as the net reads them: [(colon index, refused)] for every
    '\\ncontent-encoding:' (name in any case) in the head. Refused unless the value is exactly ' identity' (one space,
    either case) then CR; a value the head ends within ten bytes of the colon is refused too. Any refused one makes
    the response fail at its head's end (the request asks Accept-Encoding: identity; compressed bytes are never
    read)."""
    head = bytes(r[:nh + 1])
    out = []
    for m in re.finditer(rb"\n" + _ci(CE_NAME.decode()), head):
        c = m.end() - 1
        if c >= nh:
            continue
        ok = c + 10 <= nh and head[c + 1] == 32 and head[c + 2:c + 10].lower() == b"identity" and head[c + 10] == 13
        out.append((c, not ok))
    return out


def encoding_refused(r, nh):
    """The Content-Encoding value that makes the net refuse the response (bytes), or None."""
    for c, bad in encodings(r, nh):
        if bad:
            e = bytes(r[c + 1:nh + 1]).split(b"\r", 1)[0].strip()
            return e or b"(empty)"
    return None


def refusal(body):
    """Why the net will refuse a response the runtime received (bytes), as text for the runtime's log, or None: the
    reasons the net decides at the head's end that a reader of the transcript cannot see (the head is <|h_skip|>
    tokens). Today one: a Content-Encoding other than identity ("content-encoding: gzip")."""
    b = bytes(body[:PMAX])
    e = b.find(b"\r\n\r\n")
    if e < 0:
        return None
    enc = encoding_refused(list(b), e + 3)
    return None if enc is None else "content-encoding: " + enc.decode("ascii", "replace")[:40]


# ---- the spec --------------------------------------------------------------------------------------------------

def is_hex(b):
    return 48 <= b <= 57 or 97 <= b <= 102 or 65 <= b <= 70


def hexval(b):
    return b - 48 if b <= 57 else (b | 32) - 87


class Spec:
    """feed(t) -> the token the v2 net forces after t, or None. allow: the origins (v1's rule); respond(scheme, target,
    request bytes) -> (bytes, terminator) the runtime's response (terminator EOF / CUT / CFAIL), asked at <|h_send|>."""

    def __init__(self, lay, allow, respond):
        self.lay, self.allow, self.respond = lay, list(allow), respond
        self.toks = []
        self.mark = None              # (pos, token) of the latest marker
        self.prev_mark = None         # the marker before it
        self.cur = None               # the URL being fetched (the call's, then each Location's)
        self.nred = 0                 # redirects followed in this fetch
        self.blk = None               # the forced request block being emitted
        self.d = None                 # the latest <|h_send|>
        self.r = None                 # the tape: response bytes + terminator
        self.h = None                 # the head's end position (<|h_hend|>), None while in the head
        self.info = None
        self.chunk = None             # (pos, size) of the latest <|h_chunk|>
        self.fetched = []             # (scheme, target, request) of every send

    def tape(self, i):
        r = self.r
        if i < 0:
            return NONE
        return r[i] if i < len(r) else NONE

    def feed(self, t):
        lay, p = self.lay, len(self.toks)
        self.toks.append(t)
        if t in lay.mk:
            self.prev_mark, self.mark = self.mark, (p, t)
        mk = self.mark[1] if self.mark else None
        f = lay.f
        if t == f.END:
            return self._verdict(p)
        if mk in (lay.TCP, lay.TLS, lay.REQ) or mk in lay.REDIR:
            if mk in lay.REDIR and t == mk:
                self.blk = block(lay, self.cur)
                self.blk_at = p
                return self.blk[0]
            k = p - self.blk_at                               # the block's token p + 1 is blk[k]
            return self.blk[k] if k < len(self.blk) else None
        if mk == lay.SEND:
            if t == lay.SEND:
                self._send(p)
            return self._head(p)
        if mk in (lay.HEND, lay.CHUNK) or (t == lay.HEND):
            return self._body(p)
        return None

    def _verdict(self, p):
        lay, f = self.lay, self.lay.f
        pm = self.prev_mark
        if pm is None or pm[1] != f.FETCH:
            return f.ERR
        url = lay.url_text(self.toks[pm[0] + 1:p])
        v = verdict(url, self.allow)
        if v != "ok":
            return f.ERR if v == "err" else f.DENY
        self.cur, self.nred = url, 0
        self.blk = block(lay, url)
        self.blk_at = p
        return self.blk[0]

    def _send(self, p):
        lay = self.lay
        b = self.toks[self.blk_at + 1:p + 1]
        sch = "http" if b[0] == lay.TCP else "https"
        ri = b.index(lay.REQ)
        target = "".join(chr(lay.byte_of[x]) for x in b[1:ri])
        req = bytes(lay.byte_of[x] for x in b[ri + 1:-1])
        body, term = self.respond(sch, target, req)
        body = list(body)[:PMAX]
        self.fetched.append((sch, target, req))
        self.r = body + [term]
        self.d, self.h, self.chunk = p, None, None

    # the token at p + 1 while the head is open: raw index i = p - d
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
        if t == lay.HEND and self.h is None:
            self.h = p
            self.info = head_info(self.r, p - self.d - 1)
        if t == lay.CHUNK:
            i0 = p - self.d - 1
            m = 0
            while m < 6 and is_hex(self.tape(i0 - 2 - m)):
                m += 1
            self.chunk = (p, sum(hexval(self.tape(i0 - 2 - k)) * 16 ** k for k in range(m)))
        h, d = self.h, self.d
        st, mode, cl, loc = self.info
        i = p - d
        x = self.tape(i)
        if p == h:
            if encoding_refused(self.r, p - d - 1) is not None:     # compressed (or unknown) encoding: refused
                return f.FAIL
            if st in REDIRS:
                if loc is None or verdict(loc, self.allow) != "ok" or self.nred >= MAXREDIR:
                    return f.FAIL
                self.nred += 1
                self.cur = loc
                return lay.REDIR[self.nred - 1]
            if st != 200:
                return f.FAIL
        term = {EOF: f.FAIL, CUT: f.CUT, CFAIL: f.FAIL, NONE: f.FAIL}
        if mode == "length":
            o = p - h
            if o == cl:
                return f.DONE
            if x >= 256:
                return term[x]
            return self._content(p, i, x, last=(o == cl - 1))
        if mode == "eof":
            if x >= 256:
                return f.DONE if x == EOF else term[x]
            return self._content(p, i, x, last=False)
        # chunked
        q1 = p + 1
        if self.chunk is not None:
            k, n = self.chunk
            if q1 <= k + n:
                if x >= 256:
                    return term[x]
                return self._content(p, i, x, last=(q1 == k + n))
            if n == 0:
                if q1 == k + 1:
                    return lay.SKIP if x == 13 else (f.CUT if x == CUT else f.FAIL)
                return f.DONE if x == 10 else (f.CUT if x == CUT else f.FAIL)
            if q1 == k + n + 1:
                return lay.SKIP if x == 13 else (f.CUT if x == CUT else f.FAIL)
            if q1 == k + n + 2:
                return lay.SKIP if x == 10 else (f.CUT if x == CUT else f.FAIL)
            s = k + n + 3
        else:
            s = h + 1
        u = q1 - s
        if x >= 256:
            return term[x]
        if u >= 1 and self.tape(i - 1) == 13:
            return lay.CHUNK if x == 10 else f.FAIL
        if is_hex(x) and u <= 5:
            return lay.SKIP
        if x == 13 and u >= 1:
            return lay.SKIP
        return f.FAIL

    # a content byte (body data) x = r[i] at position p + 1
    def _content(self, p, i, x, last):
        lay, toks, h = self.lay, self.toks, self.h
        q = p + 1
        # in a tag: the latest '<' / '>' that was skipped, after the head
        in_tag = False
        for j in range(p, h, -1):
            if toks[j] == lay.SKIP and self.tape(j - self.d - 1) in (60, 62):
                in_tag = self.tape(j - self.d - 1) == 60
                break
        if x == 60 or in_tag:
            return lay.SKIP
        for m in range(1, 6):                         # an entity's tail: its head (a TEXT token) m back
            if q - m > h and toks[q - m] in lay.byte_of:
                for e, _ in ENTS:
                    if len(e) > m and all(self.tape(i - m + k) == ord(e[k]) for k in range(len(e))):
                        return lay.SKIP
        for e, v in ENTS:
            if all(self.tape(i + k) == ord(e[k]) for k in range(len(e))):
                return lay.tid[v]
        if x in WS:
            for j in range(p, h, -1):
                if toks[j] in lay.byte_of or toks[j] in lay.word_of:
                    return lay.SKIP if toks[j] == lay.tid[32] else lay.tid[32]
            return lay.SKIP
        if x not in lay.tid:
            return lay.SKIP
        if chr(x) in WORDCH and lay.words and (last or not (self.tape(i + 1) < 256 and chr(self.tape(i + 1)) in WORDCH)):
            run = [chr(x)]
            j = p
            while j > h and toks[j] in lay.byte_of and chr(lay.byte_of[toks[j]]) in WORDCH and len(run) <= WMAX:
                run.append(chr(lay.byte_of[toks[j]]))
                j -= 1
            w = "".join(reversed(run))
            if w in lay.words:
                return lay.words[w]
        return lay.tid[x]


def forced(lay, allow, respond, toks):
    sp = Spec(lay, allow, respond)
    return [sp.feed(t) for t in toks]


# ---- the net's embedding and the tape ----------------------------------------------------------------------------

def features(lay):
    """The token embedding for vocabulary V (fetch2_net.tern's feature names)."""
    def table(fnames, V):
        Ft = np.zeros((V, len(fnames)), I64)
        col = {n: i for i, n in enumerate(fnames)}
        ids = np.arange(V)
        for n, i in col.items():
            if n.startswith("code"):
                Ft[:, i] = 2 * ((ids >> int(n[4:])) & 1) - 1
        f = lay.f
        for t, k in lay.mk.items():
            Ft[t, col["MARK"]] = 1
            Ft[t, col["MK"]] = k
        for t in (f.FETCH, *lay.REDIR):
            Ft[t, col["RFLAG"]] = 1
        for i, t in enumerate(lay.REDIR):
            Ft[t, col["RN"]] = i + 1
        Ft[f.FETCH, col["USRC"]] = 1
        Ft[f.END, col["F_END"]] = 1
        Ft[lay.SEND, col["F_SEND"]] = 1
        Ft[lay.HEND, col["F_HEND"]] = 1
        Ft[lay.CHUNK, col["F_CHUNK"]] = 1
        Ft[[lay.TCP, lay.TLS], col["F_SCH"]] = 1
        Ft[lay.TLS, col["F_TLS"]] = 1
        for t in lay.utok.values():                      # v2's URL characters (v1's, upper case, '#')
            Ft[t, col["ISUC"]] = 1
        Ft[lay.SKIP, col["SKIP"]] = 1
        for b, t in lay.tid.items():
            Ft[t, col["ISTEXT"]] = 1
            Ft[t, col["TBYTE"]] = b
            if chr(b) in WORDCH:
                Ft[t, col["ISWC"]] = 1
                Ft[t, col["WCI"]] = WORDCH.index(chr(b))
            if b == 32:
                Ft[t, col["ISSP"]] = 1
        for t in lay.word_of:
            Ft[t, col["ISWORD"]] = 1
        return Ft
    return table


def tape_entries(r):
    """The copy head's virtual entries for a tape r (bytes + terminator): positions VB + i for i = 0 .. n + 1, each
    with its window of bytes r[i + o] (o in WIN; NONE outside the tape, the terminator repeated past its end)."""
    n = len(r) - 1
    ext = list(r) + [r[-1]]
    out = []
    for i in range(n + 2):
        w = []
        for o in range(WIN[0], WIN[1] + 1):
            j = i + o
            w.append(NONE if j < 0 else (ext[j] if j <= n + 1 else ext[-1]))
        out.append((VB + i, w))
    return out


def export(out_dir, lay, V, allow):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    allow = F.parse_allow(allow)
    words = ",".join(f"{w}:{t}" for w, t in sorted(lay.words.items())) or "-"
    (out_dir / "words.txt").write_text(words + "\n")
    r = subprocess.run([sys.executable, "-m", "tern", str(NET), str(lay.lo), str(lay.f.lo), str(V),
                        ",".join(allow) or "-", str(out_dir / "words.txt"), lay.agent, str(out_dir)],
                       capture_output=True, text=True, cwd=ROOT, timeout=3600)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-4000:] + r.stdout[-2000:])
    (out_dir / "allow.txt").write_text("".join(o + "\n" for o in allow))
    write_meta(out_dir, lay.agent)
    (out_dir / "layout.json").write_text(json.dumps([lay.lo, lay.f.lo, V]) + "\n")
    return r.stdout


def write_meta(d, agent):
    (Path(d) / "agent.txt").write_text(agent + "\n")
    (Path(d) / "version.txt").write_text(NET_VERSION + "\n")


def read_agent(d):
    """The User-Agent an exported net sends ("tern" for a net exported before it was configurable)."""
    p = Path(d) / "agent.txt"
    return p.read_text().rstrip("\n") if p.exists() else "tern"


def current(d, agent):
    """Is the exported net in d this code's (NET_VERSION) with this agent? An older one is exported again at load."""
    p = Path(d) / "version.txt"
    return p.exists() and p.read_text().strip() == NET_VERSION and read_agent(d) == agent


def reexport(tool, d, log=None):
    """An exported net older than this code (a step merged before): export it again, into a temporary directory (the
    step is left as it is). Returns that directory."""
    import tempfile
    nd = Path(tempfile.mkdtemp(prefix=f"tfs_{tool.kind}_net_"))
    print(f"{tool.kind}: {d} is a net older than this code (version {NET_VERSION}, User-Agent {tool.lay.agent!r}); "
          f"exported again into {nd}", file=log or sys.stderr)
    tool.export(nd)
    return nd


class Fetch2Net(CI.Circuit):
    """The exported net with the runtime's tapes: each response's tape in the virtual entries of the "copy" head while
    it is current (the decoder: `page(dec, r)` replaces it), and in forward(toks, tapes=[(d, r)]) every position sees
    the tape of its latest <|h_send|> (the entries of every tape, each masked to its own positions)."""

    def __init__(self, d, lay, V):
        super().__init__(d, features(lay), V)
        self.lay = lay
        p = Path(d) / "allow.txt"
        self.allow = p.read_text().split() if p.exists() else []
        self.agent = read_agent(d)
        self.win = [self.fnames.index(f"W{o - WIN[0]}") for o in range(WIN[0], WIN[1] + 1)]
        self.wt = self.fnames.index("TAPE")

    def virtual_kv(self, h, virtual):
        pos = np.array([p for p, _ in virtual], I64)
        vals = np.zeros((self.n, len(pos)), I64)
        feat = np.zeros((len(pos), len(self.fnames)), I64)
        feat[:, self.win] = np.array([w for _, w in virtual], I64)
        feat[:, self.wt] = 1
        vals[self.fw] = feat.T
        vals[self.pw] = self.pos_features(pos).T
        vals[self.cw] = self.cv[:, None]
        return h.form(h.k, vals), vals[h.v].T

    def page(self, dec, r):
        dec.virtual = []
        dec.add_virtual(tape_entries(r))

    def forward(self, toks, tapes=None):
        if not tapes:
            return super().forward(toks)
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
                Q, K = h.form(h.q, vals), h.form(h.k, vals)
                Vv = vals[h.v].T
                mask = causal
                if h.name == "copy":
                    Ks, Vs, ms = [K], [Vv], [causal]
                    for m, e in enumerate(ents):
                        K2, V2 = self.virtual_kv(h, e)
                        Ks.append(K2)
                        Vs.append(V2)
                        ms.append(np.repeat((own == m)[:, None], len(e), 1))
                    K, Vv, mask = np.concatenate(Ks), np.concatenate(Vs), np.concatenate(ms, 1)
                vals[h.o], exact[h.name] = (a.T if i == 0 else a for i, a in enumerate(self.attend(h, Q, K, Vv, mask)))
            self.ops(dd, vals)
        return dict(code=vals[self.out_w].T, fire=vals[self.fire_w], notfire=vals[self.notfire_w], vals=vals,
                    exact=exact)


class Fetch2Tool:
    """tfs/tools.py's entry. Config: allow (the origins compiled into the net), fetch_lo (v1's block, whose call and
    final markers v2 uses), words ([[word, id]]: the dictionary of base-vocabulary words the net writes as tokens),
    url_upper (the model may write upper case and '#' in a call: those 27 TEXT tokens are not reserved; set by merges
    since URLs took upper case, absent (False) in older steps, whose models never learned them), agent (the request's
    User-Agent, baked into the net; AGENT when absent)."""
    kind, priority, dirname = "fetch2", 5, "fetch2"          # before v1 (6): v2 answers <|fetch_end|>
    S = S

    @staticmethod
    def size(**_):
        return S

    def __init__(self, lo, V, allow=None, fetch_lo=None, words=None, url_upper=False, agent=AGENT, **_):
        if fetch_lo is None:
            raise ValueError("fetch2 extends the fetch tool: merge with --tools fetch,fetch2 (fetch_lo)")
        self.lo, self.V = lo, V
        self.url_upper = bool(url_upper)
        self.lay = Layout(lo, fetch_lo, [tuple(x) for x in (words or [])], agent or AGENT)
        self.allow = F.parse_allow(allow)
        self.net = None
        self.respond = lambda sch, target, req: ([], CFAIL)       # the spec's runtime: set by the runtime / tests

    def config(self):
        c = {"allow": self.allow, "fetch_lo": self.lay.f.lo, "words": sorted([w, t] for w, t in self.lay.words.items())}
        if self.url_upper:
            c["url_upper"] = True
        if self.lay.agent != AGENT:
            c["agent"] = self.lay.agent
        return c

    def tokens(self):
        return self.lay.tokens()

    def reserved(self):
        if self.url_upper:
            up = set(self.lay.upper_ids)
            return [t for t in self.lay.reserved() if t not in up]
        return self.lay.reserved()

    def export(self, d):
        return export(d, self.lay, self.V, self.allow)

    def load(self, d):
        if not current(d, self.lay.agent):
            d = reexport(self, d)
        self.net = Fetch2Net(d, self.lay, self.V)
        if self.net.allow != self.allow:
            raise ValueError(f"{d}: the net allows {self.net.allow}, tools.json {self.allow}")
        return self.net

    def spec(self):
        return Spec(self.lay, self.allow, self.respond)


def words_of(tokenizer_json, v0):
    """The dictionary from a tokenizer: every base id < v0 whose token is 1..WMAX characters of a-z 0-9."""
    vocab = json.loads(Path(tokenizer_json).read_text())["model"].get("vocab", {})
    if isinstance(vocab, list):
        vocab = {w: i for i, (w, _) in enumerate(vocab)}
    return sorted([w, int(i)] for w, i in vocab.items()
                  if int(i) < v0 and 1 <= len(w) <= WMAX and all(c in WORDCH for c in w))

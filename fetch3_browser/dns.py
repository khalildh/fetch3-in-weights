"""DNS in the net (docs/tfs_dns.md): fetch's name resolution moved into a compiled net. Fetch v1 .. v3 leave DNS to the
runtime (tfs/fetch_rt.py public_address: getaddrinfo, then "every address public"). With this tool merged, the net
examples/wire/dns_net.tern answers an allowed call's <|fetch_end|> before fetch v2 / v3 do:

    <|fetch|> https://wx.test/oslo <|fetch_end|>                         the model (v1's call)
    <|d_q4|> ID ID 01 00 00 01 00 00 00 00 00 00 02 w x 04 t e s t 00 00 01 00 01 <|d_send|>     forced: the A query
    <|d_hdr|> x11 <|d_hend|> <|d_ql|> <|d_qc|> .. <|d_qz|> <|d_qt|> x4   forced: the answer's header and question
    <|d_np0|> <|d_nq|> <|d_rr1|> <|d_f|> x8 <|d_ra|> <|d_a|> x3 <|d_ap|>  forced: answer record 1 (C0 0C, A, IN, TTL, 4)
    <|d_ip4|> 5d b8 d8 22 <|d_go|>                                      forced: the address the net chose
    <|fetch|> https://wx.test/oslo <|fetch_end|>                         forced: the call again, for fetch v2 / v3
    <|h_tls|> wx.test:443 <|h_req|> GET /oslo ...                        fetch v2 / v3, as before

The query's bytes are byte tokens <|x_HH|> (one per value 00 .. ff), built from the URL the net checked against the
allowlist (the host split into labels); the runtime (tfs/dns_rt.py) sends exactly those bytes over UDP to its one
configured resolver and pages back the datagram it receives (a tape, as fetch v2's response). The net parses the
answer as it copies it, one state token per byte: the header (ID = the query's, QR, opcode 0, TC, RCODE, QDCOUNT 1,
ANCOUNT <= RMAX), the question (byte for byte the query's), then ANCOUNT records: the owner name (labels or a
compression pointer), TYPE, CLASS, TTL, RDLENGTH, and the RDATA: A / AAAA addresses, CNAME targets (a name, pointers
followed), anything else skipped. Then it **chooses the address**: the first A (AAAA) record of the query's name
(after CNAMEs, at most CMAX) whose address is public, provided no address record in the answer is non-public (v1's
rule: every address must be public, so a mix of private and public is refused), and forces it as <|d_ip4|> / <|d_ip6|>
and its bytes. The runtime connects fetch's request to exactly that address. No address for A: the AAAA query
(<|d_q6|>) follows. Failures end with a reason and v1's <|fetch_fail|>:

    <|d_nx|>    NXDOMAIN                      <|d_priv|>  a non-public address (in the weights: the ranges below)
    <|d_bad|>   malformed / mismatched / truncated packet, bad pointer, too many records or CNAMEs
    <|d_none|>  no address (NODATA for A and AAAA, SERVFAIL, no answer)
    <|d_tc|>    TC set: over UDP the same query follows over TCP (<|d_t4|> / <|d_t6|>); over TCP a failure

Names are compared **by their labels**, case-insensitively, whatever their encoding (`name_map`, `name_match`): an
owner's bytes in place equal the chain name's at the same indices, then its zero byte, or its first pointer's target
is the chain name's byte at the same index. Sound, and complete for names in full, compressed to their first
occurrence, or compressed against the chain name's own bytes; other encodings compare unequal (fail-closed). Every
address record of the query's type is checked public, whoever's it is. With fetch v2 merged, a redirect's connect line
(at fetch v2's <|h_req|>) is resolved the same way (docs/tfs_dns.md "Redirects").

This module is the meaning (Spec, which the net is checked against, position by position), an independent reference
resolver (`ref_resolve`: real name decompression), the query encoder, the address tables, the token layout, the net's
embedding and tape, and the tool's entry for tfs/tools.py. Python because it is the spec and the glue; the net is tern.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from ._paths import ROOT

from . import circuit as CI
from . import fetch as F

I64 = np.int64
RMAX = 32                 # answer records at most
CMAX = 4                  # CNAMEs followed at most
HOPS = 4                  # compression pointers followed per name at most
UMAX = F.UMAX
WIN = (-16, 0)            # the tape's window: every entry carries r[i + o] for o in WIN[0] .. WIN[1]
NWIN = WIN[1] - WIN[0] + 1
VB = 16384
PMAX = 1024               # UDP answer bytes at most (a datagram; without EDNS a server sends <= 512)
PMAXT = 4096              # TCP answer bytes at most (the message after its 2-byte length prefix)
EOF, CUT, CFAIL, NONE = 256, 257, 258, 300
QTYPE = {"A": 1, "AAAA": 28}
NET = ROOT / "examples/wire/dns_net.tern"

# the answer's state tokens, the markers, the reasons (ids LO + 256 + k)
STATES = ["hdr", "hend", "ql", "qc", "qz", "qt", "nl0", "nl", "nc", "nz0", "nz", "np0", "np", "nq",
          "cl0", "cl", "cc", "cp0", "cp", "cq", "cz"] + [f"rr{k}" for k in range(1, RMAX + 1)] + \
         ["f", "ra", "r6"] + [f"rc{k}" for k in range(1, CMAX + 1)] + ["rx", "a", "ap", "aq", "av", "x", "ru", "au"]
MARKS = ["q4", "q6", "send", "ip4", "ip6", "go", "nx", "priv", "bad", "none", "tc", "t4", "t6"]
NAMES = [f"x_{b:02x}" for b in range(256)] + MARKS + STATES
S = len(NAMES)            # 256 + 13 + 68 = 337
MK = {n: i + 1 for i, n in enumerate(MARKS)}          # the net's MK feature
ST = {n: i + 1 for i, n in enumerate(STATES)}         # the net's ST feature
LK = ("ql", "nl0", "nl", "cl0", "cl")                 # a label's length byte
PK = ("np0", "np", "cp0", "cp")                       # a pointer's first byte
ZK = ("qz", "nz0", "nz", "cz")                        # a name's zero byte
NT = ZK + PK                                          # where a name's bytes in place end (its run's terminator)
NSTART = ("nl0", "nz0", "np0", "cl0", "cp0")          # a record's owner name / a CNAME target starts
OB = ("nl0", "nl", "nc", "nz0", "nz")                 # an owner name's bytes in place (compared with the chain's name)
QMARKS = ("q4", "q6", "t4", "t6")                     # a query: over UDP (q) or TCP (t), for A (4) or AAAA (6)
REASONS = ("nx", "priv", "bad", "none", "tc")


def fold(b):
    """A name byte as DNS compares it: ASCII letters case-insensitively, every other byte as it is."""
    return b + 32 if 65 <= b <= 90 else b

# ---- addresses: what the net refuses (fixed tables; v1's runtime uses Python's ipaddress, see docs/tfs_dns.md) ----

V4_REFUSE = [((0,), 8), ((10,), 8), ((100, 64), 10), ((127,), 8), ((169, 254), 16), ((172, 16), 12),
             ((192, 0, 0), 24), ((192, 0, 2), 24), ((192, 168), 16), ((198, 18), 15), ((198, 51, 100), 24),
             ((203, 0, 113), 24), ((224,), 3)]          # 224/3 = multicast 224/4 and reserved 240/4 (incl. broadcast)
V6_ALLOW = [((0x20,), 3)]                                # global unicast 2000::/3 only ...
V6_REFUSE = [((0x20, 0x01, 0x00), 23), ((0x20, 0x01, 0x0d, 0xb8), 32), ((0x20, 0x02), 16), ((0x3f, 0xff, 0x00), 20)]


def in_prefix(addr, pre, n):
    """addr (bytes) in the prefix pre (leading bytes) / n bits."""
    a = int.from_bytes(bytes(addr), "big")
    w = 8 * len(addr)
    p = int.from_bytes(bytes(pre) + bytes(len(addr) - len(pre)), "big")
    return (a >> (w - n)) == (p >> (w - n))


def public4(b):
    return not any(in_prefix(b, p, n) for p, n in V4_REFUSE)


def public6(b):
    return any(in_prefix(b, p, n) for p, n in V6_ALLOW) and not any(in_prefix(b, p, n) for p, n in V6_REFUSE)


def public(b):
    return public4(b) if len(b) == 4 else public6(b)


# ---- the query ----------------------------------------------------------------------------------------------------

def host_of(url):
    """The host of an allowed URL (scheme://host[:port][/path][#fragment]). Always lower case: the allowlist's origins
    are, and a URL is allowed only if its host is an origin's, character for character (fetch v2's verdict)."""
    s0 = 8 if url.startswith("https://") else 7
    rest = url[s0:]
    for i, c in enumerate(rest):
        if c in ":/#":
            return rest[:i]
    return rest


def check_name(host):
    """A host the DNS net may resolve: labels of 1..63 characters a-z 0-9 - _, not all digits (not an IP literal)."""
    labels = host.split(".")
    if not host or any(not 1 <= len(x) <= 63 for x in labels) or \
            any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in host.replace(".", "")) or \
            all(x.isdigit() for x in labels):
        raise ValueError(f"dns: not a name the net resolves: {host!r}")
    return host


def qname(host):
    out = bytearray()
    for lab in host.split("."):
        out += bytes([len(lab)]) + lab.encode()
    return bytes(out + b"\x00")


def query(host, qtype, ident):
    """The query packet the net builds: ID, RD, one question (host, qtype, IN)."""
    return bytes([ident >> 8 & 255, ident & 255, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]) + qname(host) + \
        bytes([0, qtype, 0, 1])


# ---- the layout -----------------------------------------------------------------------------------------------------

class Layout:
    """lo: the first DNS id; fetch_lo: fetch v1's block (its call, URL characters and <|fetch_fail|>); fetch2_lo:
    fetch v2's block, when merged with it: the call is then read in v2's URL characters (upper case, '#': its TEXT
    tokens) and checked by v2's verdict (the fragment dropped), so the DNS net takes over exactly the calls fetch v2 /
    v3 send; without it, v1's 46 characters and v1's verdict."""

    def __init__(self, lo, fetch_lo, fetch2_lo=None):
        from . import fetch2 as F2
        self.lo = lo
        self.f = F.Layout(fetch_lo)
        self.fetch2_lo = fetch2_lo
        self.uchars = F2.URLCH if fetch2_lo is not None else F.CHARS
        self.utok = {c: (self.f.chars[c] if c in F.CHARS else fetch2_lo + F2.EXTRA.index(ord(c)))
                     for c in self.uchars}
        self.uchar = {t: c for c, t in self.utok.items()}
        self.BYTE = [lo + b for b in range(256)]
        self.byte_of = {lo + b: b for b in range(256)}
        for i, n in enumerate(MARKS + STATES):
            setattr(self, n.upper(), lo + 256 + i)
        self.RR = [getattr(self, f"RR{k}") for k in range(1, RMAX + 1)]
        self.RC = [getattr(self, f"RC{k}") for k in range(1, CMAX + 1)]
        self.marks = {getattr(self, n.upper()): n for n in MARKS}
        self.state = {getattr(self, n.upper()): n for n in STATES}
        # fetch v2's markers a redirect's request starts with (the DNS net resolves its host, then copies them)
        self.H_TCP = self.H_TLS = self.H_REQ = None
        self.H_REDIR = ()
        if fetch2_lo is not None:
            m = {n: fetch2_lo + F2.NEXTRA + i for i, n in enumerate(F2.MARKS)}
            self.H_TCP, self.H_TLS, self.H_REQ = m["h_tcp"], m["h_tls"], m["h_req"]
            self.H_REDIR = (m["h_redir1"], m["h_redir2"], m["h_redir3"])

    def url_text(self, toks):
        if any(t not in self.uchar for t in toks):
            return None
        return "".join(self.uchar[t] for t in toks)

    def call(self, url):
        return [self.f.FETCH, *[self.utok[c] for c in url], self.f.END]

    def verdict(self, url, allow):
        """fetch v2's verdict with fetch v2 merged, else v1's: the DNS net answers exactly the calls they send."""
        from . import fetch2 as F2
        return (F2.verdict if self.fetch2_lo is not None else F.verdict)(url, allow)

    def kind(self, t):
        """The state name of a token ('nl0', 'rr', 'rc', ...), or None."""
        n = self.state.get(t)
        if n is None:
            return None
        return "rr" if n.startswith("rr") else ("rc" if n.startswith("rc") else n)

    def name(self, t):
        if self.lo <= t < self.lo + S:
            return f"<|{NAMES[t - self.lo]}|>"
        if t in self.uchar and t not in self.f.char_of:       # a v2 TEXT token in a URL (upper case, '#')
            return self.uchar[t]
        if self.fetch2_lo is not None and t in (self.H_TCP, self.H_TLS, self.H_REQ, *self.H_REDIR):
            return {self.H_TCP: "<|h_tcp|>", self.H_TLS: "<|h_tls|>", self.H_REQ: "<|h_req|>"}.get(
                t, f"<|h_redir{self.H_REDIR.index(t) + 1 if t in self.H_REDIR else 0}|>")
        return self.f.name(t)

    def tokens(self):
        return [f"<|{n}|>" for n in NAMES]

    def reserved(self):
        return range(self.lo, self.lo + S)

    def show(self, toks):
        out = []
        for t in toks:
            if t in self.byte_of:
                out.append(f"{self.byte_of[t]:02x} ")
            elif t in self.uchar:
                out.append(self.uchar[t])
            elif self.lo <= t < self.lo + S or self.f.lo <= t < self.f.lo + F.S or \
                    t in (self.H_TCP, self.H_TLS, self.H_REQ, *self.H_REDIR):
                out.append(self.name(t) + " ")
            else:
                out.append(f"<{t}> ")
        return "".join(out)


def address_of(lay, toks):
    """The address of the latest <|d_ip4|> / <|d_ip6|> block in toks: (version, bytes, position) or None."""
    k = max((i for i, t in enumerate(toks) if t in (lay.IP4, lay.IP6)), default=None)
    if k is None:
        return None
    n = 4 if toks[k] == lay.IP4 else 16
    b = toks[k + 1:k + 1 + n]
    if len(b) != n or any(t not in lay.byte_of for t in b):
        return None
    return (4 if n == 4 else 6), bytes(lay.byte_of[t] for t in b), k


def query_of(lay, toks):
    """(qtype, query bytes, the marker's position) of the query block at the end of toks (after its latest query
    marker: <|d_q4|> / <|d_q6|> over UDP, <|d_t4|> / <|d_t6|> over TCP: `is_tcp(lay, toks[k])`), or None."""
    qm = (lay.Q4, lay.Q6, lay.T4, lay.T6)
    k = max((i for i, t in enumerate(toks) if t in qm), default=None)
    if k is None or any(t not in lay.byte_of for t in toks[k + 1:]):
        return None
    return (1 if toks[k] in (lay.Q4, lay.T4) else 28), bytes(lay.byte_of[t] for t in toks[k + 1:]), k


def is_tcp(lay, t):
    return t in (lay.T4, lay.T6)


def query_of_before(lay, toks, k):
    """(qtype, query bytes, position) of the latest query block before position k (its bytes up to <|d_send|>)."""
    qm = (lay.Q4, lay.Q6, lay.T4, lay.T6)
    j = max((i for i in range(min(k, len(toks))) if toks[i] in qm), default=None)
    if j is None:
        return None
    b = []
    for t in toks[j + 1:]:
        if t not in lay.byte_of:
            break
        b.append(lay.byte_of[t])
    return (1 if toks[j] in (lay.Q4, lay.T4) else 28), bytes(b), j


def qname_host(q):
    """The host a query asks for (its QNAME as dotted text), or None."""
    i, labs = 12, []
    while i < len(q) and q[i]:
        n = q[i]
        labs.append(q[i + 1:i + 1 + n].decode(errors="replace"))
        i += 1 + n
    return ".".join(labs) if i < len(q) else None


# ---- the spec: what the net forces, position by position ---------------------------------------------------------

class Spec:
    """feed(t) -> the token the DNS net forces after t, or None. allow: the origins (v1's rule, as fetch v2's);
    respond(query bytes) -> (answer bytes, terminator EOF / CUT / CFAIL): the runtime's UDP exchange, asked at
    <|d_send|> after <|d_q4|> / <|d_q6|>; respond_tcp(query bytes): the same over TCP (after <|d_t4|> / <|d_t6|>,
    the retry of a truncated answer), `respond` if not given."""

    def __init__(self, lay, allow, respond, respond_tcp=None):
        self.lay, self.allow, self.respond = lay, list(allow), respond
        self.respond_tcp = respond_tcp
        self.toks = []
        self.mark = None                # (pos, name) of the latest DNS marker
        self.src = None                 # the call's <|fetch|> position (the latest one not forced by the copy)
        self.anyf = None                # the latest <|fetch|>
        self.q = None                   # the latest query (bytes), its qtype, its transport
        self.qtype = None
        self.tcp = False
        self.d = None
        self.r = None
        self.sent = []                  # every query sent: (bytes, tcp)
        self.last = {}                  # state name -> its latest position
        self.rr_k = {}                  # latest rr / rc index (by position)
        self.chosen = []                # (version, address bytes) of every <|d_ipN|> block
        self.rdr = False                # the source is a redirect's connect line (else a call)
        self.tls = False
        self.cmap = None                # the chain's current name (the latest CNAME target's map), None: the question's
        self.nst = []                   # positions of name starts (NSTART) after the current send
        self.nts = []                   # positions of run terminators (NT) after the current send

    # the latest position of a state kind after the current send
    def lat(self, *kinds):
        best = None
        for k in kinds:
            p = self.last.get(k)
            if p is not None and p > self.d and (best is None or p > best):
                best = p
        return best

    def url(self):
        """The source's text: the call's URL, or a redirect's connect line (HOST:PORT)."""
        return self.lay.url_text(self.toks[self.src + 1:self.call_end])

    def host(self):
        return self.url().split(":")[0] if self.rdr else host_of(self.url())

    def feed(self, t):
        lay, p = self.lay, len(self.toks)
        self.toks.append(t)
        f = lay.f
        if t == f.FETCH:
            self.anyf = p
            if not (p >= 1 and self.toks[p - 1] == lay.GO):
                self.src, self.rdr = p, False
        if t in (lay.H_TCP, lay.H_TLS) and t is not None:   # fetch v2's request starts (after a call, a redirect)
            self.anyf = p
            if p >= 1 and self.toks[p - 1] in lay.H_REDIR:    # a redirect's: its connect line is a source
                self.src, self.rdr, self.tls = p, True, t == lay.H_TLS
        if t == lay.H_REQ and t is not None:
            return self._verdict_rdr(p)
        if t in lay.marks:
            self.mark = (p, lay.marks[t])
        k = lay.kind(t)
        if k is not None:
            self.last[k] = p
            if k in ("rr", "rc"):
                self.rr_k[k] = (lay.RR if k == "rr" else lay.RC).index(t) + 1
            if k in NSTART:
                self.nst.append(p)
            if k in NT:
                self.nts.append(p)
            if k in ("cz", "cq"):          # a CNAME target ends: the chain's name is now its map
                self.cmap = self.name_map(p)
        if t == f.END:
            return self._verdict(p)
        if self.mark is None:
            return None
        mp, mk = self.mark
        off = p - mp
        if mk in QMARKS:
            if off == 0:
                self.qtype = 1 if mk in ("q4", "t4") else 28
                self.tcp = mk in ("t4", "t6")
                self.q = query(self.host(), self.qtype, mp)
            if off < len(self.q):
                return lay.BYTE[self.q[off]]
            return lay.SEND if off == len(self.q) else None
        if mk == "send":
            if off == 0:
                fn = self.respond_tcp if (self.tcp and self.respond_tcp is not None) else self.respond
                body, term = fn(self.q)
                self.sent.append((self.q, self.tcp))
                self.d = p
                self.r = list(body)[:PMAXT if self.tcp else PMAX] + [term]
                self.last, self.rr_k = {}, {}
                self.cmap, self.nst, self.nts = None, [], []
            return self._answer(p)
        if mk in ("ip4", "ip6"):
            n = 4 if mk == "ip4" else 16
            if off == 0:
                a = self.lat("ap")
                i = a - self.d - 1
                self.addr = bytes(self.r[i - n + 1:i + 1])
                self.chosen.append((4 if n == 4 else 6, self.addr))
            if off < n:
                return lay.BYTE[self.addr[off]]
            return lay.GO if off == n else None
        if mk == "go":
            u = self.url()
            if off == 0:
                return (lay.H_TLS if self.tls else lay.H_TCP) if self.rdr else f.FETCH
            if u is None:                    # a later call that is no URL (not the one copied): nothing to copy
                return None
            if off <= len(u):
                return lay.utok[u[off - 1]]
            if off == len(u) + 1:
                return lay.H_REQ if self.rdr else f.END
            return None
        if mk in REASONS:
            if off != 0:
                return None
            if mk == "tc" and not self.tcp:     # a truncated UDP answer: the same query again, over TCP
                return lay.T4 if self.qtype == 1 else lay.T6
            return f.FAIL
        return None

    def _verdict(self, p):
        """At a <|fetch_end|>: <|d_q4|> for an allowed call the model (or a follow) wrote, else silence."""
        if self.src is None or self.anyf != self.src or self.rdr:
            return None
        self.call_end = p
        u = self.url()
        if self.lay.verdict(u, self.allow) != "ok":
            return None
        return self.lay.Q4

    def _verdict_rdr(self, p):
        """At fetch v2's <|h_req|> after a redirect's connect line (not this net's copy): <|d_q4|> if it is the connect
        line of an allowed origin (fetch v2 writes it from a Location it checked), else silence."""
        if self.src is None or self.anyf != self.src or not self.rdr:
            return None
        u = self.lay.url_text(self.toks[self.src + 1:p])
        from . import fetch2 as F2
        ok = any(u == F2.connect_line(o) and self.tls == o.startswith("https://") for o in self.allow)
        if not ok:
            return None
        self.call_end = p
        return self.lay.Q4

    def tape(self, i):
        return self.r[i] if 0 <= i < len(self.r) else NONE

    def pv(self, q):
        """The pointer value at the pointer's first byte's position q."""
        i = q - self.d - 1
        return 256 * (self.tape(i) - 192) + self.tape(i + 1)

    # ---- names: runs, maps, comparison -------------------------------------------------------------------------
    # A name is read as runs of bytes in place: from its start to the first zero byte or pointer (the run's
    # terminator), then, after a pointer, the run at its target, and so on. Its map: the runs it is made of, as
    # (offset, length) segments (a zero byte counts, a pointer's two bytes do not). At most HOPS pointers followed,
    # each strictly backward, each landing on a label's length byte or a pointer that the net parsed as part of a
    # name (in the question, an owner, a CNAME target).

    def rs(self, o):
        """The run start of the name byte at offset o: the latest name start at or before it (else the question's)."""
        q = self.d + 1 + o
        s = max((j for j in self.nst if j <= q), default=None)
        return 12 if s is None else s - self.d - 1

    def kind_at(self, o):
        q = self.d + 1 + o
        if o < 0 or q >= len(self.toks):
            return None
        return self.lay.kind(self.toks[q])

    def name_map(self, p):
        """The map of the latest name (its start: the latest name start, else the question; its first run's
        terminator: the latest terminator) at position p, or None (a pointer the net refuses)."""
        d = self.d
        s0 = max((j for j in self.nst if j <= p), default=None)
        e0 = max((j for j in self.nts if j <= p), default=None)
        if e0 is None:
            return None
        start = 12 if s0 is None else s0 - d - 1
        e = e0 - d - 1
        segs, hops = [], 0
        while True:
            if self.kind_at(e) in ZK:
                segs.append((start, e - start + 1))
                return segs
            segs.append((start, e - start))
            if hops == HOPS:
                self.why = "more than HOPS pointers"
                return None
            T = self.pv(d + 1 + e)
            if not T < e:
                self.why = "a pointer not strictly backward"
                return None
            if self.kind_at(T) not in LK + PK:
                self.why = "a pointer into " + ("the header" if T < 12 else f"a {self.kind_at(T)} byte")
                return None
            hops += 1
            key = self.rs(T)
            e2 = max((j for j in self.nts if j <= p and self.rs(j - d - 1) == key), default=None)
            if e2 is None or e2 - d - 1 < T:
                self.why = "no run end"
                return None
            start, e = T, e2 - d - 1

    @staticmethod
    def map_pos(m, j):
        """The offset of the j-th byte of a name with map m (None past its end)."""
        c = 0
        for s, n in m:
            if c <= j < c + n:
                return s + j - c
            c += n
        return None

    def chain_map(self):
        return self.cmap if self.cmap is not None else [(12, len(self.q) - 16)]

    def name_match(self, m):
        """The latest owner name (map m) is the chain's current name: its bytes in place equal the chain name's
        (case-insensitively); after them its zero byte, or its first pointer's target (the first non-empty run's
        offset) is the chain name's byte at the same index. Sound; complete for names written in full, for names
        compressed to their first occurrence, and for names compressed against the chain name's own bytes."""
        cm = self.chain_map()
        s, n0 = m[0]
        for j in range(n0):
            o = self.map_pos(cm, j)
            if o is None or fold(self.tape(s + j)) != fold(self.tape(o)):
                return False
        if len(m) == 1:
            return True                            # in place to its zero byte (compared above)
        tstar = next(st for st, n in m[1:] if n > 0)
        return tstar == self.map_pos(cm, n0)

    def _answer(self, p):
        lay, d = self.lay, self.d
        i = p - d
        x = self.tape(i)
        t = self.toks[p]
        k = lay.kind(t)
        L = len(self.q)
        bad = lay.BAD
        if x >= 256 and self.reads(p, k):
            return lay.NONE if x == CFAIL else bad        # a terminator where a byte is read
        if 12 <= i < L and x != self.q[i]:
            return bad                                    # the question is not the query's

        def label(fam, at_start, room=True):
            """A name's next label byte: 0 the end, 1..63 a label, >= 192 a pointer (not in the question)."""
            if x == 0:
                return {"q": lay.QZ, "n": lay.NZ0 if at_start else lay.NZ, "c": bad}[fam]
            if not room:
                return bad
            if 1 <= x <= 63:
                return getattr(lay, fam.upper() + ("L0" if at_start and fam != "q" else "L"))
            if x >= 192 and fam != "q":
                return getattr(lay, fam.upper() + ("P0" if at_start else "P"))
            return bad

        def cont():
            if self.nrec() == self.tape(7):
                return self.decide()
            return label("n", True)

        if t == lay.SEND:
            return lay.HDR
        if k == "hdr":
            if i < 11:
                return lay.HDR
            r = self.r
            if not (r[0] == self.q[0] and r[1] == self.q[1] and 128 <= r[2] <= 135):
                return bad
            if r[2] in (130, 131, 134, 135):
                return lay.TC
            rc = r[3] % 16
            if rc == 3:
                return lay.NX
            if rc != 0:
                return lay.NONE
            if not (r[4] == 0 and r[5] == 1 and r[6] == 0 and r[7] <= RMAX):
                return bad
            return lay.HEND
        if k == "hend":
            return label("q", True)
        if k in ("ql", "qc", "nl0", "nl", "nc", "cl0", "cl", "cc"):
            kl = self.lat(*LK)
            inlab = p + 1 <= kl + self.tape(kl - d - 1)
            fam = k[0]
            if fam == "c":
                last = p + 1 == self.rd_end()          # the rdata's last byte: it must end the name
                if inlab:
                    return bad if last else lay.CC
                if x == 0:
                    return lay.CZ if last else bad
                return label("c", False, room=not last)
            if inlab:
                return getattr(lay, fam.upper() + "C")
            return label(fam, False)
        if k == "qz":
            return lay.QT
        if k == "qt":
            return lay.QT if p - self.lat("qz") < 4 else cont()
        if k in ("np0", "np"):
            return lay.NQ
        if k in ("cp0", "cp"):
            if p + 1 != self.rd_end():
                return bad
            if self.name_map(p) is None:
                return bad
            return lay.CQ
        if k in ("nz0", "nz", "nq"):
            return lay.RR[self.nrec()]
        if k in ("rr", "f"):
            if p - self.lat("rr") < 8:
                return lay.F
            return self.rdlen_decision(p, i, x)
        if k in ("ra", "r6", "ru", "a"):
            rp = self.lat("ra", "r6", "ru")
            n = 4 if self.qtype == 1 else 16
            if p - rp < n - 1:
                return lay.A
            a = bytes(self.r[i - n + 1:i + 1])
            if not public(a):
                return lay.AV
            if lay.kind(self.toks[rp]) == "ru":
                return lay.AU
            return lay.AQ if self.lat("ap") is not None else lay.AP
        if k in ("rx", "x"):
            return lay.X if p < self.rd_end() else cont()
        if k == "rc":
            return label("c", True, room=p + 1 < self.rd_end())
        if k in ("ap", "aq", "av", "au", "cz", "cq"):
            return cont()
        return None

    def nrec(self):
        """The records read so far in this answer (the latest <|d_rrK|>'s K)."""
        return self.rr_k.get("rr", 0) if self.lat("rr") is not None else 0

    def reads(self, p, k):
        """Whether the token after position p (state k) stands for a byte (else it is a decision: the answer is
        over, or the next record starts past the last one)."""
        if k in ("ap", "aq", "av", "au", "cz", "cq"):
            return self.nrec() != self.tape(7)
        if k in ("rx", "x"):
            return p < self.rd_end() or self.nrec() != self.tape(7)
        if k == "qt":
            return p - self.lat("qz") < 4 or self.nrec() != self.tape(7)
        return k is not None or self.toks[p] == self.lay.SEND

    def rd_end(self):
        """The position of the latest record's last RDATA byte's token."""
        return self.lat("ra", "r6", "rc", "rx", "ru") + self.rdlen()

    def rdlen(self):
        """RDLENGTH of the latest record (read at its RDLENGTH token)."""
        rp = self.lat("ra", "r6", "rc", "rx", "ru")
        i = rp - self.d - 1
        return 256 * self.tape(i - 1) + self.tape(i)

    def rdlen_decision(self, p, i, x):
        lay, r = self.lay, self.r
        typ = 256 * r[i - 9] + r[i - 8]
        cls = 256 * r[i - 7] + r[i - 6]
        rdlen = 256 * r[i - 1] + x
        m = self.name_map(p)
        if m is None:
            return lay.BAD
        match = self.name_match(m)
        if match and cls == 1 and typ == 1 and self.qtype == 1:
            return lay.RA if rdlen == 4 else lay.BAD
        if match and cls == 1 and typ == 28 and self.qtype == 28:
            return lay.R6 if rdlen == 16 else lay.BAD
        if match and cls == 1 and typ == 5:
            kc = (self.rr_k.get("rc", 0) if self.lat("rc") is not None else 0) + 1
            if kc > CMAX or rdlen == 0:
                return lay.BAD
            return lay.RC[kc - 1]
        if cls == 1 and typ == self.qtype and rdlen == (4 if typ == 1 else 16):
            return lay.RU                 # an address of another name: checked too (a non-public one refuses all)
        return lay.RX

    def decide(self):
        lay = self.lay
        if self.lat("av") is not None:
            return lay.PRIV
        if self.lat("ap") is not None:
            return lay.IP4 if self.qtype == 1 else lay.IP6
        return lay.Q6 if self.qtype == 1 else lay.NONE


def forced(lay, allow, respond, toks, respond_tcp=None):
    sp = Spec(lay, allow, respond, respond_tcp)
    return [sp.feed(t) for t in toks]


# ---- the independent reference: a resolver with real name decompression ------------------------------------------

class Bad(Exception):
    pass


def read_name(pkt, i, end=None):
    """(labels (lower case), the offset after the name in place). Pointers must point strictly backward."""
    labs, j, after, seen = [], i, None, 0
    while True:
        if j >= len(pkt) or (end is not None and after is None and j >= end):
            raise Bad("name runs past its end")
        n = pkt[j]
        if n == 0:
            if after is None:
                after = j + 1
            return labs, after
        if n >= 192:
            if j + 1 >= len(pkt) or (end is not None and after is None and j + 1 >= end):
                raise Bad("pointer cut")
            t = 256 * (n - 192) + pkt[j + 1]
            if t >= j:
                raise Bad("forward pointer")
            if after is None:
                after = j + 2
            seen += 1
            if seen > 64:
                raise Bad("pointer loop")
            j = t
            continue
        if n > 63:
            raise Bad("label type")
        if j + 1 + n > len(pkt):
            raise Bad("label cut")
        labs.append(pkt[j + 1:j + 1 + n].lower())
        j += 1 + n


def ref_resolve(pkt, q, qtype):
    """The outcome of one answer, as a resolver with full decompression sees it: ('ip', bytes) | ('nx',) |
    ('priv', bytes) | ('none',) | ('bad', why) | ('tc',) | ('next',) (no address for A: ask AAAA). The same bounds as the
    net (RMAX records, CMAX CNAMEs), names compared as names (case-insensitive), not by offset."""
    try:
        if len(pkt) < 12:
            raise Bad("short")
        if pkt[0:2] != q[0:2] or not 128 <= pkt[2] <= 135:
            raise Bad("id / qr / opcode")
        if pkt[2] & 2:
            return ("tc",)
        rc = pkt[3] & 15
        if rc == 3:
            return ("nx",)
        if rc:
            return ("none",)
        if pkt[4:6] != b"\x00\x01" or pkt[6] or pkt[7] > RMAX:
            raise Bad("counts")
        L = len(q)
        if pkt[12:L] != q[12:L]:
            raise Bad("question")
        cur, _ = read_name(q, 12)
        i, chain, addrs = L, 0, []
        for _ in range(pkt[7]):
            owner, i = read_name(pkt, i)
            if i + 10 > len(pkt):
                raise Bad("fixed fields cut")
            typ, cls, rdl = 256 * pkt[i] + pkt[i + 1], 256 * pkt[i + 2] + pkt[i + 3], 256 * pkt[i + 8] + pkt[i + 9]
            i += 10
            if i + rdl > len(pkt):
                raise Bad("rdata cut")
            rd = pkt[i:i + rdl]
            match = owner == cur and cls == 1
            if match and typ == qtype and typ in (1, 28):
                if rdl != (4 if typ == 1 else 16):
                    raise Bad("address length")
                addrs.append(bytes(rd))
            elif match and typ == 5:
                chain += 1
                if chain > CMAX or rdl == 0:
                    raise Bad("cname")
                tgt, e = read_name(pkt, i, i + rdl)
                if e != i + rdl:
                    raise Bad("cname length")
                cur = tgt
            i += rdl
        bad = [a for a in addrs if not public(a)]
        if bad:
            return ("priv", bad[0])
        if addrs:
            return ("ip", addrs[0])
        return ("next",) if qtype == 1 else ("none",)
    except Bad as e:
        return ("bad", str(e))
    except IndexError:
        return ("bad", "short")


def outcome(lay, forced_after):
    """The outcome the Spec / net forced after an answer: the token that ends the answer's parse."""
    t = forced_after
    return {lay.IP4: "ip", lay.IP6: "ip", lay.Q6: "next", lay.NX: "nx", lay.PRIV: "priv", lay.BAD: "bad",
            lay.NONE: "none", lay.TC: "tc"}.get(t)


# ---- the net's embedding, the tape, the net, the tool --------------------------------------------------------------

def features(lay):
    """The token embedding for vocabulary V (dns_net.tern's feature names)."""
    def table(fnames, V):
        Ft = np.zeros((V, len(fnames)), I64)
        col = {n: i for i, n in enumerate(fnames)}
        ids = np.arange(V)
        for n, i in col.items():
            if n.startswith("code"):
                Ft[:, i] = 2 * ((ids >> int(n[4:])) & 1) - 1
        f = lay.f
        Ft[f.FETCH, col["F_FETCH"]] = 1
        Ft[f.END, col["F_END"]] = 1
        for i, c in enumerate(lay.uchars):
            Ft[lay.utok[c], col["ISUC"]] = 1
            Ft[lay.utok[c], col["UCI"]] = i
            Ft[lay.utok[c], col["UCB"]] = ord(c)
        for b in range(256):
            Ft[lay.BYTE[b], col["XB"]] = b
        for t, n in lay.marks.items():
            Ft[t, col["MARK"]] = 1
            Ft[t, col["MK"]] = MK[n]
        Ft[lay.GO, col["F_GO"]] = 1
        Ft[lay.SEND, col["F_SEND"]] = 1
        Ft[[lay.Q4, lay.Q6, lay.T4, lay.T6], col["F_Q"]] = 1
        Ft[[lay.Q6, lay.T6], col["Q6"]] = 1
        Ft[[lay.T4, lay.T6], col["TCPQ"]] = 1
        if lay.fetch2_lo is not None:                  # fetch v2's request markers (a redirect's source)
            Ft[[lay.H_TCP, lay.H_TLS], col["F_SCH"]] = 1
            Ft[lay.H_TLS, col["F_TLS"]] = 1
            Ft[lay.H_REQ, col["F_REQ"]] = 1
            Ft[list(lay.H_REDIR), col["F_RDR"]] = 1
        for t, n in lay.state.items():
            k = lay.kind(t)
            Ft[t, col["ST"]] = ST[n] if k not in ("rr", "rc") else ST[n[:2] + "1"]
            Ft[t, col["ISST"]] = 1
            for key, kinds in (("LK", LK), ("PK", PK), ("NST", NSTART), ("NT", NT), ("ZK", ZK), ("OB", OB)):
                if k in kinds:
                    Ft[t, col[key]] = 1
            if k in ("cz", "cq"):
                Ft[t, col["CEND"]] = 1
            if k == "rr":
                Ft[t, col["RR"]] = 1
                Ft[t, col["RRI"]] = lay.RR.index(t) + 1
            if k == "rc":
                Ft[t, col["RC"]] = 1
                Ft[t, col["RCI"]] = lay.RC.index(t) + 1
            if k in ("ra", "r6", "rc", "rx", "ru"):
                Ft[t, col["RD"]] = 1
            if k == "r6":
                Ft[t, col["RD6"]] = 1
            if k in ("ra", "r6"):
                Ft[t, col["RDA"]] = 1
            for key in ("ap", "av", "hend", "qz"):
                if k == key:
                    Ft[t, col["F_" + key.upper()]] = 1
        return Ft
    return table


def tape_entries(r):
    """The copy head's virtual entries for a tape r (bytes + terminator): VB + i for i = 0 .. n + 1, each with its
    window r[i + o], o in WIN (NONE before the tape, the terminator repeated past its end)."""
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


def check_allow(allow):
    allow = F.parse_allow(allow)
    for o in allow:
        check_name(host_of(o))
    return allow


def export(out_dir, lay, V, allow):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    allow = check_allow(allow)
    lo2 = -1 if lay.fetch2_lo is None else lay.fetch2_lo
    r = subprocess.run([sys.executable, "-m", "tern", str(NET), str(lay.lo), str(lay.f.lo), str(V),
                        ",".join(allow) or "-", str(lo2), str(out_dir)], capture_output=True, text=True, cwd=ROOT,
                       timeout=3600)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-4000:] + r.stdout[-2000:])
    (out_dir / "allow.txt").write_text("".join(o + "\n" for o in allow))
    (out_dir / "layout.json").write_text(json.dumps([lay.lo, lay.f.lo, V] + ([lay.fetch2_lo] if lo2 >= 0 else []))
                                         + "\n")
    return r.stdout


class DnsNet(CI.Circuit):
    """The exported net with the runtime's tapes (fetch v2's mechanism): each answer's tape in the "copy" head's
    virtual entries while it is current (`page(dec, r)` replaces it), and forward(toks, tapes=[(d, r)]) with every
    tape present, each masked to the positions whose latest <|d_send|> is its own."""

    def __init__(self, d, lay, V):
        super().__init__(d, features(lay), V)
        self.lay = lay
        p = Path(d) / "allow.txt"
        self.allow = p.read_text().split() if p.exists() else []
        self.win = [self.fnames.index(f"W{o - WIN[0]}") for o in range(WIN[0], WIN[1] + 1)]
        self.wt = self.fnames.index("TAPE")

    virtual_kv = None            # set below (fetch v2's, on this net's window columns)
    page = None
    forward = None


def _bind():
    from . import fetch2 as F2
    DnsNet.virtual_kv = F2.Fetch2Net.virtual_kv

    def page(self, dec, r):
        dec.virtual = []
        dec.add_virtual(tape_entries(r))

    def forward(self, toks, tapes=None):
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

    DnsNet.page = page
    DnsNet.forward = forward


_bind()


class DnsTool:
    """tfs/tools.py's entry. Config: allow (the origins compiled into the net: fetch's), fetch_lo (v1's block: the
    call, the URL characters, <|fetch_fail|>). Priority 4.4: before fetch v3 (4.5), v2 (5) and v1 (6), so the DNS net's
    <|d_q4|> wins at an allowed call's <|fetch_end|>; at its own copy of the call it is silent and they answer."""
    kind, priority, dirname = "dns", 4.4, "dns"
    S = S

    @staticmethod
    def size(**_):
        return S

    def __init__(self, lo, V, allow=None, fetch_lo=None, fetch2_lo=None, **_):
        if fetch_lo is None:
            raise ValueError("dns extends the fetch tool: merge with fetch (and fetch2 / fetch3) first")
        self.lo, self.V = lo, V
        self.lay = Layout(lo, fetch_lo, fetch2_lo)
        self.allow = check_allow(allow)
        self.net = None
        self.respond = lambda q: ([], CFAIL)

    def config(self):
        c = {"allow": self.allow, "fetch_lo": self.lay.f.lo}
        if self.lay.fetch2_lo is not None:
            c["fetch2_lo"] = self.lay.fetch2_lo
        return c

    def tokens(self):
        return self.lay.tokens()

    def reserved(self):
        return self.lay.reserved()

    def export(self, d):
        return export(d, self.lay, self.V, self.allow)

    def load(self, d):
        self.net = DnsNet(d, self.lay, self.V)
        if self.net.allow != self.allow:
            raise ValueError(f"{d}: the net allows {self.net.allow}, tools.json {self.allow}")
        return self.net

    def spec(self):
        return Spec(self.lay, self.allow, self.respond)

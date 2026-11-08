"""Small fetch3 circuit demonstration. Python provides the reference engine and byte transport."""

import argparse
import hashlib
import ipaddress
import json
import socket
import ssl
import time
from pathlib import Path

from . import fetch2 as F2
from . import fetch3 as F3
from .dns_runtime import Resolver

WEIGHTS = Path(__file__).parent / "weights"
FIXTURE = (b"<!doctype html><html><head><title>Fetch3</title>"
           b"<style>.hidden { display:none }</style></head><body>"
           b"<h1>Hello from a browser made of weights.</h1>"
           b"<p>This page was read by a compiled integer circuit.</p>"
           b"<!-- do not show this comment -->"
           b"<script>do_not_show_this_script()</script>"
           b'<a href="/next">A numbered link</a></body></html>')


def fixture_response():
    return (b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: "
            + str(len(FIXTURE)).encode() + b"\r\nConnection: close\r\n\r\n" + FIXTURE)


def load_model():
    for name, expected in json.loads((WEIGHTS / "SHA256.json").read_text()).items():
        if hashlib.sha256((WEIGHTS / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"circuit artifact checksum mismatch: {name}")
    lo3, lo2, lo1, vocab, page, part, *_ = json.loads((WEIGHTS / "layout.json").read_text())
    words = [(w, int(t)) for w, t in (x.split(":") for x in (WEIGHTS / "words.txt").read_text().strip().split(","))]
    lay = F3.Layout(lo3, lo2, lo1, words, (WEIGHTS / "agent.txt").read_text().strip())
    return F3.Fetch3Net(WEIGHTS, lay, vocab), lay, page


def request_block(lay, toks):
    """Read the transport descriptor emitted by the circuit; do not construct HTTP here."""
    k = max((i for i, t in enumerate(toks) if t in (lay.TCP, lay.TLS)), default=None)
    if k is None:
        raise ValueError("circuit did not produce a transport marker")
    tail = toks[k + 1:]
    ri = tail.index(lay.REQ)
    target = bytes(lay.byte_of[t] for t in tail[:ri]).decode("ascii")
    request = bytes(lay.byte_of[t] for t in tail[ri + 1:])
    return ("https" if toks[k] == lay.TLS else "http"), target, request


class Transport:
    def __init__(self, allow, live=False, replay=None, timeout=15.0, max_bytes=8192,
                 resolver=None, dns_scenario="normal"):
        self.allow, self.live, self.timeout, self.max_bytes = allow, live, timeout, max_bytes
        self.replay = list(replay) if replay is not None else None
        self.receipts, self.records = [], []
        self.connections = 0
        self.resolver, self.dns_scenario = resolver, dns_scenario

    def exchange(self, scheme, target, request):
        host, sep, port_text = target.rpartition(":")
        if not sep or not host or not port_text.isdigit():
            raise ValueError("invalid circuit transport target")
        port = int(port_text)
        origin = f"{scheme}://{host}" + ("" if port == (443 if scheme == "https" else 80) else f":{port}")
        if origin not in self.allow and f"{scheme}://{target}" not in self.allow:
            raise ValueError("target outside the circuit's compiled allowlist")
        digest = hashlib.sha256(request).hexdigest()
        saved = None
        if self.replay is not None:
            if not self.replay:
                raise ValueError("replay exhausted")
            saved = self.replay.pop(0)
            if (saved["scheme"], saved["target"], saved["request_hex"]) != (scheme, target, request.hex()):
                raise ValueError("replay request differs from recorded circuit output")
            if "dns" not in saved:
                raise ValueError("recording predates compiled DNS; make a new recording with this version")
        dns = Resolver(live=self.live, resolver=self.resolver, scenario=self.dns_scenario,
                       replay=list(saved["dns"]["exchanges"]) if saved is not None else None)
        dns_proof, dns_records = dns.resolve(origin)
        selected = ipaddress.ip_address(dns_proof["selected_address"])
        if not selected.is_global or selected.is_multicast:
            raise ValueError("refusing a non-public circuit-selected address")
        if saved is not None:
            if dns_proof["selected_address"] != saved["dns"]["proof"]["selected_address"]:
                raise ValueError("DNS replay selected a different address")
            if dns_proof['token_sha256'] != saved['dns']['proof']['token_sha256']:
                raise ValueError('DNS replay token hash differs from recording')
            response, term = bytes.fromhex(saved["response_hex"]), saved["term"]
            receipt = dict(mode="replay", target=target, request_sha256=digest, response_bytes=len(response),
                           network_connections=0)
        elif not self.live:
            response, term = fixture_response(), F2.EOF
            receipt = dict(mode="fixture", target=target, request_sha256=digest, response_bytes=len(response),
                           network_connections=0)
        else:
            deadline = time.monotonic() + self.timeout
            # No host-name resolver here. Connect to the numeric circuit output,
            # while SNI and certificate verification still use the original host.
            family = socket.AF_INET6 if selected.version == 6 else socket.AF_INET
            sockaddr = (str(selected), port)
            sock = socket.socket(family, socket.SOCK_STREAM)
            try:
                sock.settimeout(max(0.1, deadline - time.monotonic()))
                sock.connect(sockaddr)
                self.connections += 1
                peer_address = str(ipaddress.ip_address(sock.getpeername()[0]))
                if ipaddress.ip_address(peer_address) != selected:
                    raise ValueError("connected peer differs from the circuit-selected address")
                if scheme == "https":
                    sock = ssl.create_default_context().wrap_socket(sock, server_hostname=host)
                tls_version = sock.version() if scheme == "https" else None
                # These are the circuit's exact request bytes: no HTTP client reconstructs them.
                sock.sendall(request)
                raw = bytearray()
                term = F2.EOF
                while True:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        term = F2.CUT
                        break
                    sock.settimeout(left)
                    chunk = sock.recv(min(4096, self.max_bytes + 1 - len(raw)))
                    if not chunk:
                        break
                    raw.extend(chunk)
                    if len(raw) > self.max_bytes:
                        raw = raw[:self.max_bytes]
                        term = F2.CUT
                        break
                response = bytes(raw)
            finally:
                sock.close()
            receipt = dict(mode="live", target=target, request_sha256=digest, sent_bytes=len(request),
                           response_bytes=len(response), network_connections=1, tls=tls_version,
                           certificate_verified=scheme == "https", sent_exact_circuit_bytes=True,
                           connected_address=str(selected),peer_address=peer_address,tls_server_hostname=host)
        receipt.update(dns=dns_proof,runtime_target_getaddrinfo_used=False)
        self.receipts.append(receipt)
        self.records.append(dict(scheme=scheme, target=target, request_hex=request.hex(),
                                 response_hex=response.hex(), term=int(term),
                                 dns=dict(proof=dns_proof,exchanges=dns_records)))
        return response, int(term)


class Session:
    def __init__(self, net, lay, page, transport):
        self.net, self.lay, self.transport = net, lay, transport
        self.dec = net.decoder()
        self.tokens, self.forced, self.answers = [], [], []
        self.pending_responses = []
        self.spec = F3.Spec(lay, net.allow, lambda *args: self.pending_responses.pop(0), page)

    def step(self, token):
        if len(self.tokens) >= self.net.TREAL:
            raise ValueError(f"circuit context limit reached ({self.net.TREAL} positions)")
        code, fire, notfire = self.dec.step(token)
        next_token = self.net.decode_code(code) if fire == 1 else None
        if fire not in (0, 1) or (fire == 0 and (code.any() or notfire != self.net.nb)):
            raise AssertionError("invalid circuit silence/fire output")
        self.tokens.append(int(token))
        self.forced.append(next_token)
        expected = self.spec.feed(token)
        if next_token != expected:
            raise AssertionError(f"circuit/spec mismatch at position {len(self.tokens)-1}: {next_token} != {expected}")
        if next_token == self.lay.SEND:
            scheme, target, request = request_block(self.lay, self.tokens)
            response, term = self.transport.exchange(scheme, target, request)
            self.net.page(self.dec, list(response) + [term])
            self.pending_responses.append((list(response), term))
        return next_token

    def emit(self, token):
        following = self.step(token)
        while following is not None:
            following = self.step(following)

    def read(self, url, max_pages=4, follow=None):
        for token in [0, 7, *self.lay.call(url)]:
            self.emit(token)
        self.finish_pages(max_pages)
        if follow is not None:
            if not 1 <= follow <= 32:
                raise ValueError("--follow must be 1..32")
            self.emit(self.lay.FOLLOW)
            self.emit(self.lay.LN[follow - 1])
            self.finish_pages(max_pages)
        start = max((i for i, t in enumerate(self.tokens) if t == self.lay.HEND), default=-1)
        return self.lay.page_text(self.tokens[start + 1:]) if start >= 0 else ""

    def finish_pages(self, maximum):
        count = 1
        while (len(self.tokens) >= 5 and self.tokens[-1] == self.lay.f.CUT
               and self.tokens[-5] == self.lay.MORE and count < maximum):
            self.emit(self.lay.NEXT)
            count += 1


def run(url="https://example.com", live=False, replay=None, pages=4, follow=None,
        resolver=None,dns_scenario="normal"):
    net, lay, page = load_model()
    transport = Transport(net.allow, live=live, replay=replay,resolver=resolver,dns_scenario=dns_scenario)
    session = Session(net, lay, page, transport)
    started = time.monotonic()
    text = session.read(url, max_pages=pages, follow=follow)
    if transport.replay:
        raise ValueError("replay has unused exchanges")
    proof = dict(backend="python-numpy-int64-reference", compiled_not_trained=True, url=url,
                 positions=len(session.tokens), circuit_matches_spec=True,
                 ending=lay.name(session.tokens[-1]),
                 network_connections=transport.connections, seconds=round(time.monotonic()-started, 2),
                 dns_in_weights=True,
                 dns_network_exchanges=sum(r['dns']['network_exchanges'] for r in transport.receipts),
                 wires=net.n, operations=len(net.op_w), attention_heads=len(net.heads), levels=net.depth,
                 receipts=transport.receipts,
                 token_sha256=hashlib.sha256(json.dumps(session.tokens,separators=(",", ":")).encode()).hexdigest())
    return text, proof, transport.records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="https://example.com")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true", help="make a real socket/TLS request (default: offline fixture)")
    mode.add_argument("--replay", type=Path, help="replay a recorded run with zero network access")
    parser.add_argument("--record", type=Path, help="save raw response bytes for replay; keep this file private")
    parser.add_argument("--pages", type=int, default=4, choices=range(1, 25))
    parser.add_argument("--follow", type=int, help="follow a circuit-numbered link, 1..32")
    parser.add_argument("--resolver", help="numeric DNS resolver IP (default: /etc/resolv.conf)")
    args = parser.parse_args()
    replay = json.loads(args.replay.read_text()) if args.replay else None
    if replay and replay["url"] != args.url:
        parser.error("--url must match the recorded URL")
    try:
        text, proof, records = run(args.url, args.live, replay["exchanges"] if replay else None,
                                   args.pages, args.follow,resolver=args.resolver)
    except (OSError, ValueError, AssertionError) as error:
        parser.exit(1, f"fetch3: {error}\n")
    print("Fetch3: browser logic compiled into circuit weights\nRunner: Python / NumPy, exact integer reference\n")
    if replay and proof['token_sha256'] != replay['proof']['token_sha256']:
        parser.exit(1,'fetch3: replay token hash differs from recording\n')
    print(text.strip())
    print("\nExecution proof:\n" + json.dumps(proof, indent=2))
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps(dict(url=args.url, exchanges=records, proof=proof), indent=2) + "\n")


if __name__ == "__main__":
    main()

# Fetch3: a text-mode browser compiled into weights

DNS resolution, HTTP request construction, and HTML parsing, represented as integer circuits with sparse ReGLU layers and attention heads. The DNS circuit emits queries, parses responses, follows CNAMEs, checks addresses, and chooses an IP. Fetch3 emits HTTP request bytes, reads the response, skips markup/scripts/comments, and produces text and numbered links.

**v0.2 includes compiled DNS and a tested DNS-to-HTTPS handoff.** The HTTP socket connects to the IP selected by the DNS circuit, with TLS certificate verification and SNI still using the original hostname. The old host-resolver path has been removed.

**This is an executable Python/NumPy reference demo of a compiled circuit. It is not an LLM, a trained browser agent, or a GPU demo.** No training, model API, or API key is required. The caller supplies the URL and navigation actions.

## Run it

Python 3.11+ and [uv](https://docs.astral.sh/uv/) are sufficient:

```sh
uv run python -m fetch3_browser
```

The default uses bundled DNS and HTML fixtures and makes **zero network connections**. Both circuits still execute. It prints:

```text
Fetch3

# Hello from a browser made of weights.
This page was read by a compiled integer circuit.
[1]A numbered link
```

Then try a real HTTPS request:

```sh
uv run python -m fetch3_browser --live --url https://example.com
```

The execution receipt records the socket connection, TLS version, certificate verification, the exact request-byte hash, response size, and agreement with the independent reference specification at every token position. On a Mac, allow roughly 10–30 seconds for a short page; this is a correctness reference, not a speed demonstration. Timing varies by CPU and response size.

DNS packets go to the numeric resolver configured in `/etc/resolv.conf`. If your network blocks that resolver, choose one explicitly with `--resolver IP`. There is no silent fallback to a host-name resolver. The receipt includes the circuit-selected IP and actual connected peer so the handoff can be checked.

Without uv:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install .
fetch3-demo --live
```

## Record and replay

```sh
uv run python -m fetch3_browser --live --record runs/example.json
uv run python -m fetch3_browser --replay runs/example.json
```

Replay checks the exact DNS queries and HTTP request, then feeds both circuits their recorded responses. Both token hashes should match the live run, with `network_connections: 0` and `dns_network_exchanges: 0` on replay. Recordings from v0.1 predate compiled DNS and must be regenerated. Keep raw recordings private: responses can contain request identifiers and cookies. `runs/` is git-ignored.

## DNS fallbacks

Run these offline, or add `--live` to exchange the circuit's queries with your configured resolver:

```sh
uv run python -m fetch3_browser.dns_runtime --scenario tcp-retry
uv run python -m fetch3_browser.dns_runtime --scenario aaaa-fallback
```

These are **explicit branch tests**. `tcp-retry` replaces the first A response with a synthetic truncated (TC) response. The circuit then constructs its retry over TCP. `aaaa-fallback` replaces the first A response with synthetic NODATA; the circuit then constructs an AAAA query. In live mode the subsequent TCP/AAAA exchanges use the real resolver. The receipt labels the injected response; these tests do not claim that the live resolver naturally returned TC or NODATA.

`--record runs/dns.json` and `--replay runs/dns.json` also work on the DNS command; pass the same `--scenario` and `--url` when replaying. These DNS-only tests do not open an HTTP connection. IPv6 DNS query/answer handling has been tested; IPv6 HTTPS connectivity is not claimed.

## What “in weights” means here

```text
URL/action tokens
       |
       v
DNS circuit -> query bytes -> UDP/TCP byte bridge
     ^                              |
     +---------- raw DNS reply -----+
     |
     +-> circuit-selected IP
                  |
Fetch3 circuit -> HTTP request -> socket/TLS bridge
     ^                              |
     +-------- raw HTTP response ---+
       |
       v
text, numbered links, next-page/follow actions
```

The Tern programs [`fetch3_net.tern`](examples/wire/fetch3_net.tern) and [`dns_net.tern`](examples/wire/dns_net.tern) were compiled to [`weights/`](fetch3_browser/weights/) and [`dns_weights/`](fetch3_browser/dns_weights/). The Python circuit evaluator executes exported sparse ReGLU operations and integer-softmax attention. DNS parsing/address selection and HTTP/HTML decisions come from the circuits. Python does not use an HTML parser or HTTP client to produce the displayed page. Separate specifications check the result but do not supply the circuits' outputs.

Python still runs the numerical interpreter, token/position features, response-tape bookkeeping, navigation loop, and socket/TLS bridge. It sequences **two circuits in separate token streams**; this is not a merged language model. The bridge chooses the configured resolver, transports the circuit's DNS packets (including DNS-over-TCP length framing), rechecks the selected destination, connects to that numeric IP, and handles TLS. TLS itself is **not** in the weights. No ordinary language-model weights are included.

The ReGLU coefficient matrices use −1, 0, +1. Constant cells and exported attention forms include larger integer values/scales. Calling every number in the artifact a trit would be inaccurate.

## Links, paging, and limits

```sh
# Follow the first numbered link after reading a page:
uv run python -m fetch3_browser --live --url https://example.com --follow 1

# Read at most two circuit page segments:
uv run python -m fetch3_browser --live --pages 2
```

The circuit is a deliberately limited text browser: no JavaScript execution, arbitrary Unicode rendering, images, authentication, or full browser compatibility. URLs are at most 64 characters, there are at most 32 numbered links per page, and a session has at most 16,384 positions. The demo caps a raw network response at 8,192 bytes. Long pages can therefore be cut before rendering is complete. After a follow, the CLI displays the most recent response.

The origin allowlist is compiled into the included artifact: `example.com` (HTTP and HTTPS), `www.iana.org`, `en.wikipedia.org`, `news.ycombinator.com`, `www.gnu.org`, `github.com`, and `httpbin.org` (HTTPS). Different origins require recompilation. The transport also rejects non-public destination addresses. The example.com page usually has no recognized numbered links; `--follow 1` is not a promise that a link exists.

## Source and verification

This is a standalone extraction from the Ternary project by Khalil Hughes. [`PROVENANCE.json`](PROVENANCE.json) records the source snapshot and artifact hashes; there is no dependency on the original checkout. Only NumPy is required to run it. Artifacts are checksum-verified when loaded.

The four Tern source files are included for inspection. Rebuilding them requires the Tern interpreter/compiler, which is not bundled in this small demo. The shipped circuits are precompiled; running them does not invoke a compiler or fetch weights from another repository.

```sh
uv run python -m unittest discover -s tests -v

# Opt-in real-network handoff test: disables socket.getaddrinfo, checks the
# selected IP against the actual TLS peer, then replays with sockets disabled.
uv run python tests/live_dns_handoff.py
```

For a short description to share, see [`SHARE.md`](SHARE.md).

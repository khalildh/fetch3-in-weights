# Fetch3: a text-mode browser compiled into weights

HTTP request construction and HTML parsing, represented as an integer circuit with sparse ReGLU layers and attention heads. The circuit emits request bytes, reads the response, skips markup/scripts/comments, and produces text and numbered links. It can follow a link or continue a page using its token protocol.

**This is an executable Python/NumPy reference demo of a compiled circuit. It is not an LLM, a trained browser agent, or a GPU demo.** No training, model API, or API key is required. The caller supplies the URL and navigation actions.

## Run it

Python 3.11+ and [uv](https://docs.astral.sh/uv/) are sufficient:

```sh
uv run python -m fetch3_browser
```

The default uses a bundled HTML fixture and makes **zero network connections**. It prints:

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

Replay checks that the circuit generates exactly the recorded request and feeds it the recorded response. The token hash should match the live run, with `network_connections: 0` on replay. Keep raw recordings private: responses can contain request identifiers and cookies. `runs/` is git-ignored.

## What “in weights” means here

```text
URL/action tokens
       |
       v
compiled circuit -> exact HTTP request bytes
                            |
                     Python socket/TLS bridge
                            |
compiled circuit <- raw HTTP response bytes
       |
       v
text, numbered links, next-page/follow actions
```

The Tern program in [`examples/wire/fetch3_net.tern`](examples/wire/fetch3_net.tern) was compiled to the artifacts in [`fetch3_browser/weights/`](fetch3_browser/weights/). The Python circuit evaluator executes exported sparse ReGLU operations and integer-softmax attention. The HTTP/HTML decisions come from the circuit; Python does not use an HTML parser or HTTP client to produce the displayed page. The separate specification checks the result but does not supply the circuit's outputs.

Python still runs the numerical interpreter, token/position features, response-tape bookkeeping, navigation loop, and socket/TLS bridge. The bridge resolves addresses, checks the destination, establishes the connection, sends the circuit's bytes verbatim, and returns raw bytes. DNS and TLS are **not** compiled into this demo's weights. No ordinary language-model weights are included.

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

The three Tern source files are included for inspection. Rebuilding them requires the Tern interpreter/compiler, which is not bundled in this small demo. The shipped circuit is precompiled; running it does not invoke a compiler or fetch weights from another repository.

```sh
uv run python -m unittest discover -s tests -v
```

For a short description to share, see [`SHARE.md`](SHARE.md).

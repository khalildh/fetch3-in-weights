# Verification

Checked on Apple Silicon on 2026-11-07 using the standalone repository's own environment.

- Offline circuit execution and exact replay: passed. Socket creation was disabled in the test. The rendered page includes text and a numbered link and excludes the fixture's script/comment contents.
- Modified-request replay rejection: passed.
- Live HTTPS to `https://example.com`: completed at `<|fetch_done|>` with one TCP connection, TLS 1.3, certificate verification, 134 request bytes and 928 raw response bytes. Circuit and specification agreed at all 1,103 token positions.
- Replay of that recorded live response: identical token-stream hash, zero network connections.

The live request SHA-256 was `d358038dbaa59af084ae0b874f4cfac45a756b2655e5172f2d67032b059e69a7`. The token-stream SHA-256 was `36d23e164285d9510996ade222ea92b106cadc8c9e7ad6bb0120cfa107f9bcf8`.

Fresh live runs may differ because real HTTP responses include changing headers and content. Replay of one saved response must reproduce that run's exact token hash. Raw network recordings are deliberately excluded from this repository.

This verifies a Python/NumPy execution of the exported circuit. It does not verify GPU execution, autonomous model navigation, or compatibility with arbitrary websites.

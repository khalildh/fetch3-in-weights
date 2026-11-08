import unittest
from unittest.mock import patch

from fetch3_browser.demo import Transport, run


class DemoTests(unittest.TestCase):
    def test_compiled_fixture_and_exact_replay(self):
        with patch("socket.socket", side_effect=AssertionError("offline execution opened a socket")):
            text, proof, records = run()
            replay_text, replay_proof, _ = run(replay=records)
        self.assertIn("Hello from a browser made of weights.", text)
        self.assertIn("[1]A numbered link", text)
        self.assertNotIn("do_not_show_this_script", text)
        self.assertNotIn("do not show this comment", text)
        self.assertEqual(text, replay_text)
        self.assertTrue(proof["circuit_matches_spec"])
        self.assertEqual(proof["network_connections"], 0)
        self.assertEqual(proof["token_sha256"], replay_proof["token_sha256"])
        request = bytes.fromhex(records[0]["request_hex"])
        self.assertTrue(request.startswith(b"GET / HTTP/1.1\r\nHost: example.com\r\n"))
        self.assertTrue(request.endswith(b"Connection: close\r\n\r\n"))

    def test_replay_rejects_changed_request(self):
        transport = Transport(["https://example.com"], replay=[dict(scheme="https", target="example.com:443",
            request_hex=b"GET /expected".hex(), response_hex="", term=0)])
        with self.assertRaisesRegex(ValueError, "differs"):
            transport.exchange("https", "example.com:443", b"GET /different")


if __name__ == "__main__":
    unittest.main()

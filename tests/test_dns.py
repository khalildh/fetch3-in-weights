import socket
import unittest
from unittest.mock import patch

from fetch3_browser.dns_runtime import Resolver


class CompiledDnsTests(unittest.TestCase):
    def test_udp_resolution_and_replay_without_sockets(self):
        with patch.object(socket,"socket",side_effect=AssertionError("offline DNS opened a socket")):
            proof,records=Resolver().resolve("https://example.com")
            replay,_=Resolver(replay=list(records)).resolve("https://example.com")
        self.assertEqual(proof['selected_address'],'93.184.216.34')
        self.assertTrue(proof['circuit_matches_spec'])
        self.assertTrue(proof['independent_resolver_matches'])
        self.assertEqual(proof['token_sha256'],replay['token_sha256'])
        self.assertEqual(proof['network_exchanges'],0)
        self.assertEqual([(x['transport'],x['qtype']) for x in records],[('udp',1)])

    def test_circuit_requests_tcp_after_truncation(self):
        proof,records=Resolver(scenario='tcp-retry').resolve("https://example.com")
        self.assertEqual([(x['transport'],x['qtype']) for x in records],[('udp',1),('tcp',1)])
        self.assertEqual(records[0]['injected_response'],'synthetic TC response')
        self.assertEqual(proof['address_version'],4)

    def test_circuit_requests_aaaa_after_nodata(self):
        proof,records=Resolver(scenario='aaaa-fallback').resolve("https://example.com")
        self.assertEqual([(x['transport'],x['qtype']) for x in records],[('udp',1),('udp',28)])
        self.assertEqual(proof['address_version'],6)
        self.assertEqual(proof['selected_address'],'2606:4700:4700::1111')

    def test_private_address_is_refused_by_circuit(self):
        with patch.object(socket,"socket",side_effect=AssertionError("private address caused I/O")):
            with self.assertRaisesRegex(ValueError,"priv"):
                Resolver(scenario='private').resolve("https://example.com")

    def test_dns_replay_rejects_changed_query(self):
        _,records=Resolver().resolve("https://example.com")
        records[0]['query_hex']='00'
        with self.assertRaisesRegex(ValueError,"DNS replay request differs"):
            Resolver(replay=records).resolve("https://example.com")


if __name__=='__main__':
    unittest.main()

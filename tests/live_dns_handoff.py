"""Opt-in real-network check. No hostname resolver may run in this process.

    uv run python tests/live_dns_handoff.py

Writes raw replay data only into git-ignored runs/.
"""
import json
import socket
from pathlib import Path
from unittest.mock import patch

from fetch3_browser.demo import run

with patch.object(socket,'getaddrinfo',side_effect=AssertionError('unexpected host name resolution')):
    text,proof,records=run(live=True)
receipt=proof['receipts'][0]
assert receipt['dns']['selected_address']==receipt['connected_address']==receipt['peer_address']
assert receipt['certificate_verified'] and receipt['tls_server_hostname']=='example.com'
assert receipt['dns']['circuit_matches_spec'] and proof['circuit_matches_spec']
assert proof['network_connections']==1 and proof['dns_network_exchanges']>=1
with patch.object(socket,'socket',side_effect=AssertionError('replay opened a socket')):
    _,replay,_=run(replay=records)
assert proof['token_sha256']==replay['token_sha256']
assert receipt['dns']['token_sha256']==replay['receipts'][0]['dns']['token_sha256']
assert replay['network_connections']==replay['dns_network_exchanges']==0
Path('runs').mkdir(exist_ok=True)
Path('runs/dns-handoff.json').write_text(json.dumps(dict(url='https://example.com',
    exchanges=records,proof=proof),indent=2)+'\n')
report=dict(host_name_resolution_disabled=True,dns_selected_peer_verified=True,
    live=proof,replay=replay,claim_limit='Two compiled circuits chained by the host runtime; no GPU or LLM policy.')
Path('runs/dns-handoff-report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

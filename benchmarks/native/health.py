"""Bounded compute health probe; never resets devices or kills other workloads."""

import argparse
import os
import socket

parser = argparse.ArgumentParser()
parser.add_argument('--device', required=True)
args = parser.parse_args()
parts = args.device.split(',')
if not all(p.isascii() and p.isdecimal() for p in parts) or len({int(p) for p in parts}) != len(parts):
    parser.error('distinct decimal device IDs required')

# torchrun rendezvous and c10d resolve the container hostname even on one node.
# In an iTASK Pod that name is the node FQDN, which is often absent from
# /etc/hosts; fail here, before a build, instead of timing out inside a rank.
name = socket.gethostname()
try:
    socket.getaddrinfo(name, None)
except socket.gaierror:
    raise SystemExit(
        f'hostname {name!r} does not resolve; torchrun will hang for 300s per rank.\n'
        f'fix inside the Pod: echo "127.0.0.1 {name}" >> /etc/hosts')

os.environ['ASCEND_RT_VISIBLE_DEVICES'] = ','.join(str(int(p)) for p in parts)
import torch  # noqa: E402
import torch_npu  # noqa: E402,F401

assert torch.npu.is_available() and torch.npu.device_count() == len(parts)
reference = torch.arange(16, dtype=torch.float32).reshape(4, 4)
for rank, physical in enumerate(parts):
    value = reference.to(f'npu:{rank}')
    torch.testing.assert_close((value @ value.T).cpu(), reference @ reference.T)
    print(f'PASS logical={rank} selected_device={physical}', flush=True)

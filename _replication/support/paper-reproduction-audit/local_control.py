#!/usr/bin/env python3
"""Documented local app-server JSON-RPC over its existing Unix WebSocket.

No thread creation, daemon restart, account/auth change or model override.
"""
import base64
import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import time


class LocalControl:
    def __init__(self, timeout=20):
        self.s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.s.settimeout(timeout)
        self.s.connect(str(Path.home() / '.codex/app-server-control/app-server-control.sock'))
        key = base64.b64encode(os.urandom(16)).decode()
        request = ('GET / HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\n'
                   'Connection: Upgrade\r\nSec-WebSocket-Version: 13\r\n'
                   f'Sec-WebSocket-Key: {key}\r\n\r\n')
        self.s.sendall(request.encode())
        header = bytearray()
        while not header.endswith(b'\r\n\r\n'):
            header += self.exact(1)
            if len(header) > 16384:
                raise RuntimeError('Oversized WebSocket handshake')
        if b' 101 ' not in header.split(b'\r\n', 1)[0]:
            raise RuntimeError('Local WebSocket handshake rejected: ' + header[:200].decode(errors='replace'))
        accept = base64.b64encode(hashlib.sha1((key + '258EAFA5-E914-47DA-95CA-C5AB0DC85B11').encode()).digest())
        if accept not in header:
            raise RuntimeError('Invalid WebSocket accept')
        self.next_id = 1
        self.events = []
        self.call('initialize', {'clientInfo': {'name': 'paper_reproduction_supervisor',
                                               'title': 'Paper reproduction supervisor', 'version': '1.0'}})
        self.send({'method': 'initialized', 'params': {}})

    def exact(self, n):
        result = bytearray()
        while len(result) < n:
            block = self.s.recv(n - len(result))
            if not block:
                raise EOFError('Local app-server closed the connection')
            result += block
        return bytes(result)

    def frame(self, payload, opcode=1):
        mask = os.urandom(4)
        n = len(payload)
        header = bytes([0x80 | opcode])
        header += (bytes([0x80 | n]) if n < 126 else
                   bytes([0x80 | 126]) + struct.pack('!H', n) if n < 65536 else
                   bytes([0x80 | 127]) + struct.pack('!Q', n))
        self.s.sendall(header + mask + bytes(v ^ mask[i % 4] for i, v in enumerate(payload)))

    def send(self, value):
        self.frame(json.dumps(value).encode())

    def receive(self):
        chunks = bytearray()
        while True:
            a, b = self.exact(2)
            n = b & 127
            if n == 126:
                n = struct.unpack('!H', self.exact(2))[0]
            elif n == 127:
                n = struct.unpack('!Q', self.exact(8))[0]
            if n > 64 * 1024 ** 2:
                raise RuntimeError('Oversized local frame')
            mask = self.exact(4) if b & 128 else None
            body = self.exact(n)
            if mask:
                body = bytes(v ^ mask[i % 4] for i, v in enumerate(body))
            opcode = a & 15
            if opcode == 8:
                raise EOFError('WebSocket closed')
            if opcode == 9:
                self.frame(body, 10)
                continue
            if opcode == 10:
                continue
            chunks += body
            if a & 128:
                return json.loads(chunks)

    def call(self, method, params):
        ident = self.next_id
        self.next_id += 1
        self.send({'id': ident, 'method': method, 'params': params})
        while True:
            value = self.receive()
            if value.get('id') == ident:
                return value
            if 'id' in value and 'method' in value:
                # No permission grant or unsolicited action is implemented here.
                self.send({'id': value['id'], 'error': {'code': -32601, 'message': 'Read/control client does not handle this request'}})
            else:
                self.events.append(value)

    def close(self):
        self.s.close()


if __name__ == '__main__':
    c = LocalControl()
    try:
        for tid in ['01a1b0f6-52ff-7141-855a-b99ba18249a0',
                    '01a1b0f8-bb07-7cd0-9865-61ffecf66ee5',
                    '01a1b0f7-53ee-76d0-a2bf-d3a205b9b74a',
                    '01a1b0f7-5d65-7900-ba0c-83ba2379099c']:
            r = c.call('thread/read', {'threadId': tid})
            t = r.get('result', {}).get('thread', {})
            print(json.dumps({'id': tid, 'title': t.get('title'), 'status': t.get('status'), 'error': r.get('error')}), flush=True)
    finally:
        c.close()

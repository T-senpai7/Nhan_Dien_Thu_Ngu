"""MQTT 3.1.1 broker toi gian (asyncio) de chay demo khong can cai Mosquitto / Internet.

Ho tro: CONNECT, PUBLISH (QoS 0/1, chuyen tiep QoS 0), SUBSCRIBE (wildcard + #),
UNSUBSCRIBE, PINGREQ, DISCONNECT. Khong ho tro retained message va session ben vung.

Chay:  python -m network.mini_broker [--host 0.0.0.0] [--port 1883]
"""
import argparse
import asyncio
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CONNECT, CONNACK, PUBLISH, PUBACK = 1, 2, 3, 4
SUBSCRIBE, SUBACK, UNSUBSCRIBE, UNSUBACK = 8, 9, 10, 11
PINGREQ, PINGRESP, DISCONNECT = 12, 13, 14


def topic_matches(sub: str, topic: str) -> bool:
    sub_parts, topic_parts = sub.split("/"), topic.split("/")
    for i, part in enumerate(sub_parts):
        if part == "#":
            return True
        if i >= len(topic_parts):
            return False
        if part != "+" and part != topic_parts[i]:
            return False
    return len(sub_parts) == len(topic_parts)


def encode_remaining_length(length: int) -> bytes:
    out = bytearray()
    while True:
        byte = length % 128
        length //= 128
        if length:
            byte |= 0x80
        out.append(byte)
        if not length:
            return bytes(out)


def encode_str(value: str) -> bytes:
    raw = value.encode()
    return struct.pack("!H", len(raw)) + raw


class Session:
    def __init__(self, writer: asyncio.StreamWriter):
        self.writer = writer
        self.client_id = "?"
        self.subscriptions = set()

    def send(self, packet_type: int, flags: int, body: bytes):
        self.writer.write(bytes([(packet_type << 4) | flags]) + encode_remaining_length(len(body)) + body)


class MiniBroker:
    def __init__(self, verbose: bool = False):
        self.sessions = set()
        self.verbose = verbose
        self.forwarded = 0

    def log(self, *args):
        if self.verbose:
            print("[broker]", *args, flush=True)

    async def read_packet(self, reader: asyncio.StreamReader):
        header = await reader.readexactly(1)
        multiplier, length = 1, 0
        while True:
            byte = (await reader.readexactly(1))[0]
            length += (byte & 0x7F) * multiplier
            if not byte & 0x80:
                break
            multiplier *= 128
        body = await reader.readexactly(length) if length else b""
        return header[0] >> 4, header[0] & 0x0F, body

    def route(self, topic: str, payload: bytes):
        body = encode_str(topic) + payload
        for session in list(self.sessions):
            if any(topic_matches(sub, topic) for sub in session.subscriptions):
                try:
                    session.send(PUBLISH, 0, body)
                    self.forwarded += 1
                except Exception:
                    pass

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        session = Session(writer)
        self.sessions.add(session)
        try:
            while True:
                ptype, flags, body = await self.read_packet(reader)
                if ptype == CONNECT:
                    name_len = struct.unpack("!H", body[:2])[0]
                    pos = 2 + name_len + 1 + 1 + 2
                    cid_len = struct.unpack("!H", body[pos:pos + 2])[0]
                    session.client_id = body[pos + 2:pos + 2 + cid_len].decode(errors="replace")
                    session.send(CONNACK, 0, b"\x00\x00")
                    self.log("connect", session.client_id)
                elif ptype == PUBLISH:
                    qos = (flags >> 1) & 0x03
                    tlen = struct.unpack("!H", body[:2])[0]
                    topic = body[2:2 + tlen].decode()
                    pos = 2 + tlen
                    if qos > 0:
                        packet_id = body[pos:pos + 2]
                        pos += 2
                        session.send(PUBACK, 0, packet_id)
                    self.route(topic, body[pos:])
                elif ptype == SUBSCRIBE:
                    packet_id, pos, granted = body[:2], 2, bytearray()
                    while pos < len(body):
                        tlen = struct.unpack("!H", body[pos:pos + 2])[0]
                        session.subscriptions.add(body[pos + 2:pos + 2 + tlen].decode())
                        pos += 2 + tlen + 1
                        granted.append(0)
                    session.send(SUBACK, 0, packet_id + bytes(granted))
                    self.log("subscribe", session.client_id, session.subscriptions)
                elif ptype == UNSUBSCRIBE:
                    packet_id, pos = body[:2], 2
                    while pos < len(body):
                        tlen = struct.unpack("!H", body[pos:pos + 2])[0]
                        session.subscriptions.discard(body[pos + 2:pos + 2 + tlen].decode())
                        pos += 2 + tlen
                    session.send(UNSUBACK, 0, packet_id)
                elif ptype == PINGREQ:
                    session.send(PINGRESP, 0, b"")
                elif ptype == DISCONNECT:
                    break
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            self.sessions.discard(session)
            self.log("disconnect", session.client_id)
            writer.close()


async def serve(host: str, port: int, verbose: bool):
    broker = MiniBroker(verbose=verbose)
    server = await asyncio.start_server(broker.handle, host, port)
    print(f"[broker] MQTT mini broker dang chay tai {host}:{port}", flush=True)
    async with server:
        await server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description="MQTT mini broker")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    try:
        asyncio.run(serve(args.host, args.port, args.verbose))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

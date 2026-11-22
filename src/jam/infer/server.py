"""Minimal policy server for closed-loop clients."""

from __future__ import annotations

import argparse
import io
import socket
import socketserver
import struct
import threading

import numpy as np

from .policy import Policy, PolicyConfig
from .sampler import MODES

REQUEST_KEYS = ("obs_latent", "text", "text_mask", "state", "future")


def pack(arrays: dict) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **{k: np.asarray(v) for k, v in arrays.items()})
    payload = buffer.getvalue()
    return struct.pack(">Q", len(payload)) + payload


def unpack(payload: bytes) -> dict:
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def read_message(stream) -> dict | None:
    header = stream.read(8)
    if len(header) < 8:
        return None
    (size,) = struct.unpack(">Q", header)
    payload = stream.read(size)
    if len(payload) < size:
        return None
    return unpack(payload)


class PolicyServer:
    def __init__(self, policy: Policy):
        self.policy = policy
        self.lock = threading.Lock()

    def infer(self, request: dict) -> dict:
        unknown = sorted(set(request) - set(REQUEST_KEYS))
        if unknown or "obs_latent" not in request or "text" not in request:
            raise ValueError(f"a request needs obs_latent and text; unknown keys: {unknown}")
        with self.lock:
            out = self.policy.act(**request)
        return {"actions": out["actions"], "actions_layout": out["actions_layout"]}

    def serve(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        owner = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                while (request := read_message(self.rfile)) is not None:
                    try:
                        reply = owner.infer(request)
                    except Exception as error:
                        reply = {"error": np.asarray(repr(error))}
                    self.wfile.write(pack(reply))

        class Server(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True

        with Server((host, port), Handler) as server:
            server.serve_forever()


class PolicyClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 8000):
        self.stream = socket.create_connection((host, port)).makefile("rwb")

    def infer(self, **arrays) -> dict:
        self.stream.write(pack({k: v for k, v in arrays.items() if v is not None}))
        self.stream.flush()
        reply = read_message(self.stream)
        if reply is None:
            raise ConnectionError("the policy server closed the connection")
        if "error" in reply:
            raise RuntimeError(str(reply["error"]))
        return reply


def main():
    parser = argparse.ArgumentParser(description="Serve a JAM policy.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", default="joint", choices=MODES)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--lag", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--future-frames", type=int, default=2)
    parser.add_argument("--consequence-mode", default="denoise", choices=("denoise", "absent"))
    parser.add_argument("--binding", help="embodiment binding used to map raw states and actions to the layout")
    parser.add_argument("--stats", help="JSON file with normalisation ranges for actions and states")
    parser.add_argument("--execute-steps", type=int, help="return only the first N actions of the chunk")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    pcfg = PolicyConfig(
        checkpoint=args.checkpoint, mode=args.mode, steps=args.steps, lag=args.lag, seed=args.seed, future_frames=args.future_frames,
        consequence_mode=args.consequence_mode, binding=args.binding, stats=args.stats, execute_steps=args.execute_steps,
    )
    PolicyServer(Policy(pcfg)).serve(args.host, args.port)


if __name__ == "__main__":
    main()

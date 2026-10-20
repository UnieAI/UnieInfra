# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Subprocess supervision helpers for ``ts serve``."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
import time

logger = logging.getLogger(__name__)

_GATEWAY_MODULE = "smg"
_ENGINE_MODULE_DEFAULT = "smg_grpc_servicer.tokenspeed"
_PIPE_LINE_LIMIT = 64 * 1024 * 1024

# ---------------------------------------------------------------------------
# smg gateway TCP_NODELAY shim
#
# The bundled smg HTTP gateway (a prebuilt Rust extension) does not set
# TCP_NODELAY on the client-facing sockets it accepts. Under HTTP keep-alive —
# which every real OpenAI client and ``vllm bench serve`` uses — the first
# streamed SSE chunk of each response is a sub-MSS write that Nagle's algorithm
# holds until the peer's delayed ACK (~40 ms on Linux) arrives, inflating
# single-stream TTFT at concurrency 1 by ~40 ms. smg resolves accept4()/accept()
# dynamically from libc, so an LD_PRELOAD shim that flips TCP_NODELAY on every
# accepted socket removes the stall without any throughput cost.
#
# Opt out with TOKENSPEED_GATEWAY_TCP_NODELAY=0.
_NODELAY_ENABLE_ENV = "TOKENSPEED_GATEWAY_TCP_NODELAY"
_NODELAY_SRC_NAME = "tcp_nodelay_preload.c"
_NODELAY_SO_NAME = "libtcp_nodelay_preload.so"


def _repo_scripts_dir() -> str | None:
    """Locate the repo ``scripts/`` dir holding the TCP_NODELAY shim source."""
    here = os.path.dirname(os.path.abspath(__file__))
    # python/tokenspeed/cli/_proc.py -> repo root is three levels up.
    root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    scripts = os.path.join(root, "scripts")
    return scripts if os.path.isdir(scripts) else None


def _ensure_nodelay_shim() -> str | None:
    """Return a path to the compiled TCP_NODELAY shim, building it if needed.

    Returns ``None`` (and logs, non-fatally) when the shim is unavailable so the
    gateway still launches — just without the Nagle mitigation.
    """
    scripts = _repo_scripts_dir()
    if scripts is None:
        return None
    so_path = os.path.join(scripts, _NODELAY_SO_NAME)
    src_path = os.path.join(scripts, _NODELAY_SRC_NAME)
    try:
        fresh = os.path.isfile(so_path) and (
            not os.path.isfile(src_path)
            or os.path.getmtime(so_path) >= os.path.getmtime(src_path)
        )
        if fresh:
            return so_path
        if not os.path.isfile(src_path):
            return so_path if os.path.isfile(so_path) else None
        import shutil
        import subprocess

        cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")
        if cc is None:
            return so_path if os.path.isfile(so_path) else None
        subprocess.run(
            [cc, "-O2", "-fPIC", "-shared", "-o", so_path, src_path, "-ldl"],
            check=True,
            capture_output=True,
        )
        logger.info("built gateway TCP_NODELAY shim: %s", so_path)
        return so_path
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "gateway TCP_NODELAY shim unavailable (%s); " "serving without it", exc
        )
        return so_path if os.path.isfile(so_path) else None


def _gateway_env() -> dict | None:
    """Build the gateway subprocess env, preloading the TCP_NODELAY shim.

    Returns ``None`` to inherit the parent env unchanged when the mitigation is
    disabled or the shim can't be produced.
    """
    if os.environ.get(_NODELAY_ENABLE_ENV, "1") == "0":
        return None
    so_path = _ensure_nodelay_shim()
    if not so_path:
        return None
    env = dict(os.environ)
    existing = env.get("LD_PRELOAD", "")
    if so_path not in existing.split(os.pathsep):
        env["LD_PRELOAD"] = so_path + os.pathsep + existing if existing else so_path
    return env


async def spawn_engine(
    args: list[str],
    *,
    host: str,
    port: int,
) -> asyncio.subprocess.Process:
    """Spawn the engine subprocess with PIPE stdio."""
    module = os.environ.get("TS_SERVE_ENGINE_MODULE", _ENGINE_MODULE_DEFAULT)
    cmd = [
        sys.executable,
        "-m",
        module,
        "--host",
        host,
        "--port",
        str(port),
        *args,
    ]
    logger.info("spawn engine: %s", " ".join(cmd))
    return await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=_PIPE_LINE_LIMIT,
    )


async def spawn_gateway(
    args: list[str],
    *,
    engine_host: str,
    engine_port: int,
) -> asyncio.subprocess.Process:
    """Spawn ``python -m smg launch`` with PIPE stdio."""
    cmd = [
        sys.executable,
        "-m",
        _GATEWAY_MODULE,
        "launch",
        "--worker-urls",
        f"grpc://{engine_host}:{engine_port}",
        "--disable-retries",
        "--disable-circuit-breaker",
        *args,
    ]
    logger.info("spawn gateway: %s", " ".join(cmd))
    return await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=_PIPE_LINE_LIMIT,
        env=_gateway_env(),
    )


async def wait_grpc_serving(
    target: str,
    *,
    timeout: float,
    poll_interval: float = 1.0,
) -> None:
    """Poll ``grpc.health.v1.Health.Check`` on ``target`` until SERVING."""
    import grpc
    from grpc_health.v1 import health_pb2, health_pb2_grpc

    deadline = time.monotonic() + timeout
    last_err: Exception | None = None
    async with grpc.aio.insecure_channel(target) as channel:
        stub = health_pb2_grpc.HealthStub(channel)
        while time.monotonic() < deadline:
            try:
                resp = await stub.Check(
                    health_pb2.HealthCheckRequest(service=""),
                    timeout=2.0,
                )
                if resp.status == health_pb2.HealthCheckResponse.SERVING:
                    return
            except Exception as exc:  # noqa: BLE001
                last_err = exc
            await asyncio.sleep(poll_interval)
    detail = f" (last error: {last_err!r})" if last_err is not None else ""
    raise TimeoutError(
        f"engine never reached SERVING on {target} within {timeout:.0f}s{detail}"
    )


async def wait_http_ready(
    url: str,
    *,
    timeout: float,
    poll_interval: float = 1.0,
) -> None:
    """Poll HTTP ``GET <url>`` until 200, or raise ``TimeoutError``."""
    import aiohttp

    deadline = time.monotonic() + timeout
    last_err: Exception | None = None
    async with aiohttp.ClientSession() as session:
        while time.monotonic() < deadline:
            try:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=2.0)
                ) as resp:
                    if resp.status == 200:
                        return
            except Exception as exc:  # noqa: BLE001
                last_err = exc
            await asyncio.sleep(poll_interval)
    detail = f" (last error: {last_err!r})" if last_err is not None else ""
    raise TimeoutError(
        f"gateway never reached 200 on {url} within {timeout:.0f}s{detail}"
    )


async def terminate_then_kill(proc, *, drain_timeout: float) -> None:
    """SIGTERM, wait up to ``drain_timeout``, then SIGKILL if still alive."""
    if proc.returncode is not None:
        return
    proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), timeout=drain_timeout)
    except asyncio.TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        with contextlib.suppress(Exception):
            await proc.wait()

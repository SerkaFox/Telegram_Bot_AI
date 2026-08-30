"""Entrypoint: `python -m fox_worker.main`. Wires config → client → services → worker loop."""
from __future__ import annotations

import asyncio
import logging
import signal

from generator import SafeGeneratorService, AdultGeneratorService

from .client import FoxCoreClient
from .config import load_config
from .worker import Worker


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    # Never let requests/urllib3 echo full URLs (which could contain query creds) at INFO.
    logging.getLogger("urllib3").setLevel(logging.WARNING)


async def _amain() -> None:
    cfg = load_config()
    client = FoxCoreClient(cfg)
    worker = Worker(cfg, client, SafeGeneratorService(), AdultGeneratorService())

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, worker.request_stop)
        except NotImplementedError:  # pragma: no cover
            pass
    await worker.run()


def main() -> None:
    _setup_logging()
    asyncio.run(_amain())


if __name__ == "__main__":
    main()

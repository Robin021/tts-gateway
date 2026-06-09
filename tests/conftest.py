"""Helpers shared across tests. We don't have pytest-asyncio in this
environment, so each async test is wrapped in asyncio.run() via run_async.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable

# Make the package importable without `pip install -e .`.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_async(coro_fn: Callable[[], Awaitable[Any]]) -> Any:
    return asyncio.run(coro_fn())

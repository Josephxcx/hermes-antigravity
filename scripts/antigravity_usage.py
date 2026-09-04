#!/usr/bin/env python3
"""CLI runner for antigravity.usage command."""

import asyncio
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from hermes_antigravity import command_antigravity_usage


async def main():
    res = await command_antigravity_usage()
    print(res)


if __name__ == "__main__":
    asyncio.run(main())

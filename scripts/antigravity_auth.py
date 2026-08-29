#!/usr/bin/env python3
"""CLI runner for antigravity.auth command."""

import asyncio
import sys
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from hermes_antigravity import command_antigravity_auth


async def main():
    callback = sys.argv[1] if len(sys.argv) > 1 else None
    res = await command_antigravity_auth(callback=callback)
    print(res)


if __name__ == "__main__":
    asyncio.run(main())

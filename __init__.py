"""Hermes plugin root initialization."""

import sys
from pathlib import Path

# Ensure plugin directory is in sys.path when imported as a user plugin
_plugin_root = str(Path(__file__).resolve().parent)
if _plugin_root not in sys.path:
    sys.path.insert(0, _plugin_root)

from hermes_antigravity import (
    AntigravityProviderProfile,
    antigravity_profile,
    command_antigravity_auth,
    command_antigravity_doctor,
    command_antigravity_quota,
)

__all__ = [
    "antigravity_profile",
    "AntigravityProviderProfile",
    "command_antigravity_auth",
    "command_antigravity_quota",
    "command_antigravity_doctor",
]

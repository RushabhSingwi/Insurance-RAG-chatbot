"""
Debug Utility Module

Provides utilities for two-level debugging:
- Level 0 (off): No debug output
- Level 1 (basic): Essential information (queries, token usage, sources)
- Level 2 (verbose): Full details (complete prompts, responses, metadata)

Set via environment variable:
- DEBUG=off or DEBUG=0 or DEBUG=false → Level 0
- DEBUG=basic or DEBUG=1 or DEBUG=true → Level 1
- DEBUG=verbose or DEBUG=2 → Level 2
"""

import os
from typing import Literal

DebugLevel = Literal[0, 1, 2]


def get_debug_level() -> DebugLevel:
    """
    Get the current debug level from environment variable.

    Returns:
        0: No debug output
        1: Basic debug output (queries, token usage, sources)
        2: Verbose debug output (full prompts, responses, all metadata)
    """
    debug = os.getenv("DEBUG", "off").lower().strip()

    # Verbose level
    if debug in ["verbose", "2", "v", "full"]:
        return 2

    # Basic level
    if debug in ["basic", "1", "b", "true", "on"]:
        return 1

    # Off
    return 0


def is_debug_basic() -> bool:
    """Check if basic debug mode is enabled (level 1 or higher)."""
    return get_debug_level() >= 1


def is_debug_verbose() -> bool:
    """Check if verbose debug mode is enabled (level 2)."""
    return get_debug_level() >= 2


def print_debug_header(title: str, level: DebugLevel = 1):
    """
    Print a formatted debug header.

    Args:
        title: Header title
        level: Debug level required (1 for basic, 2 for verbose)
    """
    if get_debug_level() >= level:
        print("\n" + "=" * 80)
        print(f"DEBUG: {title}")
        print("=" * 80)


def print_debug_footer(level: DebugLevel = 1):
    """
    Print a formatted debug footer.

    Args:
        level: Debug level required (1 for basic, 2 for verbose)
    """
    if get_debug_level() >= level:
        print("=" * 80)

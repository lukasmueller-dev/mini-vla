# The USER-facing run configuration — the scene difficulty a run trains on
# (palette size, scene density), as opposed to CONFIG (config.py), the developer
# knob sheet. Python port of js/src/run-config.ts.
#
# It deliberately does NOT change any model shape: the color head stays 8-wide
# regardless of the selection — numColors only restricts what the samplers draw,
# so every RunConfig trains the same architecture and the calibrated CONFIG
# numbers stay comparable.

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass
class RunConfig:
    # Palette size — scenes draw colors from the FIRST N entries of COLORS.
    numColors: int = 8  # one of {2, 4, 8}
    # Scene density cap — a scene holds 2..min(maxBlocks, numColors) blocks
    # (colors are unique per scene, so the palette also caps the count).
    maxBlocks: int = 4  # one of {2, 3, 4}


# Default landing-page behavior — what trains when the menu is untouched.
DEFAULT_RUN_CONFIG = RunConfig(numColors=8, maxBlocks=4)

# Active run config (module state, like the JS singleton). task.py reads it.
_current = replace(DEFAULT_RUN_CONFIG)


def set_run_config(rc: RunConfig) -> None:
    """Install the active run config (defensive copy)."""
    global _current
    _current = replace(rc)


def run_config() -> RunConfig:
    return _current

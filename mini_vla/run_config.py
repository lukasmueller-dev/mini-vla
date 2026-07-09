# The run configuration — the scene difficulty a run trains on (palette size,
# scene density), as opposed to CONFIG (config.py), the developer knob sheet.
# Python port of js/src/run-config.ts.
#
# The free user picker is GONE: instead of choosing numColors/maxBlocks by hand,
# the site ships two FIXED named profiles (PRESETS below) and serves one by
# device class — DESKTOP (the hardest setting) by default, MOBILE (a lighter,
# faster-converging task) on phone-class devices.
#
# It deliberately does NOT change any model shape: the color head stays 8-wide
# regardless of the profile — numColors only restricts what the samplers draw,
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


# ---- named profiles the site ships (both get ported to run-config.ts) ------
# DESKTOP — the HARDEST setting: the full 8-color palette, up to 4 blocks/scene.
DESKTOP_RUN_CONFIG = RunConfig(numColors=8, maxBlocks=4)
# MOBILE — the lighter budget profile: a 4-color palette, up to 3 blocks/scene.
# An easier task that converges in fewer batches, to fit a phone's train budget.
MOBILE_RUN_CONFIG = RunConfig(numColors=4, maxBlocks=3)

# Selectable by name — train.py's --preset, and the site's device-class pick.
PRESETS: dict[str, RunConfig] = {
    "desktop": DESKTOP_RUN_CONFIG,
    "mobile": MOBILE_RUN_CONFIG,
}

# Default landing-page behavior — the hardest (desktop) profile.
DEFAULT_RUN_CONFIG = DESKTOP_RUN_CONFIG

# Active run config (module state, like the JS singleton). task.py reads it.
_current = replace(DEFAULT_RUN_CONFIG)


def set_run_config(rc: RunConfig) -> None:
    """Install the active run config (defensive copy)."""
    global _current
    _current = replace(rc)


def run_config() -> RunConfig:
    return _current

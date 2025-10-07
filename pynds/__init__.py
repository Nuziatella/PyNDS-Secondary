"""PyNDS: A Python wrapper for the Nintendo DS emulator.

Your digital time machine to the Nintendo DS universe! This package brings the
power of NooDS emulation to Python, perfect for bot development and automated gaming magic.

Main Components:
    PyNDS: Your core emulator class for loading ROMs and controlling emulation
    config: Configuration management for your emulator settings
    button: Input handling for buttons and touch screen interactions
    memory: Memory access and manipulation toolkit
    window: Display and rendering interface for your games

Examples
--------
    >>> import pynds
    >>> # Basic emulator usage
    >>> nds = pynds.PyNDS("game.nds")
    >>> nds.tick()
    >>> top, bottom = nds.get_frame()

    >>> # Advanced features (Roadmap 3)
    >>> state = nds.save_state()
    >>> nds.load_state(state)
    >>> nds.export_frame("screenshot.png")
"""

from .button import KEY_MAP as BUTTON_KEYS
from .config import config
from .debugger import PyNDSDebugger
from .game_wrappers import (
    BaseGameWrapper,
    MarioWrapper,
    PokemonWrapper,
    ZeldaWrapper,
    create_game_wrapper,
)
from .pynds import PyNDS
from .recorder import PyNDSRecorder

# Friendly alias for just the names (stable order not guaranteed).
VALID_KEYS = tuple(BUTTON_KEYS.keys())

# Export a plain version string for tooling and diagnostics.
# Keep it in lockstep with pyproject.toml's [project].version.
__version__ = "0.0.5-alpha"


def get_build_info() -> dict:
    """Return a small basket of build/runtime facts for logs and bug reports.

    Contents are intentionally boring: version, Python, platform, and package name.
    """
    import platform as _platform
    import sys as _sys

    return {
        "name": "pynds",
        "version": __version__,
        "python": _sys.version.split(" (", 1)[0].strip(),
        "platform": _platform.platform(),
        "implementation": _platform.python_implementation(),
    }


def list_features() -> dict:
    """Return a compact capability map of the installed package.

    We keep this intentionally lightweight: what's available in `pynds.config`
    and a list of headline instance methods apps tend to care about. Think of
    it as a table of contents for your wheel.
    """
    feats: dict = {"module": {}, "config": {}, "instance_api": []}

    # Module facts
    feats["module"] = get_build_info()

    # Config getters/setters
    try:
        cfg = config
        setters = []
        getters = []
        for name in dir(cfg):
            if name.startswith("__"):
                continue
            attr = getattr(cfg, name, None)
            if callable(attr) and name.startswith("set_"):
                setters.append(name)
            if callable(attr) and name.startswith("get_"):
                getters.append(name)
        feats["config"] = {
            "setters": sorted(set(setters)),
            "getters": sorted(set(getters)),
        }
    except Exception:
        feats["config"] = {"error": "unavailable"}

    # Instance API (names only; actual presence depends on the class definition)
    feats["instance_api"] = sorted(
        {
            # lifecycle
            "is_initialized",
            "open_window",
            "close_window",
            "close",
            "reset",
            # stepping
            "tick",
            "run_until_frame",
            "step",
            "run_seconds",
            # frames
            "get_frame",
            "get_frame_as_image",
            "export_frame",
            "export_frames",
            "get_frame_shape",
            "get_frame_format",
            # state
            "save_state",
            "save_state_to_file",
            "load_state",
            "load_state_from_file",
            "get_state_size",
            "validate_state",
            # input/audio/layout
            "button",
            "set_mute",
            "set_layout",
            # platform/timing
            "get_platform",
            "platform",
            "get_fps",
            "get_timing_info",
        }
    )

    return feats


__all__ = [
    "PyNDS",
    "config",
    "BUTTON_KEYS",
    "VALID_KEYS",
    "__version__",
    "get_build_info",
    "list_features",
    # New PyBoy-inspired features
    "PyNDSDebugger",
    "PyNDSRecorder",
    "create_game_wrapper",
    "BaseGameWrapper",
    "PokemonWrapper",
    "MarioWrapper",
    "ZeldaWrapper",
]

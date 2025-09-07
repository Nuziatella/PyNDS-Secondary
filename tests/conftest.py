import os
import tempfile
import warnings
from pathlib import Path

import pytest

# Suppress pkg_resources deprecation warnings from pygame
# This warning is emitted at import time, so we need to set it early
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
warnings.filterwarnings(
    "ignore", message=".*pkg_resources is deprecated.*", category=UserWarning
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files.

    Returns
    -------
    Path
        Path to the temporary directory
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_rom_path():
    """Provide a path to a sample ROM file for testing.

    Note: This fixture assumes you have a test ROM available.
    In a real testing environment, you'd want to include a small
    test ROM or create mock ROM files.

    Returns
    -------
    str
        Path to a sample ROM file, or None if not available
    """
    # In a real implementation, you might:
    # 1. Include a small test ROM in the repository
    # 2. Download a test ROM during test setup
    # 3. Create a mock ROM file for testing

    # For now, return None to indicate no test ROM available
    # Tests that require a ROM should be skipped or mocked
    return None


@pytest.fixture
def mock_rom_path(temp_dir):
    """Create a mock ROM file for testing.

    This creates a fake ROM file that can be used to test
    file validation and basic initialization without requiring
    a real ROM.

    Parameters
    ----------
    temp_dir : Path
        Temporary directory to create the mock ROM in

    Returns
    -------
    str
        Path to the mock ROM file
    """
    mock_rom = temp_dir / "test_game.nds"

    # Create a minimal mock ROM file (just some dummy data)
    # Real ROMs have specific headers, but for basic testing
    # we just need a file that exists
    mock_rom.write_bytes(b"NDS\x00" + b"\x00" * 1000)  # Minimal NDS header

    return str(mock_rom)


@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration settings.

    Returns
    -------
    dict
        Configuration dictionary for tests
    """
    return {
        "test_timeout": 30,  # seconds
        "max_frames": 100,  # maximum frames to run in tests
        "memory_test_size": 1024,  # bytes for memory tests
    }

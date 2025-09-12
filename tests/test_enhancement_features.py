"""Tests for enhancement features: State Management, Frame Control, and Screen Export.

This module contains comprehensive tests for the new enhancement functionality
including in-memory state management, enhanced frame control, and screen export
capabilities using real PyNDS instances with mocked C++ components.
"""

from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import numpy as np
import pytest

from pynds import BUTTON_KEYS, VALID_KEYS
from pynds.button import KEY_MAP

# Import the classes we're testing
from pynds.pynds import PyNDS


class TestInMemoryStateManagement:
    """Test in-memory state management functionality (save_state, load_state, etc.)."""

    @pytest.fixture
    def mock_pynds_instance(self):
        """Create a real PyNDS instance with mocked C++ components."""
        with patch("pynds.pynds.cnds") as mock_cnds:
            with patch("os.path.isfile", return_value=True):
                # Mock the C++ NDS class
                mock_nds = Mock()
                mock_cnds.Nds.return_value = mock_nds

                # Create real PyNDS instance
                pynds = PyNDS("fake_rom.nds")
                pynds._nds = mock_nds
                pynds._closed = False

                return pynds

    def test_save_state_basic(self, mock_pynds_instance):
        """Test basic save_state functionality."""
        pynds = mock_pynds_instance

        # Mock file operations
        with patch("builtins.open", mock_open(read_data=b"fake_state_data")):
            with patch("tempfile.NamedTemporaryFile") as mock_temp:
                mock_temp.return_value.__enter__.return_value.name = "/tmp/test_state"

                state_data = pynds.save_state()

                assert isinstance(state_data, bytes)
                assert state_data == b"fake_state_data"

    def test_load_state_basic(self, mock_pynds_instance):
        """Test basic load_state functionality."""
        pynds = mock_pynds_instance

        with patch("builtins.open", mock_open()):
            with patch("tempfile.NamedTemporaryFile") as mock_temp:
                mock_temp.return_value.__enter__.return_value.name = "/tmp/test_state"

                pynds.load_state(b"fake_state_data")

                # Should not raise any exceptions
                assert True

    def test_load_state_empty_data(self, mock_pynds_instance):
        """Test load_state with empty data raises ValueError."""
        pynds = mock_pynds_instance

        with pytest.raises(ValueError, match="State data cannot be empty"):
            pynds.load_state(b"")

    def test_get_state_size_basic(self, mock_pynds_instance):
        """Test basic get_state_size functionality."""
        pynds = mock_pynds_instance

        # Mock save_state to return specific data
        with patch.object(pynds, "save_state", return_value=b"fake_data_123"):
            size = pynds.get_state_size()
            assert size == 13  # Length of 'fake_data_123'

    def test_validate_state_empty(self, mock_pynds_instance):
        """Test validate_state with empty data returns False."""
        pynds = mock_pynds_instance

        result = pynds.validate_state(b"")
        assert result is False


class TestEnhancedFrameControl:
    """Test enhanced frame control functionality (step, run_seconds, etc.)."""

    @pytest.fixture
    def mock_pynds_instance(self):
        """Create a real PyNDS instance with mocked C++ components."""
        with patch("pynds.pynds.cnds") as mock_cnds:
            with patch("os.path.isfile", return_value=True):
                # Mock the C++ NDS class
                mock_nds = Mock()
                mock_cnds.Nds.return_value = mock_nds

                # Create real PyNDS instance
                pynds = PyNDS("fake_rom.nds")
                pynds._nds = mock_nds
                pynds._closed = False

                return pynds

    def test_step_basic(self, mock_pynds_instance):
        """Test basic step functionality."""
        pynds = mock_pynds_instance

        # Mock get_frame to return test data
        mock_frame_data = np.array([[1, 2], [3, 4]])
        with patch.object(pynds, "get_frame", return_value=mock_frame_data):
            with patch.object(pynds, "tick"):
                result = pynds.step(5)

                assert np.array_equal(result, mock_frame_data)

    def test_get_frame_count_basic(self, mock_pynds_instance):
        """Test basic get_frame_count functionality."""
        pynds = mock_pynds_instance

        # Initialize frame count
        pynds._frame_count = 42

        count = pynds.get_frame_count()
        assert count == 42

    def test_get_frame_count_initialization(self, mock_pynds_instance):
        """Test get_frame_count initializes to zero."""
        pynds = mock_pynds_instance

        # Remove _frame_count if it exists
        if hasattr(pynds, "_frame_count"):
            delattr(pynds, "_frame_count")

        count = pynds.get_frame_count()
        assert count == 0
        assert hasattr(pynds, "_frame_count")

    def test_run_seconds_basic(self, mock_pynds_instance):
        """Test basic run_seconds functionality."""
        pynds = mock_pynds_instance

        with patch.object(pynds, "tick") as mock_tick:
            pynds.run_seconds(2.0)  # 2 seconds at 60 FPS = 120 frames
            mock_tick.assert_called_once_with(120)

    def test_run_seconds_zero(self, mock_pynds_instance):
        """Test run_seconds with zero seconds."""
        pynds = mock_pynds_instance

        with patch.object(pynds, "tick") as mock_tick:
            pynds.run_seconds(0.0)
            mock_tick.assert_not_called()

    def test_run_seconds_negative(self, mock_pynds_instance):
        """Test run_seconds with negative seconds raises ValueError."""
        pynds = mock_pynds_instance

        with pytest.raises(ValueError, match="Seconds cannot be negative"):
            pynds.run_seconds(-1.0)


class TestScreenExportFunctionality:
    """Test screen export functionality (export_frame, export_frames, etc.)."""

    @pytest.fixture
    def mock_pynds_instance(self):
        """Create a real PyNDS instance with mocked C++ components."""
        with patch("pynds.pynds.cnds") as mock_cnds:
            with patch("os.path.isfile", return_value=True):
                # Mock the C++ NDS class
                mock_nds = Mock()
                mock_cnds.Nds.return_value = mock_nds

                # Create real PyNDS instance
                pynds = PyNDS("fake_rom.nds")
                pynds._nds = mock_nds
                pynds._closed = False

                return pynds

    def test_export_frame_basic(self, mock_pynds_instance):
        """Test basic export_frame functionality."""
        pynds = mock_pynds_instance

        # Mock get_frame and PIL Image
        mock_frame_data = np.array([[1, 2], [3, 4]])
        with patch.object(pynds, "get_frame", return_value=mock_frame_data):
            with patch.object(pynds, "_numpy_to_pil_image") as mock_convert:
                mock_image = Mock()
                mock_convert.return_value = mock_image

                with patch("builtins.open", mock_open()):
                    pynds.export_frame("test.png")

                    mock_convert.assert_called_once_with(mock_frame_data)
                    mock_image.save.assert_called_once()

    def test_export_frame_invalid_format(self, mock_pynds_instance):
        """Test export_frame with invalid format raises ValueError."""
        pynds = mock_pynds_instance

        with pytest.raises(ValueError, match="Unsupported format"):
            pynds.export_frame("test.gif", format="gif")

    def test_export_frame_invalid_quality(self, mock_pynds_instance):
        """Test export_frame with invalid quality raises ValueError."""
        pynds = mock_pynds_instance

        with pytest.raises(ValueError, match="Quality must be between 1 and 100"):
            pynds.export_frame("test.jpg", quality=150)

    def test_export_frames_basic(self, mock_pynds_instance):
        """Test basic export_frames functionality."""
        pynds = mock_pynds_instance

        # Mock get_frame and PIL Image
        mock_frame_data = np.array([[1, 2], [3, 4]])
        with patch.object(pynds, "get_frame", return_value=mock_frame_data):
            with patch.object(pynds, "_numpy_to_pil_image") as mock_convert:
                with patch.object(pynds, "tick"):
                    mock_image = Mock()
                    mock_convert.return_value = mock_image

                    with patch("os.makedirs") as mock_makedirs:
                        with patch("builtins.open", mock_open()):
                            pynds.export_frames("test_dir/", count=3)

                            mock_makedirs.assert_called_once_with(
                                "test_dir/", exist_ok=True
                            )
                            assert mock_convert.call_count == 3

    def test_get_frame_as_image_basic(self, mock_pynds_instance):
        """Test basic get_frame_as_image functionality."""
        pynds = mock_pynds_instance

        # Mock get_frame and PIL Image
        mock_frame_data = np.array([[1, 2], [3, 4]])
        with patch.object(pynds, "get_frame", return_value=mock_frame_data):
            with patch.object(pynds, "_numpy_to_pil_image") as mock_convert:
                mock_image = Mock()
                mock_convert.return_value = mock_image

                result = pynds.get_frame_as_image()

                assert result == mock_image
                mock_convert.assert_called_once_with(mock_frame_data)

    def test_numpy_to_pil_image_gba(self, mock_pynds_instance):
        """Test _numpy_to_pil_image with GBA frame."""
        pynds = mock_pynds_instance
        pynds.is_gba = True

        frame_data = np.array([[[255, 0, 0, 255], [0, 255, 0, 255]]])

        with patch("PIL.Image") as mock_image_class:
            mock_image = Mock()
            mock_image_class.fromarray.return_value = mock_image

            result = pynds._numpy_to_pil_image(frame_data)

            assert result == mock_image
            mock_image_class.fromarray.assert_called_once_with(frame_data, "RGBA")

    def test_numpy_to_pil_image_nds(self, mock_pynds_instance):
        """Test _numpy_to_pil_image with NDS dual frame."""
        pynds = mock_pynds_instance
        pynds.is_gba = False

        top_frame = np.array([[[255, 0, 0, 255]]])
        bottom_frame = np.array([[[0, 255, 0, 255]]])
        frame_data = (top_frame, bottom_frame)

        with patch("PIL.Image") as mock_image_class:
            mock_top_img = Mock()
            mock_bottom_img = Mock()
            mock_combined = Mock()

            mock_image_class.fromarray.side_effect = [mock_top_img, mock_bottom_img]
            mock_image_class.new.return_value = mock_combined

            mock_top_img.width = 100
            mock_top_img.height = 50
            mock_bottom_img.width = 100
            mock_bottom_img.height = 50

            result = pynds._numpy_to_pil_image(frame_data)

            assert result == mock_combined
            mock_combined.paste.assert_any_call(mock_top_img, (0, 0))
            mock_combined.paste.assert_any_call(mock_bottom_img, (0, 50))


class TestNewApiHelpers:
    """Tests for newly added helper APIs in 0.0.5-alpha."""

    @pytest.fixture
    def mock_pynds_instance(self):
        """Create a real PyNDS instance with mocked C++ components and config."""
        with patch("pynds.pynds.cnds") as mock_cnds:
            with patch("os.path.isfile", return_value=True):
                # Mock the C++ NDS class and config
                mock_nds = Mock()
                mock_cnds.Nds.return_value = mock_nds
                # Provide a config with screen filters disabled for predictable sizes
                mock_cfg = Mock()
                mock_cfg.get_high_res_3d.return_value = False
                mock_cfg.get_screen_filter.return_value = 0
                with patch("pynds.pynds.config", mock_cfg):
                    pynds = PyNDS("fake_rom.nds")
                    pynds._nds = mock_nds
                    pynds._closed = False
                    return pynds

    def test_run_until_frame_increments_frame_count(self, mock_pynds_instance):
        emu = mock_pynds_instance
        # Ensure fresh state
        if hasattr(emu, "_frame_count"):
            delattr(emu, "_frame_count")

        # Drive two frames; mock perf_counter for stable timing
        with patch("pynds.pynds.time.perf_counter", side_effect=[1000.0, 1000.016]):
            emu.run_until_frame()
            emu.run_until_frame()

        assert getattr(emu, "_frame_count", 0) == 2

    def test_platform_helpers(self, mock_pynds_instance):
        emu = mock_pynds_instance
        assert emu.get_platform() == "nds"
        emu.is_gba = True
        assert emu.platform == "gba"

    def test_set_mute_bridges_to_cnds_config(self):
        with patch("pynds.pynds.cnds") as mock_cnds:
            with patch("os.path.isfile", return_value=True):
                mock_nds = Mock()
                mock_cnds.Nds.return_value = mock_nds
                # Attach a config with set_emulate_audio
                mock_cnds.config.set_emulate_audio = Mock()
                emu = PyNDS("fake_rom.nds")
                emu.set_mute(True)
                emu.set_mute(False)
                # True => emulate_audio False; False => emulate_audio True
                mock_cnds.config.set_emulate_audio.assert_any_call(False)
                mock_cnds.config.set_emulate_audio.assert_any_call(True)

    def test_get_frame_format_dimensions_and_platform(self, mock_pynds_instance):
        emu = mock_pynds_instance
        # NDS expectations: 192x256 h/w with RGBA
        fmt = emu.get_frame_format()
        assert fmt["platform"] == "nds"
        assert fmt["height"] == 192 and fmt["width"] == 256
        assert fmt["channels"] == 4 and fmt["layout"] == "rgba"

        # Switch to GBA and re-check
        emu.is_gba = True
        fmt = emu.get_frame_format()
        assert fmt["platform"] == "gba"
        assert fmt["height"] == 160 and fmt["width"] == 240

    def test_get_fps_and_timing_info(self, mock_pynds_instance):
        emu = mock_pynds_instance
        # Initially no samples
        assert emu.get_fps() is None

        # Two frames ~16ms apart (~62.5 FPS)
        with patch("pynds.pynds.time.perf_counter", side_effect=[1000.0, 1000.016]):
            emu.run_until_frame()
            emu.run_until_frame()

        fps = emu.get_fps()
        assert fps is not None and 55.0 <= fps <= 70.0
        info = emu.get_timing_info()
        assert "fps" in info and info["frames"] >= 2

    def test_autodetect_agb_uppercase(self):
        with patch("pynds.pynds.cnds") as mock_cnds:
            with patch("os.path.isfile", return_value=True):
                mock_nds = Mock()
                mock_cnds.Nds.return_value = mock_nds
                emu = PyNDS("GAME.AGB")
                assert emu.is_gba is True


class TestRoadmap3IntegrationScenarios:
    """Test integration scenarios combining multiple Roadmap 3 features."""

    @pytest.fixture
    def mock_pynds_instance(self):
        """Create a real PyNDS instance with mocked C++ components."""
        with patch("pynds.pynds.cnds") as mock_cnds:
            with patch("os.path.isfile", return_value=True):
                # Mock the C++ NDS class
                mock_nds = Mock()
                mock_cnds.Nds.return_value = mock_nds

                # Create real PyNDS instance
                pynds = PyNDS("fake_rom.nds")
                pynds._nds = mock_nds
                pynds._closed = False

                return pynds

    def test_rl_episode_workflow(self, mock_pynds_instance):
        """Test complete RL episode workflow using all Roadmap 3 features."""
        pynds = mock_pynds_instance

        # Mock all the methods
        with patch.object(
            pynds, "save_state", return_value=b"initial_state"
        ) as mock_save:
            with patch.object(pynds, "load_state") as mock_load:
                with patch.object(
                    pynds, "step", return_value=np.array([[1, 2], [3, 4]])
                ) as mock_step:
                    with patch.object(pynds, "export_frame") as mock_export:
                        with patch.object(pynds, "get_frame_count", return_value=100):
                            # Simulate RL episode
                            initial_state = pynds.save_state()

                            for step in range(10):
                                pynds.step()  # We don't need to store the frame
                                if step % 5 == 0:
                                    pynds.export_frame(f"debug_step_{step}.png")

                            # Restore initial state
                            pynds.load_state(initial_state)

                            # Verify calls
                            assert mock_save.call_count == 1
                            assert mock_step.call_count == 10
                            assert mock_export.call_count == 2  # steps 0 and 5
                            assert mock_load.call_count == 1


class TestInputHelpers:
    """Tests for input convenience helpers and top-level key exports."""

    @pytest.fixture
    def mock_pynds_instance(self):
        """Create a real PyNDS instance with mocked C++ components."""
        with patch("pynds.pynds.cnds") as mock_cnds:
            with patch("os.path.isfile", return_value=True):
                mock_nds = Mock()
                mock_cnds.Nds.return_value = mock_nds
                pynds = PyNDS("fake_rom.nds")
                pynds._nds = mock_nds
                pynds._closed = False
                return pynds

    def test_press_and_release_holds_for_n_frames(self, mock_pynds_instance):
        emu = mock_pynds_instance
        frames = 3
        emu.button.press_and_release("a", frames=frames)
        # Key mapping should route through the underlying backend
        emu._nds.press_key.assert_called_once_with(KEY_MAP["a"])
        emu._nds.release_key.assert_called_once_with(KEY_MAP["a"])
        assert emu._nds.run_until_frame.call_count == frames
        assert emu._nds.get_frame.call_count == frames

    def test_tap_holds_touch_then_releases(self, mock_pynds_instance):
        emu = mock_pynds_instance
        frames = 2
        x, y = 120, 88
        emu.button.tap(x, y, frames=frames)
        emu._nds.set_touch_input.assert_called_once_with(x, y)
        emu._nds.touch_input.assert_called_once()
        emu._nds.release_touch_input.assert_called_once()
        assert emu._nds.run_until_frame.call_count == frames
        assert emu._nds.get_frame.call_count == frames

    def test_top_level_key_exports_and_pathlike(self):
        # Expect canonical keys to include 'a'
        assert "a" in VALID_KEYS
        assert BUTTON_KEYS["a"] == KEY_MAP["a"]

        # PyNDS accepts PathLike
        with patch("pynds.pynds.cnds") as mock_cnds:
            with patch("os.path.isfile", return_value=True):
                mock_nds = Mock()
                mock_cnds.Nds.return_value = mock_nds
                rom = Path("fake_rom.nds")
                emu = PyNDS(rom)
                assert emu._rom_path == str(rom)
                mock_cnds.Nds.assert_called_once_with(str(rom), False)

    def test_set_touch_bounds_validation(self, mock_pynds_instance):
        emu = mock_pynds_instance
        # In-bounds should not raise
        emu.button.set_touch(0, 0)
        emu.button.set_touch(255, 191)
        # Out-of-bounds should raise ValueError
        with pytest.raises(ValueError):
            emu.button.set_touch(-1, 0)
        with pytest.raises(ValueError):
            emu.button.set_touch(0, 192)
        with pytest.raises(ValueError):
            emu.button.set_touch(256, 191)

    def test_set_speed_multiplier_bridging(self, mock_pynds_instance):
        emu = mock_pynds_instance
        # Patch config to expose a specific speed setter
        with patch("pynds.pynds.config") as mock_cfg:
            mock_cfg.set_speed_multiplier = Mock()
            ok = emu.set_speed_multiplier(1.25)
            assert ok is True
            mock_cfg.set_speed_multiplier.assert_called_once_with(1.25)

    def test_batch_frame_export_workflow(self, mock_pynds_instance):
        """Test batch frame export workflow using frame control and export features."""
        pynds = mock_pynds_instance

        with patch.object(pynds, "export_frames") as mock_export_frames:
            with patch.object(pynds, "run_seconds") as mock_run_seconds:
                # Export frames for 2 seconds
                pynds.run_seconds(2.0)
                pynds.export_frames("episode_frames/", count=120, prefix="ep1")

                mock_run_seconds.assert_called_once_with(2.0)
                mock_export_frames.assert_called_once_with(
                    "episode_frames/", count=120, prefix="ep1"
                )

    def test_state_checkpointing_workflow(self, mock_pynds_instance):
        """Test state checkpointing workflow using state management features."""
        pynds = mock_pynds_instance

        with patch.object(
            pynds, "save_state", side_effect=[b"state1", b"state2", b"state3"]
        ) as mock_save:
            with patch.object(pynds, "load_state") as mock_load:
                with patch.object(
                    pynds, "validate_state", return_value=True
                ) as mock_validate:
                    # Save checkpoints
                    checkpoints = []
                    for i in range(3):
                        state = pynds.save_state()
                        if pynds.validate_state(state):
                            checkpoints.append(state)

                    # Restore from checkpoint
                    pynds.load_state(checkpoints[1])

                    assert len(checkpoints) == 3
                    assert mock_save.call_count == 3
                    assert mock_validate.call_count == 3
                    assert mock_load.call_count == 1

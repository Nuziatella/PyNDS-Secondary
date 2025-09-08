"""Tests for enhancement features: State Management, Frame Control, and Screen Export.

This module contains comprehensive tests for the new enhancement functionality
including in-memory state management, enhanced frame control, and screen export
capabilities using real PyNDS instances with mocked C++ components.
"""

from unittest.mock import Mock, mock_open, patch

import numpy as np
import pytest

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

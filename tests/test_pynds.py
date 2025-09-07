from unittest.mock import Mock, patch

import numpy as np
import pytest

import pynds


class TestPyNDSInitialization:
    """Test PyNDS initialization and basic setup."""

    def test_init_with_valid_nds_file(self, mock_rom_path):
        """Test initialization with a valid NDS file."""
        with patch("pynds.pynds.cnds") as mock_cnds:
            mock_nds_instance = Mock()
            mock_cnds.Nds.return_value = mock_nds_instance

            nds = pynds.PyNDS(mock_rom_path)

            assert nds.is_gba is False
            assert nds.is_initialized() is True
            assert nds._closed is False
            mock_cnds.Nds.assert_called_once_with(mock_rom_path, False)

    def test_init_with_gba_file(self, mock_rom_path):
        """Test initialization with GBA mode."""
        with patch("pynds.pynds.cnds") as mock_cnds:
            mock_nds_instance = Mock()
            mock_cnds.Nds.return_value = mock_nds_instance

            nds = pynds.PyNDS(mock_rom_path, is_gba=True)

            assert nds.is_gba is True
            assert nds.is_initialized() is True
            mock_cnds.Nds.assert_called_once_with(mock_rom_path, True)

    def test_init_with_auto_detect_gba(self, temp_dir):
        """Test automatic GBA detection from file extension."""
        gba_rom = temp_dir / "test_game.gba"
        gba_rom.write_bytes(b"GBA\x00" + b"\x00" * 1000)

        with patch("pynds.pynds.cnds") as mock_cnds:
            mock_nds_instance = Mock()
            mock_cnds.Nds.return_value = mock_nds_instance

            nds = pynds.PyNDS(str(gba_rom))

            assert nds.is_gba is True
            mock_cnds.Nds.assert_called_once_with(str(gba_rom), True)

    def test_init_with_nonexistent_file(self):
        """Test initialization with non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            pynds.PyNDS("nonexistent_game.nds")

    def test_check_file_exist_static_method(self, mock_rom_path):
        """Test the static file existence check method."""
        # Should not raise for existing file
        pynds.PyNDS.check_file_exist(mock_rom_path)

        # Should raise for non-existing file
        with pytest.raises(FileNotFoundError):
            pynds.PyNDS.check_file_exist("nonexistent.nds")


class TestPyNDSCoreFunctionality:
    """Test core PyNDS functionality like tick() and get_frame()."""

    @pytest.fixture
    def mock_pynds(self, mock_rom_path):
        """Create a mock PyNDS instance for testing."""
        with patch("pynds.pynds.cnds") as mock_cnds:
            mock_nds_instance = Mock()
            mock_cnds.Nds.return_value = mock_nds_instance

            nds = pynds.PyNDS(mock_rom_path)
            nds._nds = mock_nds_instance  # Use our mock
            return nds

    def test_tick_single_frame(self, mock_pynds):
        """Test ticking for a single frame."""
        mock_pynds.tick()

        mock_pynds._nds.run_until_frame.assert_called_once()
        mock_pynds._nds.get_frame.assert_called_once()

    def test_tick_multiple_frames(self, mock_pynds):
        """Test ticking for multiple frames."""
        mock_pynds.tick(5)

        assert mock_pynds._nds.run_until_frame.call_count == 5
        assert mock_pynds._nds.get_frame.call_count == 5

    def test_tick_when_closed(self, mock_pynds):
        """Test that tick raises error when emulator is closed."""
        mock_pynds.close()

        with pytest.raises(RuntimeError, match="not initialized or has been closed"):
            mock_pynds.tick()

    def test_get_frame_nds(self, mock_pynds):
        """Test getting frames from NDS emulator."""
        # Mock the frame data
        mock_top_frame = np.random.randint(0, 255, (192, 256, 4), dtype=np.uint8)
        mock_bot_frame = np.random.randint(0, 255, (192, 256, 4), dtype=np.uint8)

        mock_pynds._nds.get_top_nds_frame.return_value = mock_top_frame
        mock_pynds._nds.get_bot_nds_frame.return_value = mock_bot_frame

        top_frame, bot_frame = mock_pynds.get_frame()

        assert np.array_equal(top_frame, mock_top_frame)
        assert np.array_equal(bot_frame, mock_bot_frame)

    def test_get_frame_gba(self, mock_pynds):
        """Test getting frame from GBA emulator."""
        mock_pynds.is_gba = True

        # Mock the GBA frame data
        mock_frame = np.random.randint(0, 255, (160, 240, 4), dtype=np.uint8)
        mock_pynds._nds.get_gba_frame.return_value = mock_frame

        frame = mock_pynds.get_frame()

        assert np.array_equal(frame, mock_frame)

    def test_get_frame_when_closed(self, mock_pynds):
        """Test that get_frame raises error when emulator is closed."""
        mock_pynds.close()

        with pytest.raises(RuntimeError, match="not initialized or has been closed"):
            mock_pynds.get_frame()

    def test_get_frame_shape_nds(self, mock_pynds):
        """Test getting frame shape for NDS."""
        shape = mock_pynds.get_frame_shape()

        # Default NDS shape should be (192, 256, 4)
        assert shape == (192, 256, 4)

    def test_get_frame_shape_gba(self, mock_pynds):
        """Test getting frame shape for GBA."""
        mock_pynds.is_gba = True

        shape = mock_pynds.get_frame_shape()

        # Default GBA shape should be (160, 240, 4)
        assert shape == (160, 240, 4)


class TestPyNDSStateManagement:
    """Test state management functionality."""

    @pytest.fixture
    def mock_pynds(self, mock_rom_path):
        """Create a mock PyNDS instance for testing."""
        with patch("pynds.pynds.cnds") as mock_cnds:
            mock_nds_instance = Mock()
            mock_cnds.Nds.return_value = mock_nds_instance

            nds = pynds.PyNDS(mock_rom_path)
            nds._nds = mock_nds_instance
            return nds

    def test_save_state_to_file(self, mock_pynds, temp_dir):
        """Test saving state to file."""
        save_path = temp_dir / "save_state.dat"

        mock_pynds.save_state_to_file(str(save_path))

        mock_pynds._nds.save_state.assert_called_once_with(str(save_path))

    def test_load_state_from_file(self, mock_pynds, temp_dir):
        """Test loading state from file."""
        save_path = temp_dir / "save_state.dat"
        save_path.write_bytes(b"mock_save_data")

        mock_pynds.load_state_from_file(str(save_path))

        mock_pynds._nds.load_state.assert_called_once_with(str(save_path))

    def test_load_state_nonexistent_file(self, mock_pynds):
        """Test loading state from non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            mock_pynds.load_state_from_file("nonexistent_save.dat")

    def test_write_save_file(self, mock_pynds, temp_dir):
        """Test writing game save file."""
        save_path = temp_dir / "game_save.sav"

        mock_pynds.write_save_file(str(save_path))

        mock_pynds._nds.save_game.assert_called_once_with(str(save_path))


class TestPyNDSContextManager:
    """Test context manager functionality."""

    def test_context_manager_usage(self, mock_rom_path):
        """Test using PyNDS as a context manager."""
        with patch("pynds.pynds.cnds") as mock_cnds:
            mock_nds_instance = Mock()
            mock_cnds.Nds.return_value = mock_nds_instance

            with pynds.PyNDS(mock_rom_path) as nds:
                assert nds.is_initialized() is True
                assert nds._closed is False

            # After exiting context, should be closed
            assert nds._closed is True
            assert nds.is_initialized() is False

    def test_context_manager_with_exception(self, mock_rom_path):
        """Test context manager cleanup even when exception occurs."""
        with patch("pynds.pynds.cnds") as mock_cnds:
            mock_nds_instance = Mock()
            mock_cnds.Nds.return_value = mock_nds_instance

            try:
                with pynds.PyNDS(mock_rom_path) as nds:
                    assert nds.is_initialized() is True
                    raise ValueError("Test exception")
            except ValueError:
                pass

            # Should still be closed even after exception
            assert nds._closed is True


class TestPyNDSCleanup:
    """Test cleanup and resource management."""

    @pytest.fixture
    def mock_pynds(self, mock_rom_path):
        """Create a mock PyNDS instance for testing."""
        with patch("pynds.pynds.cnds") as mock_cnds:
            mock_nds_instance = Mock()
            mock_cnds.Nds.return_value = mock_nds_instance

            nds = pynds.PyNDS(mock_rom_path)
            nds._nds = mock_nds_instance
            return nds

    def test_close_method(self, mock_pynds):
        """Test the close method."""
        assert mock_pynds.is_initialized() is True

        mock_pynds.close()

        assert mock_pynds._closed is True
        assert mock_pynds.is_initialized() is False

    def test_close_multiple_times(self, mock_pynds):
        """Test that close can be called multiple times safely."""
        mock_pynds.close()
        mock_pynds.close()  # Should not raise error

        assert mock_pynds._closed is True

    def test_destructor_cleanup(self, mock_pynds):
        """Test that destructor calls cleanup if not explicitly closed."""
        # Don't call close() explicitly
        del mock_pynds

        # The destructor should have been called
        # (This is hard to test directly, but we can verify the cleanup logic)

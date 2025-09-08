from unittest.mock import Mock

import pytest

from pynds.button import Button


class TestButtonInitialization:
    """Test Button class initialization."""

    def test_button_init(self):
        """Test Button initialization."""
        mock_nds = Mock()

        button = Button(mock_nds)

        assert button._nds is mock_nds
        assert button._initialized is True

    def test_button_is_initialized(self):
        """Test button initialization check."""
        mock_nds = Mock()

        button = Button(mock_nds)

        assert button.is_initialized() is True

        # Test when not initialized
        button._initialized = False
        assert button.is_initialized() is False

        # Test when NDS is None
        button._nds = None
        assert button.is_initialized() is False


class TestButtonInput:
    """Test button input functionality."""

    @pytest.fixture
    def mock_button(self):
        """Create a mock Button instance for testing."""
        mock_nds = Mock()
        button = Button(mock_nds)
        return button

    def test_press_key_valid(self, mock_button):
        """Test pressing a valid button."""
        mock_button.press_key("a")

        # Should call the underlying NDS with the mapped key code
        mock_button._nds.press_key.assert_called_once_with(0)  # 'a' maps to 0

    def test_press_key_invalid(self, mock_button):
        """Test pressing an invalid button raises error."""
        with pytest.raises(KeyError):
            mock_button.press_key("invalid_button")

    def test_release_key_valid(self, mock_button):
        """Test releasing a valid button."""
        mock_button.release_key("b")

        mock_button._nds.release_key.assert_called_once_with(1)  # 'b' maps to 1

    def test_release_key_invalid(self, mock_button):
        """Test releasing an invalid button raises error."""
        with pytest.raises(KeyError):
            mock_button.release_key("invalid_button")

    def test_all_valid_keys(self, mock_button):
        """Test that all valid keys work correctly."""
        valid_keys = [
            "a",
            "b",
            "select",
            "start",
            "right",
            "left",
            "up",
            "down",
            "r",
            "l",
            "x",
            "y",
        ]
        expected_codes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        for key, expected_code in zip(valid_keys, expected_codes):
            mock_button.press_key(key)
            mock_button._nds.press_key.assert_called_with(expected_code)

            mock_button.release_key(key)
            mock_button._nds.release_key.assert_called_with(expected_code)


class TestTouchScreenInput:
    """Test touch screen input functionality."""

    @pytest.fixture
    def mock_button(self):
        """Create a mock Button instance for testing."""
        mock_nds = Mock()
        button = Button(mock_nds)
        return button

    def test_set_touch(self, mock_button):
        """Test setting touch coordinates."""
        mock_button.set_touch(100, 150)

        mock_button._nds.set_touch_input.assert_called_once_with(100, 150)

    def test_clear_touch(self, mock_button):
        """Test clearing touch input."""
        mock_button.clear_touch()

        mock_button._nds.clear_touch_input.assert_called_once()

    def test_touch(self, mock_button):
        """Test activating touch input."""
        mock_button.touch()

        mock_button._nds.touch_input.assert_called_once()

    def test_release_touch(self, mock_button):
        """Test releasing touch input."""
        mock_button.release_touch()

        mock_button._nds.release_touch_input.assert_called_once()

    def test_touch_sequence(self, mock_button):
        """Test a complete touch sequence."""
        # Set coordinates
        mock_button.set_touch(50, 75)
        mock_button._nds.set_touch_input.assert_called_with(50, 75)

        # Touch
        mock_button.touch()
        mock_button._nds.touch_input.assert_called_once()

        # Release
        mock_button.release_touch()
        mock_button._nds.release_touch_input.assert_called_once()

        # Clear
        mock_button.clear_touch()
        mock_button._nds.clear_touch_input.assert_called_once()


class TestButtonCleanup:
    """Test Button cleanup functionality."""

    def test_close_method(self):
        """Test the close method."""
        mock_nds = Mock()
        button = Button(mock_nds)

        assert button._initialized is True

        button.close()

        assert button._initialized is False

    def test_close_when_not_initialized(self):
        """Test closing when not initialized."""
        mock_nds = Mock()
        button = Button(mock_nds)
        button._initialized = False

        # Should not raise error
        button.close()

        assert button._initialized is False

    def test_destructor_cleanup(self):
        """Test that destructor calls cleanup if not explicitly closed."""
        mock_nds = Mock()
        button = Button(mock_nds)

        # Don't call close() explicitly
        del button

        # The destructor should have been called
        # (This is hard to test directly, but we can verify the cleanup logic)


class TestButtonKeyMapping:
    """Test the KEY_MAP constant and its usage."""

    def test_key_map_completeness(self):
        """Test that KEY_MAP contains all expected keys."""
        from pynds.button import KEY_MAP

        expected_keys = [
            "a",
            "b",
            "select",
            "start",
            "right",
            "left",
            "up",
            "down",
            "r",
            "l",
            "x",
            "y",
        ]

        for key in expected_keys:
            assert key in KEY_MAP

        # Should have exactly 12 keys
        assert len(KEY_MAP) == 12

    def test_key_map_values(self):
        """Test that KEY_MAP values are sequential integers."""
        from pynds.button import KEY_MAP

        values = list(KEY_MAP.values())
        values.sort()

        # Should be sequential integers starting from 0
        assert values == list(range(12))

    def test_key_map_consistency(self):
        """Test that KEY_MAP is consistent with expected mappings."""
        from pynds.button import KEY_MAP

        expected_mappings = {
            "a": 0,
            "b": 1,
            "select": 2,
            "start": 3,
            "right": 4,
            "left": 5,
            "up": 6,
            "down": 7,
            "r": 8,
            "l": 9,
            "x": 10,
            "y": 11,
        }

        assert KEY_MAP == expected_mappings

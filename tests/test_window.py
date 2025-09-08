from unittest.mock import Mock, patch

import numpy as np
import pytest

from pynds.window import Window


class TestWindowInitialization:
    """Test Window class initialization."""

    def test_window_init(self):
        """Test Window initialization."""
        mock_pynds = Mock()
        mock_pynds.get_is_gba.return_value = False

        window = Window(mock_pynds)

        assert window._pynds is mock_pynds
        assert window.is_gba is False
        assert window.running is False
        assert window._initialized is False
        assert window._pygame_initialized is False

    def test_window_init_gba_mode(self):
        """Test Window initialization in GBA mode."""
        mock_pynds = Mock()
        mock_pynds.get_is_gba.return_value = True

        window = Window(mock_pynds)

        assert window.is_gba is True


class TestWindowLifecycle:
    """Test window creation and cleanup."""

    @pytest.fixture
    def mock_window(self):
        """Create a mock Window instance for testing."""
        mock_pynds = Mock()
        mock_pynds.get_is_gba.return_value = False

        window = Window(mock_pynds)
        return window

    @patch("pynds.window.pygame")
    def test_init_window(self, mock_pygame, mock_window):
        """Test window initialization."""
        mock_surface = Mock()
        mock_pygame.display.set_mode.return_value = mock_surface

        mock_window.init(800, 600)

        assert mock_window.running is True
        assert mock_window._initialized is True
        assert mock_window.screen is mock_surface

        mock_pygame.init.assert_called_once()
        mock_pygame.display.set_mode.assert_called_once_with(
            (800, 600), mock_pygame.RESIZABLE
        )
        mock_pygame.display.set_caption.assert_called_once_with("PyNDS")

    @patch("pynds.window.pygame")
    def test_init_window_already_initialized(self, mock_pygame, mock_window):
        """Test window initialization when pygame is already initialized."""
        mock_surface = Mock()
        mock_pygame.display.set_mode.return_value = mock_surface
        mock_window._pygame_initialized = True

        mock_window.init(800, 600)

        # Should not call pygame.init() again
        mock_pygame.init.assert_not_called()
        mock_pygame.display.set_mode.assert_called_once()

    @patch("pynds.window.pygame")
    def test_close_window(self, mock_pygame, mock_window):
        """Test window closing."""
        mock_window.running = True
        mock_window._pygame_initialized = True

        mock_window.close()

        assert mock_window.running is False
        assert mock_window._initialized is False
        assert mock_window._pygame_initialized is False

        mock_pygame.quit.assert_called_once()

    def test_close_window_when_not_running(self, mock_window):
        """Test closing window when not running."""
        mock_window.running = False

        # Should not raise error
        mock_window.close()

        assert mock_window.running is False

    def test_is_initialized(self, mock_window):
        """Test window initialization check."""
        assert mock_window.is_initialized() is False

        mock_window.running = True
        mock_window._initialized = True

        assert mock_window.is_initialized() is True

        mock_window.running = False
        assert mock_window.is_initialized() is False


class TestWindowRendering:
    """Test window rendering functionality."""

    @pytest.fixture
    def mock_window(self):
        """Create a mock Window instance for testing."""
        mock_pynds = Mock()
        mock_pynds.get_is_gba.return_value = False

        window = Window(mock_pynds)
        window.running = True
        window._initialized = True
        window.screen = Mock()
        return window

    def test_render_when_not_initialized(self):
        """Test rendering when window is not initialized."""
        mock_pynds = Mock()
        window = Window(mock_pynds)

        # Should not raise error, just return early
        window.render()

    @patch("pynds.window.pygame")
    def test_render_nds_frame(self, mock_pygame, mock_window):
        """Test rendering NDS frames."""
        mock_window.is_gba = False

        # Mock the frame data
        mock_top_frame = np.random.randint(0, 255, (192, 256, 4), dtype=np.uint8)
        mock_bot_frame = np.random.randint(0, 255, (192, 256, 4), dtype=np.uint8)

        mock_window._pynds.get_frame.return_value = (mock_top_frame, mock_bot_frame)
        mock_window.screen.get_size.return_value = (800, 600)

        # Mock pygame image creation
        mock_surface_top = Mock()
        mock_surface_bot = Mock()
        mock_pygame.image.frombuffer.side_effect = [mock_surface_top, mock_surface_bot]

        mock_window.process_frame_nds()

        # Should create surfaces for both frames
        assert mock_pygame.image.frombuffer.call_count == 2
        mock_pygame.display.flip.assert_called_once()

    @patch("pynds.window.pygame")
    def test_render_gba_frame(self, mock_pygame, mock_window):
        """Test rendering GBA frame."""
        mock_window.is_gba = True

        # Mock the frame data
        mock_frame = np.random.randint(0, 255, (160, 240, 4), dtype=np.uint8)

        mock_window._pynds.get_frame.return_value = mock_frame
        mock_window.screen.get_size.return_value = (800, 600)

        # Mock pygame image creation
        mock_surface = Mock()
        mock_pygame.image.frombuffer.return_value = mock_surface

        mock_window.process_frame_gba()

        # Should create surface for the frame
        mock_pygame.image.frombuffer.assert_called_once()
        mock_pygame.display.flip.assert_called_once()


class TestWindowInputHandling:
    """Test window input event handling."""

    @pytest.fixture
    def mock_window(self):
        """Create a mock Window instance for testing."""
        mock_pynds = Mock()
        mock_pynds.get_is_gba.return_value = False

        window = Window(mock_pynds)
        window.running = True
        window.screen = Mock()
        return window

    @patch("pynds.window.pygame")
    def test_handle_events_quit(self, mock_pygame, mock_window):
        """Test handling quit event."""
        mock_event = Mock()
        mock_event.type = mock_pygame.QUIT

        mock_pygame.event.get.return_value = [mock_event]

        mock_window.handle_events()

        # Should close the window
        assert mock_window.running is False

    @patch("pynds.window.pygame")
    def test_handle_events_mouse_motion(self, mock_pygame, mock_window):
        """Test handling mouse motion event."""
        mock_event = Mock()
        mock_event.type = mock_pygame.MOUSEMOTION
        mock_event.pos = (100, 400)  # Below middle (y > 300 for 600 height)

        mock_pygame.event.get.return_value = [mock_event]
        mock_window.screen.get_size.return_value = (800, 600)

        mock_window.handle_events()

        # Should set touch coordinates
        mock_window._pynds.button.set_touch.assert_called_once()

    @patch("pynds.window.pygame")
    def test_handle_events_mouse_motion_top_screen(self, mock_pygame, mock_window):
        """Test handling mouse motion on top screen."""
        mock_event = Mock()
        mock_event.type = mock_pygame.MOUSEMOTION
        mock_event.pos = (100, 200)  # Above middle (y < 300 for 600 height)

        mock_pygame.event.get.return_value = [mock_event]
        mock_window.screen.get_size.return_value = (800, 600)

        mock_window.handle_events()

        # Should clear touch
        mock_window._pynds.button.clear_touch.assert_called_once()

    @patch("pynds.window.pygame")
    def test_handle_events_mouse_button_down(self, mock_pygame, mock_window):
        """Test handling mouse button down event."""
        mock_event = Mock()
        mock_event.type = mock_pygame.MOUSEBUTTONDOWN

        mock_pygame.event.get.return_value = [mock_event]

        mock_window.handle_events()

        # Should touch
        mock_window._pynds.button.touch.assert_called_once()

    @patch("pynds.window.pygame")
    def test_handle_events_mouse_button_up(self, mock_pygame, mock_window):
        """Test handling mouse button up event."""
        mock_event = Mock()
        mock_event.type = mock_pygame.MOUSEBUTTONUP

        mock_pygame.event.get.return_value = [mock_event]

        mock_window.handle_events()

        # Should release touch
        mock_window._pynds.button.release_touch.assert_called_once()

    @patch("pynds.window.pygame")
    def test_handle_events_key_down(self, mock_pygame, mock_window):
        """Test handling key down events."""
        mock_event = Mock()
        mock_event.type = mock_pygame.KEYDOWN
        mock_event.key = mock_pygame.K_w  # 'w' key for up

        mock_pygame.event.get.return_value = [mock_event]

        mock_window.handle_events()

        # Should press up key
        mock_window._pynds.button.press_key.assert_called_once_with("up")

    @patch("pynds.window.pygame")
    def test_handle_events_key_up(self, mock_pygame, mock_window):
        """Test handling key up events."""
        mock_event = Mock()
        mock_event.type = mock_pygame.KEYUP
        mock_event.key = mock_pygame.K_w  # 'w' key for up

        mock_pygame.event.get.return_value = [mock_event]

        mock_window.handle_events()

        # Should release up key
        mock_window._pynds.button.release_key.assert_called_once_with("up")


class TestWindowCleanup:
    """Test Window cleanup functionality."""

    def test_destructor_cleanup(self):
        """Test that destructor calls cleanup if not explicitly closed."""
        mock_pynds = Mock()
        window = Window(mock_pynds)
        window.running = True

        # Don't call close() explicitly
        del window

        # The destructor should have been called
        # (This is hard to test directly, but we can verify the cleanup logic)


class TestWindowErrorHandling:
    """Test window error handling."""

    @pytest.fixture
    def mock_window(self):
        """Create a mock Window instance for testing."""
        mock_pynds = Mock()
        mock_pynds.get_is_gba.return_value = False

        window = Window(mock_pynds)
        window.running = True
        window._initialized = True
        window.screen = Mock()
        return window

    @patch("pynds.window.pygame")
    def test_process_frame_gba_error(self, mock_pygame, mock_window):
        """Test error handling in GBA frame processing."""
        mock_window.is_gba = True

        # Make get_frame raise an exception
        mock_window._pynds.get_frame.side_effect = Exception("Frame error")

        # Should not raise exception, just log error
        mock_window.process_frame_gba()

    @patch("pynds.window.pygame")
    def test_process_frame_nds_error(self, mock_pygame, mock_window):
        """Test error handling in NDS frame processing."""
        mock_window.is_gba = False

        # Make get_frame raise an exception
        mock_window._pynds.get_frame.side_effect = Exception("Frame error")

        # Should not raise exception, just log error
        mock_window.process_frame_nds()

    @patch("pynds.window.pygame")
    def test_init_window_error(self, mock_pygame, mock_window):
        """Test error handling in window initialization."""
        mock_pygame.init.side_effect = Exception("Pygame init error")

        with pytest.raises(Exception):
            mock_window.init(800, 600)

        # Should mark as not initialized
        assert mock_window._initialized is False

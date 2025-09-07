"""Window management for PyNDS display output.

This module provides a pygame-based window interface for displaying Nintendo DS
and Game Boy Advance emulation output. It handles both dual-screen NDS display
and single-screen GBA display with proper scaling and input handling.
"""

import logging

import numpy as np
import pygame

# Set up logging for window management
logger = logging.getLogger(__name__)


class Window:
    """Pygame-based window interface for Nintendo DS emulation display.

    Your digital window into the Nintendo DS universe! This class provides a
    graphical window for displaying emulated Nintendo DS or Game Boy Advance
    output, complete with dual-screen NDS layout and single-screen GBA magic.

    Handles automatic scaling and input event processing, so you can watch
    your digital adventures unfold in glorious pixels on your screen.

    Attributes
    ----------
    _pynds : PyNDS
        Reference to the parent PyNDS instance (the emulation wizard)
    is_gba : bool
        Boolean indicating if running in GBA mode (single screen magic)
    running : bool
        Boolean indicating if the window is currently active (is it alive?)
    screen : pygame.Surface
        Pygame surface for rendering (your digital canvas)

    Examples
    --------
    >>> nds = PyNDS("game.nds")
    >>> nds.open_window(800, 600)  # Open your portal to the digital world
    >>> nds.render()  # Watch the magic happen frame by frame
    """

    def __init__(self, pynds) -> None:
        """Initialize Window interface.

        Parameters
        ----------
        pynds : PyNDS
            The parent PyNDS instance
        """
        self._pynds = pynds
        self.is_gba = pynds.get_is_gba()

        self.running = False
        self._initialized = False
        self._pygame_initialized = False

        logger.debug("Window interface initialized")

    def init(self, width: int, height: int) -> None:
        """Initialize the pygame window.

        Creates a resizable pygame window with the specified dimensions.
        It's like opening a portal to another dimension, but with pixels
        instead of interdimensional travel.

        Parameters
        ----------
        width : int
            Window width in pixels (how wide your digital world)
        height : int
            Window height in pixels (how tall your digital world)

        Raises
        ------
        pygame.error
            If pygame initialization fails (portal malfunction)
        """
        try:
            # Initialize pygame if not already done
            if not self._pygame_initialized:
                pygame.init()
                self._pygame_initialized = True
                logger.debug("Pygame initialized")

            # Create the display surface
            self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
            pygame.display.set_caption("PyNDS")

            self.running = True
            self._initialized = True

            logger.info(f"Window initialized: {width}x{height}")

        except Exception as e:
            logger.error(f"Failed to initialize window: {e}")
            self._initialized = False
            raise

    def close(self) -> None:
        """Close the pygame window.

        Shuts down pygame and marks the window as no longer running.
        Safe to call multiple times.
        """
        if not self.running:
            return

        try:
            logger.debug("Closing pygame window...")

            # Clean up pygame resources
            if self._pygame_initialized:
                pygame.quit()
                self._pygame_initialized = False
                logger.debug("Pygame quit")

            self.running = False
            self._initialized = False

            logger.info("Window closed successfully")

        except Exception as e:
            logger.error(f"Error during window cleanup: {e}")
            # Still mark as closed even if cleanup failed
            self.running = False
            self._initialized = False

    def __del__(self) -> None:
        """Destructor to ensure cleanup happens even if close() wasn't called.

        This is a safety net to prevent memory leaks if the user forgets to
        call close() explicitly.
        """
        if self.running:
            logger.warning(
                "Window was not properly closed! Calling cleanup in destructor."
            )
            self.close()

    def is_initialized(self) -> bool:
        """Check if the window is properly initialized and running.

        Returns
        -------
        bool
            True if window is initialized and running, False otherwise
        """
        return self._initialized and self.running

    def render(self) -> None:
        """Render the current emulation frame to the window.

        The moment of truth! Processes input events and renders the appropriate
        frame (GBA or NDS) to the pygame window. It's like painting a masterpiece
        one frame at a time, except the masterpiece is a video game.

        Only renders if the window is currently running (because dead windows
        can't display anything, obviously).
        """
        if not self.is_initialized():
            logger.warning("Attempted to render on uninitialized window")
            return

        if self.running:
            self.handle_events()

            if self.is_gba and self.running:
                self.process_frame_gba()
            elif self.running:
                self.process_frame_nds()

    def handle_events(self) -> None:
        """Handle pygame input events.

        Processes keyboard and mouse input events, translating them to
        appropriate Nintendo DS button presses and touch screen interactions.
        Handles window close events and updates touch coordinates.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

            if event.type == pygame.MOUSEMOTION:
                width, height = self.screen.get_size()
                mid_height = height // 2
                x, y = event.pos
                if y > mid_height:
                    scale_y = mid_height / 192
                    y -= mid_height
                    y = int(y / scale_y)

                    scale_x = width / 256
                    x = int(x / scale_x)
                    self._pynds.button.set_touch(x, y)
                else:
                    self._pynds.button.clear_touch()

            if event.type == pygame.MOUSEBUTTONDOWN:
                self._pynds.button.touch()

            if event.type == pygame.MOUSEBUTTONUP:
                self._pynds.button.release_touch()

            if event.type == pygame.KEYDOWN:  # Check for key press events
                if event.key == pygame.K_w:
                    self._pynds.button.press_key("up")
                if event.key == pygame.K_d:
                    self._pynds.button.press_key("right")
                if event.key == pygame.K_a:
                    self._pynds.button.press_key("left")
                if event.key == pygame.K_s:
                    self._pynds.button.press_key("down")
                if event.key == pygame.K_p:
                    self._pynds.button.press_key("start")
                if event.key == pygame.K_o:
                    self._pynds.button.press_key("select")
                if event.key == pygame.K_l:
                    self._pynds.button.press_key("a")
                if event.key == pygame.K_k:
                    self._pynds.button.press_key("b")
                if event.key == pygame.K_i:
                    self._pynds.button.press_key("x")
                if event.key == pygame.K_j:
                    self._pynds.button.press_key("y")
                if event.key == pygame.K_e:
                    self._pynds.button.press_key("l")
                if event.key == pygame.K_u:
                    self._pynds.button.press_key("r")

            if event.type == pygame.KEYUP:  # Check for key release events
                if event.key == pygame.K_w:
                    self._pynds.button.release_key("up")
                if event.key == pygame.K_d:
                    self._pynds.button.release_key("right")
                if event.key == pygame.K_a:
                    self._pynds.button.release_key("left")
                if event.key == pygame.K_s:
                    self._pynds.button.release_key("down")
                if event.key == pygame.K_p:
                    self._pynds.button.release_key("start")
                if event.key == pygame.K_o:
                    self._pynds.button.release_key("select")
                if event.key == pygame.K_l:
                    self._pynds.button.release_key("a")
                if event.key == pygame.K_k:
                    self._pynds.button.release_key("b")
                if event.key == pygame.K_i:
                    self._pynds.button.release_key("x")
                if event.key == pygame.K_j:
                    self._pynds.button.release_key("y")
                if event.key == pygame.K_e:
                    self._pynds.button.release_key("l")
                if event.key == pygame.K_u:
                    self._pynds.button.release_key("r")

    def process_frame_gba(self) -> None:
        """Process and display a Game Boy Advance frame.

        Retrieves the current GBA frame from the emulator, reshapes it for
        pygame display, and renders it scaled to fit the window dimensions.
        The GBA frame is displayed as a single screen.
        """
        try:
            width, height = self.screen.get_size()

            frame = self._pynds.get_frame()
            # Ensure we have a contiguous array to prevent memory issues
            frame = np.ascontiguousarray(frame)
            frame = frame.reshape((frame.shape[1], frame.shape[0], 4))

            # Create surface from frame data
            surface = pygame.image.frombuffer(
                frame, (frame.shape[0], frame.shape[1]), "RGBA"
            )
            surface = pygame.transform.scale(surface, (width, height))

            self.screen.blit(surface, surface.get_rect(topleft=(0, 0)))
            pygame.display.flip()

        except Exception as e:
            logger.error(f"Error processing GBA frame: {e}")
            # Don't re-raise to prevent crashing the emulation loop

    def process_frame_nds(self) -> None:
        """Process and display Nintendo DS dual-screen frames.

        The Nintendo DS's signature move - dual screens! Retrieves both top
        and bottom screen frames from the emulator and displays them in all
        their dual-screen glory. It's like having two windows into the same
        digital universe.

        The top screen gets the upper half (prime real estate), while the
        bottom screen gets the lower half (still pretty good real estate).
        """
        try:
            width, height = self.screen.get_size()

            frame_top, frame_bot = self._pynds.get_frame()

            # Ensure contiguous arrays to prevent memory issues
            frame_top = np.ascontiguousarray(frame_top)
            frame_top = frame_top.reshape((frame_top.shape[1], frame_top.shape[0], 4))

            frame_bot = np.ascontiguousarray(frame_bot)
            frame_bot = frame_bot.reshape((frame_bot.shape[1], frame_bot.shape[0], 4))

            # Process top screen
            surface_top = pygame.image.frombuffer(
                frame_top, (frame_top.shape[0], frame_top.shape[1]), "RGBA"
            )
            surface_top = pygame.transform.scale(surface_top, (width, height // 2))
            self.screen.blit(surface_top, surface_top.get_rect(topleft=(0, 0)))

            # Process bottom screen
            surface_bot = pygame.image.frombuffer(
                frame_bot, (frame_bot.shape[0], frame_bot.shape[1]), "RGBA"
            )
            surface_bot = pygame.transform.scale(surface_bot, (width, height // 2))
            self.screen.blit(
                surface_bot, surface_bot.get_rect(topleft=(0, height // 2))
            )

            pygame.display.flip()

        except Exception as e:
            logger.error(f"Error processing NDS frame: {e}")
            # Don't re-raise to prevent crashing the emulation loop

"""Button input mapping for Nintendo DS controls.

Maps human-readable button names to their corresponding internal key codes
used by the NooDS emulator. This mapping follows the standard Nintendo DS
button layout.
"""

import logging

# Set up logging for button management
logger = logging.getLogger(__name__)

KEY_MAP = {
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


class Button:
    """Button input interface for Nintendo DS emulation.

    Your digital finger puppet for the Nintendo DS. This class provides methods
    to simulate button presses, releases, and touch screen interactions,
    letting you become the invisible hand guiding the game's destiny.

    All input operations are performed on the underlying C++ emulator instance,
    so your button mashing is as real as it gets in the digital realm.

    Attributes
    ----------
    _nds : object
        The underlying C++ NDS emulator instance (your digital puppet master)

    Examples
    --------
    >>> nds = PyNDS("game.nds")
    >>> button = nds.button
    >>> button.press_key('a')  # Press the magic A button
    >>> button.set_touch(100, 150)  # Point at the screen
    >>> button.touch()  # Make contact with the digital world
    """

    def __init__(self, nds) -> None:
        """Initialize Button interface.

        Parameters
        ----------
        nds : object
            The underlying C++ NDS emulator instance
        """
        self._nds = nds
        self._initialized = True

        logger.debug("Button interface initialized")

    def set_touch(self, x: int, y: int) -> None:
        """Set touch screen coordinates.

        Positions your digital finger on the touch screen without actually
        touching it yet. It's like hovering your finger over the screen,
        building suspense for the inevitable tap.

        Parameters
        ----------
        x : int
            X coordinate on the touch screen (0-255) - horizontal position
        y : int
            Y coordinate on the touch screen (0-191) - vertical position

        Raises
        ------
        RuntimeError
            If emulator not initialized (no screen to touch)
        ValueError
            If coordinates are outside valid range (finger slipped off screen)
        """
        # Enforce DS touch bounds (inclusive) to match docstring semantics.
        if not (0 <= x <= 255 and 0 <= y <= 191):
            raise ValueError(
                "touch coordinates out of range: x in [0,255], y in [0,191]"
            )
        self._nds.set_touch_input(x, y)

    def clear_touch(self) -> None:
        """Clear touch screen input.

        Removes any active touch input from the touch screen.

        Raises
        ------
        RuntimeError
            If emulator not initialized
        """
        self._nds.clear_touch_input()

    def touch(self) -> None:
        """Activate touch screen input.

        Makes contact with the digital world! Registers a touch input at the
        previously set coordinates. It's the moment your digital finger
        finally touches the screen and magic happens.

        Raises
        ------
        RuntimeError
            If emulator not initialized (no digital world to touch)
        """
        self._nds.touch_input()

    def release_touch(self) -> None:
        """Release touch screen input.

        Deactivates any active touch input on the touch screen.

        Raises
        ------
        RuntimeError
            If emulator not initialized
        """
        self._nds.release_touch_input()

    def press_key(self, key: str) -> None:
        """Press a button on the Nintendo DS.

        Simulates pressing a button on the Nintendo DS. It's like having
        invisible fingers that can press any button instantly. The game
        will never know it wasn't a real human (or will it?).

        Parameters
        ----------
        key : str
            Button name ('a', 'b', 'select', 'start', 'right', 'left',
            'up', 'down', 'r', 'l', 'x', 'y') - your digital button collection

        Raises
        ------
        RuntimeError
            If emulator not initialized (no puppet master)
        KeyError
            If key name is not valid (button doesn't exist in this reality)
        """
        self._nds.press_key(KEY_MAP[key])

    def release_key(self, key: str) -> None:
        """Release a button on the Nintendo DS.

        Parameters
        ----------
        key : str
            Button name ('a', 'b', 'select', 'start', 'right', 'left',
            'up', 'down', 'r', 'l', 'x', 'y')

        Raises
        ------
        RuntimeError
            If emulator not initialized
        KeyError
            If key name is not valid
        """
        self._nds.release_key(KEY_MAP[key])

    def press_and_release(self, key: str, frames: int = 1) -> None:
        """Press a button, hold for N frames, then release.

        A small ergonomic helper for common input patterns in agents/tests.

        Parameters
        ----------
        key : str
            One of the standard DS buttons, e.g., 'a', 'b', 'start', 'up', etc.
        frames : int
            Number of frames to hold before releasing (minimum 1).
        """
        if frames < 1:
            frames = 1
        self.press_key(key)
        try:
            for _ in range(frames):
                # Advance one frame to actually register and hold the press
                self._nds.run_until_frame()
                # Pull a frame to keep parity with the main loop
                try:
                    self._nds.get_frame()
                except Exception as e:  # nosec B110: optional fetch may be unsupported
                    # Some backends fetch via platform-specific getters; not fatal.
                    logger.debug("optional get_frame() failed during press: %s", e)
        finally:
            self.release_key(key)

    def tap(self, x: int, y: int, frames: int = 1) -> None:
        """Touch the screen at (x, y), hold for N frames, then release.

        Parameters
        ----------
        x : int
            X coordinate on the touch screen (0-255).
        y : int
            Y coordinate on the touch screen (0-191).
        frames : int
            Number of frames to hold before releasing (minimum 1).
        """
        if frames < 1:
            frames = 1
        self.set_touch(x, y)
        self.touch()
        try:
            for _ in range(frames):
                self._nds.run_until_frame()
                try:
                    self._nds.get_frame()
                except Exception as e:  # nosec B110
                    logger.debug("optional get_frame() failed during tap: %s", e)
        finally:
            self.release_touch()

    def is_initialized(self) -> bool:
        """Check if the button interface is properly initialized.

        Returns
        -------
        bool
            True if button interface is initialized and ready to use, False otherwise
        """
        return self._initialized and self._nds is not None

    def close(self) -> None:
        """Close the button interface and clean up resources.

        Properly shuts down the button interface. This is mainly for consistency
        with other classes - the actual cleanup is handled by the parent PyNDS class.
        """
        if not self._initialized:
            return

        try:
            logger.debug("Closing button interface...")
            self._initialized = False
            logger.debug("Button interface closed")

        except Exception as e:
            logger.error(f"Error during button cleanup: {e}")
            self._initialized = False

    def __del__(self) -> None:
        """Destructor to ensure cleanup happens even if close() wasn't called.

        This is a safety net to prevent memory leaks if the user forgets to
        call close() explicitly.
        """
        if self._initialized:
            logger.warning(
                "Button interface was not properly closed! Calling cleanup in destructor."
            )
            self.close()

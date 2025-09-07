"""Core PyNDS emulator class for Nintendo DS and Game Boy Advance emulation.

The heart and soul of your digital time machine! This module contains the main
PyNDS class that brings your favorite games to life in Python. Load ROMs, control
emulation, grab frame data, and coordinate all the magic that makes PyNDS tick.

Classes:
    PyNDS: Your main emulator class for Nintendo DS and Game Boy Advance adventures
"""

import logging
import os
from typing import Tuple
from typing import Tuple as TupleType
from typing import Union

import numpy as np

import cnds

from .button import Button
from .config import config
from .memory import Memory
from .window import Window

# Set up logging for memory management
logger = logging.getLogger(__name__)


class PyNDS:
    """Main interface for Nintendo DS emulation in Python.

    Welcome to the digital time machine! PyNDS provides a Python wrapper around
    the NooDS Nintendo DS emulator, giving you the power to control Nintendo DS
    and Game Boy Advance games programmatically. Perfect for reinforcement learning,
    bot development, and automated testing (or just having fun with digital nostalgia).

    The emulator supports both Nintendo DS (.nds) and Game Boy Advance (.gba)
    ROM files with automatic format detection. It's like having a Nintendo DS
    that lives inside your Python code and does whatever you tell it to do.

    Attributes
    ----------
    is_gba : bool
        Boolean indicating if running in Game Boy Advance mode (single screen magic)
    button : Button
        Button input interface for game controls (your digital fingers)
    memory : Memory
        Memory access interface for reading/writing game state (digital mind reading)
    window : Window
        Window interface for graphical display (your portal to the digital world)

    Examples
    --------
    >>> import pynds
    >>> nds = pynds.PyNDS("game.nds")  # Load your digital adventure
    >>> nds.tick()  # Run emulation for one frame (watch the magic happen)
    >>> top_frame, bottom_frame = nds.get_frame()  # Capture both screens
    >>> nds.button.press_key('a')  # Press the magic A button
    >>> nds.memory.write_ram_u32(0x02000000, 0x12345678)  # Rewrite reality
    """

    def __init__(
        self, path: str, auto_detect: bool = True, is_gba: bool = False
    ) -> None:
        """Initialize PyNDS emulator with a ROM file.

        The moment of digital birth! Loads a ROM file and brings it to life
        in your Python environment. It's like opening a digital time capsule
        and watching history unfold in real-time.

        Parameters
        ----------
        path : str
            Path to the ROM file (.nds or .gba) - your digital treasure map
        auto_detect : bool, optional
            Whether to automatically detect ROM format from file extension (smart mode), by default True
        is_gba : bool, optional
            Force Game Boy Advance mode (overrides auto_detect) - single screen mode, by default False

        Raises
        ------
        FileNotFoundError
            If the ROM file does not exist (treasure not found)
        RuntimeError
            If emulator initialization fails (digital resurrection failed)
        ValueError
            If ROM file format is invalid (corrupted digital artifact)

        Examples
        --------
        >>> nds = pynds.PyNDS("pokemon.nds")  # Auto-detect NDS (gotta catch 'em all!)
        >>> gba = pynds.PyNDS("game.gba")    # Auto-detect GBA (retro magic)
        >>> nds = pynds.PyNDS("rom.bin", auto_detect=False, is_gba=True)  # Force GBA mode
        """
        if auto_detect:
            is_gba = path.endswith(".gba") or is_gba

        self.is_gba = is_gba

        self.check_file_exist(path)
        self._nds = cnds.Nds(path, is_gba)

        self.button = Button(self._nds)
        self.memory = Memory(self._nds)
        self.window = Window(self)

        # Memory management state
        self._initialized = True
        self._closed = False

        logger.info(f"PyNDS initialized with ROM: {path} (GBA: {is_gba})")

    def get_is_gba(self) -> bool:
        """Check if emulator is running in Game Boy Advance mode.

        Returns
        -------
        bool
            True if running GBA ROM, False if running NDS ROM
        """
        return self.is_gba

    def tick(self, count: int = 1) -> None:
        """Run the emulator for the specified number of frames.

        The heartbeat of your digital world! Advances the emulation by running
        until the specified number of frames have been generated. Each tick
        brings your game one step closer to its digital destiny.

        Parameters
        ----------
        count : int, optional
            Number of frames to advance (how many heartbeats), by default 1

        Raises
        ------
        RuntimeError
            If emulator is not properly initialized (no heartbeat)
        ValueError
            If emulator has been closed

        Examples
        --------
        >>> nds.tick()      # Advance one frame (one heartbeat)
        >>> nds.tick(60)     # Advance 60 frames (1 second at 60fps - time travel!)
        """
        if not self.is_initialized():
            raise RuntimeError("Emulator is not initialized or has been closed")

        for i in range(count):
            self._nds.run_until_frame()
            self._nds.get_frame()

    def get_frame(self) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Get the current emulation frame(s).

        Capture the digital moment! Returns the most recently rendered frame(s)
        from the emulator, frozen in time as numpy arrays. It's like taking
        a screenshot of your digital world.

        For Nintendo DS, you get both screens (double the fun!). For Game Boy
        Advance, you get a single screen (still plenty of fun!).

        Returns
        -------
        Union[Tuple[np.ndarray, np.ndarray], np.ndarray]
            For NDS: Tuple of (top_screen, bottom_screen) as numpy arrays (dual screen magic)
            For GBA: Single numpy array representing the screen (single screen magic)

        Raises
        ------
        RuntimeError
            If emulator is not properly initialized (no digital world to capture)

        Examples
        --------
        >>> # Nintendo DS (dual screen adventure)
        >>> top, bottom = nds.get_frame()
        >>> print(f"Top screen shape: {top.shape}")  # The main event
        >>>
        >>> # Game Boy Advance (single screen nostalgia)
        >>> frame = gba.get_frame()
        >>> print(f"Screen shape: {frame.shape}")  # Pure retro magic
        """
        if not self.is_initialized():
            raise RuntimeError("Emulator is not initialized or has been closed")

        if self.is_gba:
            frame = self._nds.get_gba_frame()
            return frame
        else:
            top_frame = self._nds.get_top_nds_frame()
            bot_frame = self._nds.get_bot_nds_frame()
            return (top_frame, bot_frame)

    def get_frame_shape(self) -> TupleType[int, int, int]:
        """Get the shape of frames returned by get_frame().

        Returns the dimensions of the frame arrays, accounting for any
        scaling applied by the emulator configuration.

        Returns
        -------
        TupleType[int, int, int]
            Tuple of (height, width, channels) where channels is always 4 (RGBA)

        Examples
        --------
        >>> shape = nds.get_frame_shape()
        >>> print(f"Frame dimensions: {shape}")  # (384, 512, 4) for scaled NDS
        """
        NDS = (192, 256)
        GBA = (160, 240)
        if config.get_high_res_3d() or config.get_screen_filter() == 1:
            scale = 2
        else:
            scale = 1

        if self.is_gba:
            return (GBA[0] * scale, GBA[1] * scale, 4)
        else:
            return (NDS[0] * scale, NDS[1] * scale, 4)

    def open_window(self, width: int = 800, height: int = 800) -> None:
        """Open a pygame window for visual display.

        Open your portal to the digital world! Creates a resizable pygame window
        for displaying the emulation output. It's like opening a window into
        another dimension, but with pixels instead of interdimensional travel.

        If a window is already open, it will be closed first (because you
        can't have two portals to the same dimension, obviously).

        Parameters
        ----------
        width : int, optional
            Window width in pixels (how wide your digital world), by default 800
        height : int, optional
            Window height in pixels (how tall your digital world), by default 800

        Raises
        ------
        pygame.error
            If pygame initialization fails (portal malfunction)

        Examples
        --------
        >>> nds.open_window(1024, 768)  # Custom size (bigger portal!)
        >>> nds.open_window()           # Default 800x800 (standard portal)
        """
        if self.window.running:
            self.window.close()

        self.window.init(width, height)

    def close_window(self) -> None:
        """Close the pygame display window.

        Closes the pygame window if it is currently open. Safe to call
        multiple times or when no window is open.
        """
        self.window.close()

    def render(self) -> None:
        """Render the current frame to the display window.

        Updates the pygame window with the current emulation frame.
        Only renders if a window is currently open.

        Examples
        --------
        >>> nds.open_window()
        >>> nds.tick()
        >>> nds.render()  # Display the frame
        """
        if self.window.running:
            self.window.render()

    def save_state_to_file(self, path: str) -> None:
        """Save the current emulation state to a file.

        Creates a save state file containing the complete emulation state,
        allowing the game to be resumed from this exact point later.

        Parameters
        ----------
        path : str
            File path where to save the state

        Raises
        ------
        RuntimeError
            If emulator is not properly initialized
        OSError
            If the file cannot be written

        Examples
        --------
        >>> nds.save_state_to_file("save_state.dat")
        """
        self._nds.save_state(path)

    def load_state_from_file(self, path: str) -> None:
        """Load an emulation state from a file.

        Restores the emulation state from a previously saved state file,
        allowing the game to resume from that exact point.

        Parameters
        ----------
        path : str
            File path to the save state file

        Raises
        ------
        FileNotFoundError
            If the save state file does not exist
        RuntimeError
            If emulator is not properly initialized or state is invalid

        Examples
        --------
        >>> nds.load_state_from_file("save_state.dat")
        """
        self.check_file_exist(path)
        self._nds.load_state(path)

    def write_save_file(self, path: str) -> None:
        """Write the game's save data to a file.

        Saves the in-game save data (not emulation state) to a file.
        This is equivalent to saving the game from within the emulated system.

        Parameters
        ----------
        path : str
            File path where to save the game data

        Raises
        ------
        RuntimeError
            If emulator is not properly initialized
        OSError
            If the file cannot be written

        Examples
        --------
        >>> nds.write_save_file("game_save.sav")
        """
        self._nds.save_game(path)

    def close(self) -> None:
        """Close the emulator and clean up all resources.

        Properly shuts down the emulator, closes any open windows, and cleans up
        all C++ resources. This method should be called when you're done with
        the emulator to prevent memory leaks.

        Safe to call multiple times - subsequent calls will be ignored.

        Examples
        --------
        >>> nds = pynds.PyNDS("game.nds")
        >>> # ... use the emulator ...
        >>> nds.close()  # Clean up when done
        """
        if self._closed:
            return

        try:
            logger.info("Closing PyNDS emulator and cleaning up resources...")

            # Close the window first to clean up pygame resources
            if hasattr(self, "window") and self.window:
                self.window.close()

            # Clean up C++ resources
            if hasattr(self, "_nds") and self._nds:
                # Note: The C++ destructor should handle cleanup automatically
                # but we'll set it to None to help with garbage collection
                self._nds = None

            # Mark as closed
            self._closed = True
            self._initialized = False

            logger.info("PyNDS emulator closed successfully")

        except Exception as e:
            logger.error(f"Error during PyNDS cleanup: {e}")
            # Still mark as closed even if cleanup failed
            self._closed = True
            self._initialized = False

    def __del__(self) -> None:
        """Destructor to ensure cleanup happens even if close() wasn't called.

        This is a safety net to prevent memory leaks if the user forgets to
        call close() explicitly. However, relying on __del__ is not recommended
        - always call close() explicitly when you're done with the emulator.
        """
        if hasattr(self, "_closed") and not self._closed:
            logger.warning(
                "PyNDS was not properly closed! Calling cleanup in destructor."
            )
            self.close()

    def __enter__(self) -> "PyNDS":
        """Context manager entry - returns self for use in 'with' statements.

        Returns
        -------
        PyNDS
            Self for use in context manager

        Examples
        --------
        >>> with pynds.PyNDS("game.nds") as nds:
        ...     nds.tick()
        ...     frame = nds.get_frame()
        >>> # Automatically cleaned up when exiting the 'with' block
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - automatically cleans up resources.

        Parameters
        ----------
        exc_type : type, optional
            Exception type if an exception occurred
        exc_val : Exception, optional
            Exception value if an exception occurred
        exc_tb : traceback, optional
            Exception traceback if an exception occurred
        """
        self.close()

    def is_initialized(self) -> bool:
        """Check if the emulator is properly initialized and not closed.

        Returns
        -------
        bool
            True if emulator is initialized and ready to use, False otherwise
        """
        return self._initialized and not self._closed

    @staticmethod
    def check_file_exist(path: str) -> None:
        """Check if a file exists and raise an error if it doesn't.

        Utility method to validate file existence before attempting to use it.

        Parameters
        ----------
        path : str
            File path to check

        Raises
        ------
        FileNotFoundError
            If the file does not exist

        Examples
        --------
        >>> PyNDS.check_file_exist("game.nds")
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File '{path}' does not exist.")

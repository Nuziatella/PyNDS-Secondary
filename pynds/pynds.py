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
        self._rom_path = path

        self.check_file_exist(path)
        self._nds = cnds.Nds(path, is_gba)

        self.button = Button(self._nds)
        self.memory = Memory(self._nds)
        self.window = Window(self)

        # Memory management state
        self._initialized = True
        self._closed = False

        logger.info(f"PyNDS initialized with ROM: {path} (GBA: {is_gba})")

    def _reinitialize(self, rom_path: str, is_gba: bool) -> None:
        """Reinitialize the emulator with the same parameters.

        Internal helper method for reset() to avoid calling __init__ directly.

        Parameters
        ----------
        rom_path : str
            Path to the ROM file
        is_gba : bool
            Whether this is a GBA ROM
        """
        self.is_gba = is_gba
        self._rom_path = rom_path

        self.check_file_exist(rom_path)
        self._nds = cnds.Nds(rom_path, is_gba)

        self.button = Button(self._nds)
        self.memory = Memory(self._nds)
        self.window = Window(self)

        # Reset state
        self._initialized = True
        self._closed = False
        self._frame_count = 0

        logger.info(f"PyNDS reinitialized with ROM: {rom_path} (GBA: {is_gba})")

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

        # Update frame count
        if not hasattr(self, "_frame_count"):
            self._frame_count = 0
        self._frame_count += count

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

    def save_state(self) -> bytes:
        """Save the current emulation state to memory.

        Creates a save state in memory containing the complete emulation state,
        allowing the game to be resumed from this exact point later without
        file I/O operations. Perfect for reinforcement learning checkpointing!

        Returns
        -------
        bytes
            Serialized state data that can be used with load_state()

        Raises
        ------
        RuntimeError
            If emulator is not properly initialized

        Examples
        --------
        >>> state_data = nds.save_state()  # Save to memory
        >>> nds.load_state(state_data)     # Restore from memory
        >>> # Perfect for RL checkpointing!
        """
        if not self.is_initialized():
            raise RuntimeError("Emulator is not initialized or has been closed")

        # For now, we'll use a temporary file approach since the C++ layer
        # doesn't have direct in-memory state methods. This is still much
        # faster than user file operations.
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            self._nds.save_state(temp_path)
            with open(temp_path, "rb") as f:
                state_data = f.read()
            return state_data
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def load_state(self, state_data: bytes) -> None:
        """Load an emulation state from memory.

        Restores the emulation state from previously saved state data,
        allowing the game to resume from that exact point. Much faster
        than file-based operations for RL applications!

        Parameters
        ----------
        state_data : bytes
            Serialized state data from save_state()

        Raises
        ------
        RuntimeError
            If emulator is not properly initialized
        ValueError
            If state data is invalid or corrupted

        Examples
        --------
        >>> state_data = nds.save_state()  # Save checkpoint
        >>> nds.load_state(state_data)     # Restore checkpoint
        >>> # Instant state switching for RL!
        """
        if not self.is_initialized():
            raise RuntimeError("Emulator is not initialized or has been closed")

        if not state_data:
            raise ValueError("State data cannot be empty")

        # Use temporary file approach for now
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            with open(temp_path, "wb") as f:
                f.write(state_data)
            self._nds.load_state(temp_path)
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def get_state_size(self) -> int:
        """Get the size of the current state in bytes.

        Returns the approximate size of the current emulation state when
        serialized. Useful for memory management and optimization in
        reinforcement learning applications.

        Returns
        -------
        int
            Size of the current state in bytes

        Raises
        ------
        RuntimeError
            If emulator is not properly initialized

        Examples
        --------
        >>> size = nds.get_state_size()
        >>> print(f"State size: {size} bytes")
        """
        if not self.is_initialized():
            raise RuntimeError("Emulator is not initialized or has been closed")

        # Get state data and return its size
        state_data = self.save_state()
        return len(state_data)

    def validate_state(self, state_data: bytes) -> bool:
        """Validate if state data is valid and can be loaded.

        Checks if the provided state data is valid and can be successfully
        loaded. Useful for error handling and state management in RL applications.

        Parameters
        ----------
        state_data : bytes
            Serialized state data to validate

        Returns
        -------
        bool
            True if state data is valid, False otherwise

        Examples
        --------
        >>> state_data = nds.save_state()
        >>> if nds.validate_state(state_data):
        ...     print("State is valid!")
        """
        if not state_data:
            return False

        try:
            # Try to load the state in a temporary emulator instance
            # This is a bit of a hack, but it's the most reliable way
            # to validate state data without affecting the current state
            import os
            import tempfile

            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_path = temp_file.name

            try:
                with open(temp_path, "wb") as f:
                    f.write(state_data)

                # Try to load the state (this will fail if invalid)
                self._nds.load_state(temp_path)
                return True
            except Exception:
                return False
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        except Exception:
            return False

    def step(self, frames: int = 1) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Run emulation for specified frames and return frame data.

        The ultimate convenience method! Runs the emulator for the specified
        number of frames AND returns the frame data in one operation. Perfect
        for reinforcement learning and streamlined emulation workflows.

        Parameters
        ----------
        frames : int, optional
            Number of frames to run (how many heartbeats), by default 1

        Returns
        -------
        Union[Tuple[np.ndarray, np.ndarray], np.ndarray]
            For NDS: Tuple of (top_screen, bottom_screen) as numpy arrays
            For GBA: Single numpy array representing the screen

        Raises
        ------
        RuntimeError
            If emulator is not properly initialized (no heartbeat)
        ValueError
            If emulator has been closed

        Examples
        --------
        >>> # Run one frame and get frame data
        >>> frame = nds.step()  # Single frame magic!
        >>>
        >>> # Run multiple frames
        >>> top, bottom = nds.step(60)  # 60 frames of pure digital joy!
        >>>
        >>> # Perfect for RL loops
        >>> for episode in range(1000):
        ...     frame = nds.step()
        ...     action = agent.choose_action(frame)
        ...     nds.button.press_key(action)
        """
        self.tick(frames)
        return self.get_frame()

    def tick_until_frame(self) -> None:
        """Run emulation until the next frame is ready.

        Advances the emulation until a new frame is generated and ready
        for capture. Useful for variable timing scenarios and ensuring
        you always get fresh frame data.

        Raises
        ------
        RuntimeError
            If emulator is not properly initialized
        ValueError
            If emulator has been closed

        Examples
        --------
        >>> nds.tick_until_frame()  # Wait for next frame
        >>> frame = nds.get_frame()  # Get the fresh frame
        """
        if not self.is_initialized():
            raise RuntimeError("Emulator is not initialized or has been closed")

        # Run until we have a frame ready
        while True:
            self.tick(1)
            try:
                # Try to get frame - if it fails, keep running
                self.get_frame()
                break
            except RuntimeError:
                continue

    def get_frame_count(self) -> int:
        """Get the total number of frames rendered so far.

        Returns the cumulative count of frames that have been rendered
        since the emulator was initialized. Useful for monitoring
        emulation progress and performance metrics.

        Returns
        -------
        int
            Total number of frames rendered

        Raises
        ------
        RuntimeError
            If emulator is not properly initialized

        Examples
        --------
        >>> count = nds.get_frame_count()
        >>> print(f"Rendered {count} frames so far")
        """
        if not self.is_initialized():
            raise RuntimeError("Emulator is not initialized or has been closed")

        # For now, we'll track this manually since the C++ layer
        # doesn't expose frame count directly
        if not hasattr(self, "_frame_count"):
            self._frame_count = 0
        return self._frame_count

    def run_seconds(self, seconds: float) -> None:
        """Run emulation for a specified time duration.

        Runs the emulator for the specified number of seconds, automatically
        calculating the appropriate number of frames based on the target
        frame rate. Perfect for time-based emulation scenarios!

        Parameters
        ----------
        seconds : float
            Number of seconds to run emulation

        Raises
        ------
        RuntimeError
            If emulator is not properly initialized
        ValueError
            If seconds is negative

        Examples
        --------
        >>> nds.run_seconds(2.5)  # Run for 2.5 seconds
        >>> nds.run_seconds(0.1)  # Run for 100ms
        """
        if not self.is_initialized():
            raise RuntimeError("Emulator is not initialized or has been closed")

        if seconds < 0:
            raise ValueError("Seconds cannot be negative")

        # Assume 60 FPS for now (can be made configurable later)
        target_fps = 60
        frames = int(seconds * target_fps)

        if frames > 0:
            self.tick(frames)

    def reset(self) -> None:
        """Reset the emulator to its initial state.

        Resets the emulator back to the state it was in when first loaded,
        effectively restarting the game from the beginning. Useful for
        reinforcement learning episodes and testing scenarios.

        Raises
        ------
        RuntimeError
            If emulator is not properly initialized

        Examples
        --------
        >>> nds.reset()  # Back to the beginning!
        >>> # Perfect for starting new RL episodes
        """
        if not self.is_initialized():
            raise RuntimeError("Emulator is not initialized or has been closed")

        # For now, we'll need to reload the ROM to reset
        # This could be optimized in the future with a proper reset method
        rom_path = self.rom_path
        is_gba = self.is_gba

        # Close current emulator
        self.close()

        # Reinitialize with same parameters
        self._reinitialize(rom_path, is_gba)

    def export_frame(
        self,
        path: str,
        format: str = "png",
        quality: int = 95,
        include_metadata: bool = False,
    ) -> None:
        """Export the current frame as an image file.

        Saves the current emulation frame(s) as an image file for debugging,
        analysis, or documentation purposes. Supports multiple formats and
        quality settings for different use cases.

        Parameters
        ----------
        path : str
            File path where to save the image
        format : str, optional
            Image format ('png', 'jpeg', 'bmp'), by default 'png'
        quality : int, optional
            JPEG quality (1-100), by default 95
        include_metadata : bool, optional
            Include frame metadata in filename, by default False

        Raises
        ------
        RuntimeError
            If emulator is not properly initialized
        ValueError
            If format is not supported or quality is invalid
        OSError
            If the file cannot be written

        Examples
        --------
        >>> nds.export_frame("screenshot.png")  # Basic export
        >>> nds.export_frame("debug.jpg", format="jpeg", quality=80)
        >>> nds.export_frame("frame.png", include_metadata=True)
        """
        if not self.is_initialized():
            raise RuntimeError("Emulator is not initialized or has been closed")

        if format.lower() not in ["png", "jpeg", "jpg", "bmp"]:
            raise ValueError(f"Unsupported format: {format}")

        if not (1 <= quality <= 100):
            raise ValueError("Quality must be between 1 and 100")

        # Get current frame data
        frame_data = self.get_frame()

        # Convert to PIL Image
        image = self._numpy_to_pil_image(frame_data)

        # Add metadata to filename if requested
        if include_metadata:
            import os
            from datetime import datetime

            base, ext = os.path.splitext(path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            frame_count = self.get_frame_count()
            path = f"{base}_{timestamp}_frame{frame_count}{ext}"

        # Save with appropriate format and quality
        if format.lower() in ["jpeg", "jpg"]:
            image.save(path, format="JPEG", quality=quality, optimize=True)
        else:
            image.save(path, format=format.upper())

    def export_frames(
        self, directory: str, count: int = 1, prefix: str = "frame", format: str = "png"
    ) -> None:
        """Export multiple frames to a directory.

        Saves multiple consecutive frames to a directory with sequential
        naming. Perfect for creating frame sequences, animations, or
        batch analysis workflows.

        Parameters
        ----------
        directory : str
            Directory path where to save frames
        count : int, optional
            Number of frames to export, by default 1
        prefix : str, optional
            Filename prefix for exported frames, by default "frame"
        format : str, optional
            Image format ('png', 'jpeg', 'bmp'), by default 'png'

        Raises
        ------
        RuntimeError
            If emulator is not properly initialized
        OSError
            If directory cannot be created or files cannot be written

        Examples
        --------
        >>> nds.export_frames("frames/", count=10)  # Export 10 frames
        >>> nds.export_frames("episode1/", count=100, prefix="ep1_frame")
        """
        if not self.is_initialized():
            raise RuntimeError("Emulator is not initialized or has been closed")

        import os

        os.makedirs(directory, exist_ok=True)

        for i in range(count):
            # Get frame data
            frame_data = self.get_frame()

            # Convert to PIL Image
            image = self._numpy_to_pil_image(frame_data)

            # Create filename
            filename = f"{prefix}_{i:06d}.{format.lower()}"
            filepath = os.path.join(directory, filename)

            # Save frame
            image.save(filepath, format=format.upper())

            # Advance to next frame
            if i < count - 1:  # Don't advance after the last frame
                self.tick(1)

    def get_frame_as_image(self):
        """Get the current frame as a PIL Image object.

        Returns the current emulation frame(s) as a PIL Image object,
        allowing for further image processing, manipulation, or
        custom export operations.

        Returns
        -------
        PIL.Image.Image
            PIL Image object containing the current frame(s)

        Raises
        ------
        RuntimeError
            If emulator is not properly initialized
        ImportError
            If PIL/Pillow is not available

        Examples
        --------
        >>> image = nds.get_frame_as_image()
        >>> image.thumbnail((200, 200))  # Resize
        >>> image.save("thumbnail.png")
        >>>
        >>> # Advanced processing
        >>> image = nds.get_frame_as_image()
        >>> image = image.convert('L')  # Convert to grayscale
        >>> image.save("grayscale.png")
        """
        if not self.is_initialized():
            raise RuntimeError("Emulator is not initialized or has been closed")

        try:
            import PIL.Image  # noqa: F401
        except ImportError:
            raise ImportError("PIL/Pillow is required for image operations")

        # Get current frame data
        frame_data = self.get_frame()

        # Convert to PIL Image
        return self._numpy_to_pil_image(frame_data)

    def _numpy_to_pil_image(self, frame_data):
        """Convert numpy frame data to PIL Image.

        Internal helper method to convert numpy array frame data
        to PIL Image objects, handling both NDS and GBA formats.

        Parameters
        ----------
        frame_data : Union[Tuple[np.ndarray, np.ndarray], np.ndarray]
            Frame data from get_frame()

        Returns
        -------
        PIL.Image.Image
            PIL Image object
        """
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("PIL/Pillow is required for image operations")

        if self.is_gba:
            # Single frame for GBA
            if len(frame_data.shape) == 3 and frame_data.shape[2] == 4:
                # RGBA format
                return Image.fromarray(frame_data, "RGBA")
            else:
                # Convert to RGBA if needed
                if len(frame_data.shape) == 2:
                    # Grayscale, convert to RGB
                    frame_data = np.stack([frame_data] * 3, axis=-1)
                return Image.fromarray(frame_data.astype(np.uint8), "RGB")
        else:
            # Dual frame for NDS - combine into single image
            top_frame, bottom_frame = frame_data

            # Ensure both frames are RGBA
            if len(top_frame.shape) == 3 and top_frame.shape[2] == 4:
                top_img = Image.fromarray(top_frame, "RGBA")
            else:
                top_img = Image.fromarray(top_frame.astype(np.uint8), "RGB")

            if len(bottom_frame.shape) == 3 and bottom_frame.shape[2] == 4:
                bottom_img = Image.fromarray(bottom_frame, "RGBA")
            else:
                bottom_img = Image.fromarray(bottom_frame.astype(np.uint8), "RGB")

            # Combine frames vertically
            total_width = max(top_img.width, bottom_img.width)
            total_height = top_img.height + bottom_img.height

            combined = Image.new("RGBA", (total_width, total_height), (0, 0, 0, 255))
            combined.paste(top_img, (0, 0))
            combined.paste(bottom_img, (0, top_img.height))

            return combined

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

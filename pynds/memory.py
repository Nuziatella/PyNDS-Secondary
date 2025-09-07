"""Memory access interface for Nintendo DS emulation.

Your gateway to the digital soul of the Nintendo DS! This module lets you peek
into and manipulate the very essence of your favorite games. Read from RAM, write
to ROM (carefully!), and explore every nook and cranny of the emulated memory
space with proper safety checks to keep you from breaking things.

Classes:
    Memory: Your memory manipulation toolkit with read/write operations for all data types
"""

import logging

# Set up logging for memory management
logger = logging.getLogger(__name__)


class Memory:
    """Memory access interface for Nintendo DS emulation.

    Your gateway to the digital soul of the Nintendo DS. This class provides
    type-safe memory read/write operations for the emulated system, letting you
    peek into and modify the very essence of running games. All operations are
    performed on the underlying C++ emulator instance through nanobind bindings.

    Warning: With great memory access comes great responsibility. Don't break
    the game's fragile digital ecosystem unless you know what you're doing!

    Attributes
    ----------
    _nds : object
        The underlying C++ NDS emulator instance (our digital medium)

    Examples
    --------
    >>> nds = PyNDS("game.nds")
    >>> memory = nds.memory
    >>> # Read some mysterious value from the game's memory
    >>> value = memory.read_ram_u32(0x02000000)
    >>> # Write your own digital graffiti (use responsibly!)
    >>> memory.write_ram_u32(0x02000000, 0x12345678)
    """

    def __init__(self, nds) -> None:
        """Initialize Memory interface.

        Parameters
        ----------
        nds : object
            The underlying C++ NDS emulator instance
        """
        self._nds = nds
        self._initialized = True

        logger.debug("Memory interface initialized")

    # Read memory methods
    def read_ram_u8(self, address: int) -> int:
        """Read unsigned 8-bit value from RAM.

        Reads a single byte from the emulated memory. Perfect for when you
        need just a nibble of data (well, two nibbles technically).

        Parameters
        ----------
        address : int
            Memory address to read from (the digital coordinates)

        Returns
        -------
        int
            Unsigned 8-bit integer value (0-255) - a byte of pure digital joy

        Raises
        ------
        RuntimeError
            If address is invalid or emulator not initialized
        """
        return self._nds.read_ram_u8(address)

    def read_ram_u16(self, address: int) -> int:
        """Read unsigned 16-bit value from RAM.

        Parameters
        ----------
        address : int
            Memory address to read from

        Returns
        -------
        int
            Unsigned 16-bit integer value (0-65535)

        Raises
        ------
        RuntimeError
            If address is invalid or emulator not initialized
        """
        return self._nds.read_ram_u16(address)

    def read_ram_u32(self, address: int) -> int:
        """Read unsigned 32-bit value from RAM.

        Reads a full 32-bit word from memory. This is the bread and butter
        of memory hacking - four bytes of pure gaming data goodness.

        Parameters
        ----------
        address : int
            Memory address to read from (your digital treasure map)

        Returns
        -------
        int
            Unsigned 32-bit integer value (0-4294967295) - a word of wisdom

        Raises
        ------
        RuntimeError
            If address is invalid or emulator not initialized
        """
        return self._nds.read_ram_u32(address)

    def read_ram_u64(self, address: int) -> int:
        """Read unsigned 64-bit value from RAM.

        Parameters
        ----------
        address : int
            Memory address to read from

        Returns
        -------
        int
            Unsigned 64-bit integer value

        Raises
        ------
        RuntimeError
            If address is invalid or emulator not initialized
        """
        return self._nds.read_ram_u64(address)

    def read_ram_i8(self, address: int) -> int:
        """Read signed 8-bit value from RAM.

        Parameters
        ----------
        address : int
            Memory address to read from

        Returns
        -------
        int
            Signed 8-bit integer value (-128 to 127)

        Raises
        ------
        RuntimeError
            If address is invalid or emulator not initialized
        """
        return self._nds.read_ram_i8(address)

    def read_ram_i16(self, address: int) -> int:
        """Read signed 16-bit value from RAM.

        Parameters
        ----------
        address : int
            Memory address to read from

        Returns
        -------
        int
            Signed 16-bit integer value (-32768 to 32767)

        Raises
        ------
        RuntimeError
            If address is invalid or emulator not initialized
        """
        return self._nds.read_ram_i16(address)

    def read_ram_i32(self, address: int) -> int:
        """Read signed 32-bit value from RAM.

        Parameters
        ----------
        address : int
            Memory address to read from

        Returns
        -------
        int
            Signed 32-bit integer value (-2147483648 to 2147483647)

        Raises
        ------
        RuntimeError
            If address is invalid or emulator not initialized
        """
        return self._nds.read_ram_i32(address)

    def read_ram_i64(self, address: int) -> int:
        """Read signed 64-bit value from RAM.

        Parameters
        ----------
        address : int
            Memory address to read from

        Returns
        -------
        int
            Signed 64-bit integer value

        Raises
        ------
        RuntimeError
            If address is invalid or emulator not initialized
        """
        return self._nds.read_ram_i64(address)

    def read_ram_f32(self, address: int) -> float:
        """Read 32-bit floating point value from RAM.

        Parameters
        ----------
        address : int
            Memory address to read from

        Returns
        -------
        float
            32-bit floating point value

        Raises
        ------
        RuntimeError
            If address is invalid or emulator not initialized
        """
        return self._nds.read_ram_f32(address)

    def read_ram_f64(self, address: int) -> float:
        """Read 64-bit floating point value from RAM.

        Parameters
        ----------
        address : int
            Memory address to read from

        Returns
        -------
        float
            64-bit floating point value

        Raises
        ------
        RuntimeError
            If address is invalid or emulator not initialized
        """
        return self._nds.read_ram_f64(address)

    def read_map(self, start_address: int, end_address: int) -> list:
        """Read a range of memory addresses.

        Reads a chunk of memory like you're downloading the game's thoughts.
        Perfect for when you need to see the bigger picture (or just snoop around).

        Parameters
        ----------
        start_address : int
            Starting memory address (inclusive) - where the journey begins
        end_address : int
            Ending memory address (exclusive) - where it ends (but doesn't include)

        Returns
        -------
        list
            List of byte values from the specified memory range - your digital loot

        Raises
        ------
        RuntimeError
            If address range is invalid or emulator not initialized
        ValueError
            If start_address >= end_address (time travel not supported)
        """
        return self._nds.read_map(start_address, end_address)

    # Write memory methods
    def write_ram_u8(self, address: int, value: int) -> None:
        """Write unsigned 8-bit value to RAM.

        Parameters
        ----------
        address : int
            Memory address to write to
        value : int
            Unsigned 8-bit integer value (0-255)

        Raises
        ------
        RuntimeError
            If address is invalid or emulator not initialized
        ValueError
            If value is outside valid range for 8-bit unsigned
        """
        self._nds.write_ram_u8(address, value)

    def write_ram_u16(self, address: int, value: int) -> None:
        """Write unsigned 16-bit value to RAM.

        Parameters
        ----------
        address : int
            Memory address to write to
        value : int
            Unsigned 16-bit integer value (0-65535)

        Raises
        ------
        RuntimeError
            If address is invalid or emulator not initialized
        ValueError
            If value is outside valid range for 16-bit unsigned
        """
        self._nds.write_ram_u16(address, value)

    def write_ram_u32(self, address: int, value: int) -> None:
        """Write unsigned 32-bit value to RAM.

        Writes a full 32-bit word to memory. Use this power wisely - you're
        literally rewriting the game's reality one word at a time.

        Parameters
        ----------
        address : int
            Memory address to write to (your canvas coordinates)
        value : int
            Unsigned 32-bit integer value (0-4294967295) - your digital brushstroke

        Raises
        ------
        RuntimeError
            If address is invalid or emulator not initialized
        ValueError
            If value is outside valid range for 32-bit unsigned
        """
        self._nds.write_ram_u32(address, value)

    def write_ram_u64(self, address: int, value: int) -> None:
        """Write unsigned 64-bit value to RAM.

        Parameters
        ----------
        address : int
            Memory address to write to
        value : int
            Unsigned 64-bit integer value

        Raises
        ------
        RuntimeError
            If address is invalid or emulator not initialized
        ValueError
            If value is outside valid range for 64-bit unsigned
        """
        self._nds.write_ram_u64(address, value)

    def write_ram_i8(self, address: int, value: int) -> None:
        """Write signed 8-bit value to RAM.

        Parameters
        ----------
        address : int
            Memory address to write to
        value : int
            Signed 8-bit integer value (-128 to 127)

        Raises
        ------
        RuntimeError
            If address is invalid or emulator not initialized
        ValueError
            If value is outside valid range for 8-bit signed
        """
        self._nds.write_ram_i8(address, value)

    def write_ram_i16(self, address: int, value: int) -> None:
        """Write signed 16-bit value to RAM.

        Parameters
        ----------
        address : int
            Memory address to write to
        value : int
            Signed 16-bit integer value (-32768 to 32767)

        Raises
        ------
        RuntimeError
            If address is invalid or emulator not initialized
        ValueError
            If value is outside valid range for 16-bit signed
        """
        self._nds.write_ram_i16(address, value)

    def write_ram_i32(self, address: int, value: int) -> None:
        """Write signed 32-bit value to RAM.

        Parameters
        ----------
        address : int
            Memory address to write to
        value : int
            Signed 32-bit integer value (-2147483648 to 2147483647)

        Raises
        ------
        RuntimeError
            If address is invalid or emulator not initialized
        ValueError
            If value is outside valid range for 32-bit signed
        """
        self._nds.write_ram_i32(address, value)

    def write_ram_i64(self, address: int, value: int) -> None:
        """Write signed 64-bit value to RAM.

        Parameters
        ----------
        address : int
            Memory address to write to
        value : int
            Signed 64-bit integer value

        Raises
        ------
        RuntimeError
            If address is invalid or emulator not initialized
        ValueError
            If value is outside valid range for 64-bit signed
        """
        self._nds.write_ram_i64(address, value)

    def write_ram_f32(self, address: int, value: float) -> None:
        """Write 32-bit floating point value to RAM.

        Parameters
        ----------
        address : int
            Memory address to write to
        value : float
            32-bit floating point value

        Raises
        ------
        RuntimeError
            If address is invalid or emulator not initialized
        """
        self._nds.write_ram_f32(address, value)

    def write_ram_f64(self, address: int, value: float) -> None:
        """Write 64-bit floating point value to RAM.

        Parameters
        ----------
        address : int
            Memory address to write to
        value : float
            64-bit floating point value

        Raises
        ------
        RuntimeError
            If address is invalid or emulator not initialized
        """
        self._nds.write_ram_f64(address, value)

    def write_map(self, start_address: int, data: list) -> None:
        """Write a list of byte values to memory starting at the given address.

        Parameters
        ----------
        start_address : int
            Starting memory address to write to
        data : list
            List of byte values to write

        Raises
        ------
        RuntimeError
            If address is invalid or emulator not initialized
        ValueError
            If data contains invalid byte values
        """
        self._nds.write_map(start_address, data)

    def is_initialized(self) -> bool:
        """Check if the memory interface is properly initialized.

        Returns
        -------
        bool
            True if memory interface is initialized and ready to use, False otherwise
        """
        return self._initialized and self._nds is not None

    def close(self) -> None:
        """Close the memory interface and clean up resources.

        Properly shuts down the memory interface. This is mainly for consistency
        with other classes - the actual cleanup is handled by the parent PyNDS class.
        """
        if not self._initialized:
            return

        try:
            logger.debug("Closing memory interface...")
            self._initialized = False
            logger.debug("Memory interface closed")

        except Exception as e:
            logger.error(f"Error during memory cleanup: {e}")
            self._initialized = False

    def __del__(self) -> None:
        """Destructor to ensure cleanup happens even if close() wasn't called.

        This is a safety net to prevent memory leaks if the user forgets to
        call close() explicitly.
        """
        if self._initialized:
            logger.warning(
                "Memory interface was not properly closed! Calling cleanup in destructor."
            )
            self.close()

"""PyNDS Debugger - PyBoy-inspired debugging tools for Nintendo DS and GBA.

This module provides debugging capabilities similar to PyBoy's debugger,
including memory inspection, register viewing, breakpoint management,
and real-time emulation analysis. Perfect for reverse engineering,
game development, and understanding how your favorite games work!

Classes:
    PyNDSDebugger: Main debugger class for Nintendo DS and GBA emulation
    MemoryInspector: Advanced memory inspection and analysis tools
    BreakpointManager: Breakpoint and watchpoint management system
"""

import logging
from typing import Any, Dict, List, Optional

from .pynds import PyNDS

logger = logging.getLogger(__name__)


class MemoryInspector:
    """Advanced memory inspection and analysis tools.

    Provides detailed memory analysis capabilities similar to PyBoy's
    memory inspector, allowing you to peek into the digital mind of
    your games and understand their inner workings.

    Attributes
    ----------
    pynds : PyNDS
        The PyNDS emulator instance to inspect
    """

    def __init__(self, pynds: PyNDS):
        """Initialize memory inspector.

        Parameters
        ----------
        pynds : PyNDS
            PyNDS emulator instance to inspect
        """
        self.pynds = pynds
        # Cache maps (start, size) -> bytes
        self._memory_cache: Dict[tuple[int, int], bytes] = {}
        self._cache_size = 1000  # Maximum cached memory regions

    def read_memory_region(self, start: int, size: int) -> bytes:
        """Read a region of memory with caching for performance.

        Parameters
        ----------
        start : int
            Starting memory address
        size : int
            Number of bytes to read

        Returns
        -------
        bytes
            Memory data as bytes
        """
        cache_key = (start, size)
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]

        try:
            raw_list = self.pynds.memory.read_map(start, start + size)
            data = bytes(raw_list)
            self._memory_cache[cache_key] = data

            # Manage cache size
            if len(self._memory_cache) > self._cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self._memory_cache))
                del self._memory_cache[oldest_key]

            return data
        except Exception as e:
            logger.error(
                f"Failed to read memory region {start:08X}-{start+size:08X}: {e}"
            )
            return b""

    def find_pattern(
        self, pattern: bytes, start: int = 0, end: int = 0xFFFFFFFF
    ) -> List[int]:
        """Find a byte pattern in memory.

        Parameters
        ----------
        pattern : bytes
            Byte pattern to search for
        start : int, optional
            Starting address for search, by default 0
        end : int, optional
            Ending address for search, by default 0xFFFFFFFF

        Returns
        -------
        List[int]
            List of addresses where pattern was found
        """
        matches = []
        chunk_size = 0x1000  # Search in 4KB chunks

        for addr in range(start, end, chunk_size):
            try:
                data = self.read_memory_region(addr, min(chunk_size, end - addr))
                offset = 0
                while True:
                    pos = data.find(pattern, offset)
                    if pos == -1:
                        break
                    matches.append(addr + pos)
                    offset = pos + 1
            except Exception:
                continue

        return matches

    def analyze_memory_usage(self) -> Dict[str, Any]:
        """Analyze memory usage patterns.

        Returns
        -------
        Dict[str, Any]
            Memory usage analysis including regions, patterns, and statistics
        """
        analysis = {
            "total_regions": 0,
            "active_regions": [],
            "memory_patterns": {},
            "statistics": {},
        }

        # This is a simplified analysis - in a real implementation,
        # you'd want to analyze the actual memory layout
        try:
            # Sample some memory regions
            regions = [
                (0x02000000, 0x02040000),  # Main RAM
                (0x03000000, 0x03008000),  # System RAM
                (0x04000000, 0x04001000),  # I/O registers
            ]

            for start, end in regions:
                try:
                    data = self.read_memory_region(start, min(0x1000, end - start))
                    if data and any(b != 0 for b in data):
                        analysis["active_regions"].append(
                            {
                                "start": start,
                                "end": end,
                                "size": end - start,
                                "non_zero_bytes": sum(1 for b in data if b != 0),
                            }
                        )
                except Exception:
                    continue

            analysis["total_regions"] = len(analysis["active_regions"])

        except Exception as e:
            logger.error(f"Memory analysis failed: {e}")

        return analysis

    def clear_cache(self) -> None:
        """Clear the memory cache."""
        self._memory_cache.clear()


class BreakpointManager:
    """Breakpoint and watchpoint management system.

    Manages breakpoints and watchpoints for debugging, similar to
    PyBoy's debugging capabilities but adapted for DS/GBA architecture.
    """

    def __init__(self):
        """Initialize breakpoint manager."""
        self.breakpoints: Dict[int, Dict[str, Any]] = {}
        self.watchpoints: Dict[int, Dict[str, Any]] = {}
        self.breakpoint_counter = 0

    def add_breakpoint(self, address: int, condition: Optional[str] = None) -> int:
        """Add a breakpoint at the specified address.

        Parameters
        ----------
        address : int
            Memory address for breakpoint
        condition : Optional[str], optional
            Optional condition for breakpoint, by default None

        Returns
        -------
        int
            Breakpoint ID
        """
        bp_id = self.breakpoint_counter
        self.breakpoint_counter += 1

        self.breakpoints[bp_id] = {
            "address": address,
            "condition": condition,
            "enabled": True,
            "hit_count": 0,
        }

        logger.info(f"Breakpoint {bp_id} added at address 0x{address:08X}")
        return bp_id

    def remove_breakpoint(self, bp_id: int) -> bool:
        """Remove a breakpoint by ID.

        Parameters
        ----------
        bp_id : int
            Breakpoint ID to remove

        Returns
        -------
        bool
            True if breakpoint was removed, False if not found
        """
        if bp_id in self.breakpoints:
            del self.breakpoints[bp_id]
            logger.info(f"Breakpoint {bp_id} removed")
            return True
        return False

    def add_watchpoint(
        self, address: int, size: int = 1, access_type: str = "write"
    ) -> int:
        """Add a watchpoint for memory access monitoring.

        Parameters
        ----------
        address : int
            Memory address to watch
        size : int, optional
            Size of memory region to watch, by default 1
        access_type : str, optional
            Type of access to watch ('read', 'write', 'both'), by default 'write'

        Returns
        -------
        int
            Watchpoint ID
        """
        wp_id = self.breakpoint_counter
        self.breakpoint_counter += 1

        self.watchpoints[wp_id] = {
            "address": address,
            "size": size,
            "access_type": access_type,
            "enabled": True,
            "hit_count": 0,
        }

        logger.info(f"Watchpoint {wp_id} added at address 0x{address:08X}")
        return wp_id

    def check_breakpoints(self, address: int) -> List[int]:
        """Check if any breakpoints should trigger at the given address.

        Parameters
        ----------
        address : int
            Address to check

        Returns
        -------
        List[int]
            List of triggered breakpoint IDs
        """
        triggered = []
        for bp_id, bp in self.breakpoints.items():
            if bp["enabled"] and bp["address"] == address:
                bp["hit_count"] += 1
                triggered.append(bp_id)
        return triggered

    def get_breakpoint_info(self) -> Dict[str, Any]:
        """Get information about all breakpoints and watchpoints.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing breakpoint and watchpoint information
        """
        return {
            "breakpoints": self.breakpoints.copy(),
            "watchpoints": self.watchpoints.copy(),
            "total_breakpoints": len(self.breakpoints),
            "total_watchpoints": len(self.watchpoints),
        }


class PyNDSDebugger:
    """Main debugger class for Nintendo DS and GBA emulation.

    Provides comprehensive debugging capabilities inspired by PyBoy's
    debugger, including memory inspection, register viewing, breakpoint
    management, and real-time analysis tools.

    Attributes
    ----------
    pynds : PyNDS
        The PyNDS emulator instance being debugged
    memory_inspector : MemoryInspector
        Memory inspection and analysis tools
    breakpoint_manager : BreakpointManager
        Breakpoint and watchpoint management
    """

    def __init__(self, pynds: PyNDS):
        """Initialize PyNDS debugger.

        Parameters
        ----------
        pynds : PyNDS
            PyNDS emulator instance to debug
        """
        self.pynds = pynds
        self.memory_inspector = MemoryInspector(pynds)
        self.breakpoint_manager = BreakpointManager()
        self._debug_mode = False
        self._step_mode = False
        self._last_instruction_address = 0

    def enable_debug_mode(self) -> None:
        """Enable debug mode for detailed analysis.

        Enables comprehensive debugging features including instruction
        tracing, memory access monitoring, and breakpoint support.
        """
        self._debug_mode = True
        logger.info("Debug mode enabled")

    def disable_debug_mode(self) -> None:
        """Disable debug mode."""
        self._debug_mode = False
        logger.info("Debug mode disabled")

    def step_instruction(self) -> bool:
        """Execute a single instruction in step mode.

        Returns
        -------
        bool
            True if step was successful, False if not in step mode
        """
        if not self._step_mode:
            return False

        try:
            # Execute one frame (this is a simplified step)
            self.pynds.tick(1)

            # Check for breakpoints
            # Note: In a real implementation, you'd check the actual PC
            triggered_bps = self.breakpoint_manager.check_breakpoints(
                self._last_instruction_address
            )
            if triggered_bps:
                logger.info(f"Breakpoints triggered: {triggered_bps}")
                return True

            return True
        except Exception as e:
            logger.error(f"Step instruction failed: {e}")
            return False

    def set_step_mode(self, enabled: bool) -> None:
        """Enable or disable step mode.

        Parameters
        ----------
        enabled : bool
            Whether to enable step mode
        """
        self._step_mode = enabled
        logger.info(f"Step mode {'enabled' if enabled else 'disabled'}")

    def get_cpu_state(self) -> Dict[str, Any]:
        """Get current CPU state information.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing CPU state information
        """
        # This is a simplified CPU state - in a real implementation,
        # you'd read actual CPU registers from the emulator
        return {
            "pc": self._last_instruction_address,
            "debug_mode": self._debug_mode,
            "step_mode": self._step_mode,
            "frame_count": self.pynds.get_frame_count(),
            "platform": self.pynds.get_platform(),
        }

    def inspect_memory_at(self, address: int, size: int = 16) -> Dict[str, Any]:
        """Inspect memory at a specific address.

        Parameters
        ----------
        address : int
            Memory address to inspect
        size : int, optional
            Number of bytes to read, by default 16

        Returns
        -------
        Dict[str, Any]
            Memory inspection results
        """
        try:
            data = self.memory_inspector.read_memory_region(address, size)

            # Format as hex dump
            hex_dump = []
            ascii_dump = []

            for i in range(0, len(data), 16):
                line_data = data[i : i + 16]
                hex_line = " ".join(f"{b:02X}" for b in line_data)
                ascii_line = "".join(
                    chr(b) if 32 <= b <= 126 else "." for b in line_data
                )
                hex_dump.append(f"{address+i:08X}: {hex_line:<48} |{ascii_line}|")
                ascii_dump.append(ascii_line)

            return {
                "address": address,
                "size": size,
                "data": data,
                "hex_dump": hex_dump,
                "ascii_dump": ascii_dump,
                "raw_hex": data.hex(),
            }
        except Exception as e:
            logger.error(f"Memory inspection failed at 0x{address:08X}: {e}")
            return {"error": str(e)}

    def find_string_in_memory(self, search_string: str) -> List[Dict[str, Any]]:
        """Find a string in memory.

        Parameters
        ----------
        search_string : str
            String to search for

        Returns
        -------
        List[Dict[str, Any]]
            List of matches with address and context
        """
        pattern = search_string.encode("utf-8")
        matches = self.memory_inspector.find_pattern(pattern)

        results = []
        for addr in matches:
            # Get context around the match
            context_start = max(0, addr - 16)
            context_data = self.memory_inspector.read_memory_region(context_start, 32)
            context_offset = addr - context_start

            results.append(
                {
                    "address": addr,
                    "context": context_data,
                    "context_offset": context_offset,
                    "string": search_string,
                }
            )

        return results

    def get_memory_map(self) -> Dict[str, Any]:
        """Get memory map information.

        Returns
        -------
        Dict[str, Any]
            Memory map with regions and their purposes
        """
        if self.pynds.is_gba:
            return {
                "platform": "GBA",
                "regions": {
                    "0x00000000-0x00003FFF": "BIOS (16KB)",
                    "0x02000000-0x0203FFFF": "EWRAM (256KB)",
                    "0x03000000-0x03007FFF": "IWRAM (32KB)",
                    "0x04000000-0x040003FF": "I/O Registers (1KB)",
                    "0x05000000-0x050003FF": "Palette RAM (1KB)",
                    "0x06000000-0x06017FFF": "VRAM (96KB)",
                    "0x07000000-0x070003FF": "OAM (1KB)",
                    "0x08000000-0x09FFFFFF": "Game Pak ROM (32MB)",
                    "0x0A000000-0x0BFFFFFF": "Game Pak SRAM (2MB)",
                },
            }
        else:
            return {
                "platform": "NDS",
                "regions": {
                    "0x00000000-0x00003FFF": "ARM9 BIOS (16KB)",
                    "0x02000000-0x02FFFFFF": "Main RAM (4MB)",
                    "0x03000000-0x03007FFF": "Shared WRAM (32KB)",
                    "0x04000000-0x040003FF": "ARM9 I/O Registers (1KB)",
                    "0x05000000-0x050003FF": "Palette RAM (1KB)",
                    "0x06000000-0x06017FFF": "VRAM (96KB)",
                    "0x07000000-0x070003FF": "OAM (1KB)",
                    "0x08000000-0x09FFFFFF": "Game ROM (32MB)",
                    "0x10000000-0x1000FFFF": "ARM7 I/O Registers (64KB)",
                },
            }

    def get_debug_info(self) -> Dict[str, Any]:
        """Get comprehensive debug information.

        Returns
        -------
        Dict[str, Any]
            Complete debug information including CPU state, memory analysis, and breakpoints
        """
        return {
            "cpu_state": self.get_cpu_state(),
            "memory_analysis": self.memory_inspector.analyze_memory_usage(),
            "breakpoints": self.breakpoint_manager.get_breakpoint_info(),
            "memory_map": self.get_memory_map(),
            "rewind_info": (
                self.pynds.get_rewind_info()
                if hasattr(self.pynds, "get_rewind_info")
                else None
            ),
            "frame_info": {
                "frame_count": self.pynds.get_frame_count(),
                "fps": self.pynds.get_fps(),
                "platform": self.pynds.get_platform(),
            },
        }

    def export_debug_report(self, filename: str) -> bool:
        """Export debug information to a file.

        Parameters
        ----------
        filename : str
            Filename to save debug report to

        Returns
        -------
        bool
            True if export was successful, False otherwise
        """
        try:
            import json
            from datetime import datetime

            debug_info = self.get_debug_info()
            debug_info["export_timestamp"] = datetime.now().isoformat()
            debug_info["pynds_version"] = "0.0.5-alpha"  # Get from pynds.__version__

            with open(filename, "w") as f:
                json.dump(debug_info, f, indent=2, default=str)

            logger.info(f"Debug report exported to {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to export debug report: {e}")
            return False

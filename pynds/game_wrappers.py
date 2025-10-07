"""PyNDS Game Wrappers - PyBoy-inspired game-specific interfaces.

This module provides game-specific wrappers that make it easier to interact
with popular Nintendo DS and GBA games. Inspired by PyBoy's game wrapper
system, these wrappers provide high-level interfaces for common game actions,
memory access, and state management.

Classes:
    BaseGameWrapper: Base class for all game wrappers
    PokemonWrapper: Wrapper for Pokemon games (Ruby/Sapphire/Emerald, Diamond/Pearl/Platinum)
    MarioWrapper: Wrapper for Mario games (Super Mario 64 DS, New Super Mario Bros)
    ZeldaWrapper: Wrapper for Zelda games (Phantom Hourglass, Spirit Tracks)
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

from .pynds import PyNDS

logger = logging.getLogger(__name__)


class BaseGameWrapper(ABC):
    """Base class for all game wrappers.

    Provides common functionality and interface that all game-specific
    wrappers should implement. This ensures consistency and makes it
    easy to create new game wrappers.
    """

    def __init__(self, pynds: PyNDS):
        """Initialize game wrapper.

        Parameters
        ----------
        pynds : PyNDS
            PyNDS emulator instance
        """
        self.pynds = pynds
        self.game_name = "Unknown Game"
        self.platform = pynds.get_platform()
        self._memory_addresses: Dict[str, int] = {}
        self._last_state: Dict[str, Any] = {}

    @abstractmethod
    def get_game_state(self) -> Dict[str, Any]:
        """Get current game state.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing current game state information
        """
        pass

    @abstractmethod
    def get_available_actions(self) -> List[str]:
        """Get list of available actions for this game.

        Returns
        -------
        List[str]
            List of action names that can be performed
        """
        pass

    @abstractmethod
    def perform_action(self, action: str, **kwargs) -> bool:
        """Perform a game action.

        Parameters
        ----------
        action : str
            Action to perform
        **kwargs
            Additional parameters for the action

        Returns
        -------
        bool
            True if action was successful, False otherwise
        """
        pass

    def get_screen_analysis(self) -> Dict[str, Any]:
        """Analyze the current screen for game-specific information.

        Returns
        -------
        Dict[str, Any]
            Screen analysis results
        """
        try:
            frame = self.pynds.get_frame()
            if isinstance(frame, tuple):
                # NDS: analyze both screens
                top_frame, bottom_frame = frame
                return {
                    "top_screen": self._analyze_frame(top_frame),
                    "bottom_screen": self._analyze_frame(bottom_frame),
                    "platform": "nds",
                }
            else:
                # GBA: single screen
                return {"screen": self._analyze_frame(frame), "platform": "gba"}
        except Exception as e:
            logger.error(f"Screen analysis failed: {e}")
            return {"error": str(e)}

    def _analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze a single frame for game information.

        Parameters
        ----------
        frame : np.ndarray
            Frame data to analyze

        Returns
        -------
        Dict[str, Any]
            Frame analysis results
        """
        return {
            "shape": frame.shape,
            "mean_brightness": float(np.mean(frame)),
            "color_histogram": self._get_color_histogram(frame),
            "edge_density": self._get_edge_density(frame),
        }

    def _get_color_histogram(self, frame: np.ndarray) -> Dict[str, List[int]]:
        """Get color histogram for a frame.

        Parameters
        ----------
        frame : np.ndarray
            Frame data

        Returns
        -------
        Dict[str, List[int]]
            Color histogram data
        """
        if len(frame.shape) == 3:
            return {
                "red": np.histogram(frame[:, :, 0], bins=32, range=(0, 255))[
                    0
                ].tolist(),
                "green": np.histogram(frame[:, :, 1], bins=32, range=(0, 255))[
                    0
                ].tolist(),
                "blue": np.histogram(frame[:, :, 2], bins=32, range=(0, 255))[
                    0
                ].tolist(),
            }
        return {}

    def _get_edge_density(self, frame: np.ndarray) -> float:
        """Calculate edge density for a frame.

        Parameters
        ----------
        frame : np.ndarray
            Frame data

        Returns
        -------
        float
            Edge density (0.0 to 1.0)
        """
        try:
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray = np.dot(frame[..., :3], [0.299, 0.587, 0.114])
            else:
                gray = frame

            # Simple edge detection using gradient
            grad_x = np.abs(np.diff(gray, axis=1))
            grad_y = np.abs(np.diff(gray, axis=0))

            edge_density = (np.mean(grad_x) + np.mean(grad_y)) / 255.0
            return min(edge_density, 1.0)
        except Exception:
            return 0.0

    def save_game_state(self) -> Dict[str, Any]:
        """Save current game state for analysis.

        Returns
        -------
        Dict[str, Any]
            Saved game state
        """
        state = {
            "game_state": self.get_game_state(),
            "screen_analysis": self.get_screen_analysis(),
            "frame_count": self.pynds.get_frame_count(),
            "timestamp": self.pynds.get_timing_info(),
        }
        self._last_state = state
        return state

    def compare_states(
        self, state1: Dict[str, Any], state2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare two game states.

        Parameters
        ----------
        state1 : Dict[str, Any]
            First game state
        state2 : Dict[str, Any]
            Second game state

        Returns
        -------
        Dict[str, Any]
            Comparison results
        """
        changes = {}

        for key in state1.get("game_state", {}):
            if key in state2.get("game_state", {}):
                val1 = state1["game_state"][key]
                val2 = state2["game_state"][key]
                if val1 != val2:
                    changes[key] = {"from": val1, "to": val2}

        return {
            "changes": changes,
            "num_changes": len(changes),
            "frame_diff": state2.get("frame_count", 0) - state1.get("frame_count", 0),
        }


class PokemonWrapper(BaseGameWrapper):
    """Wrapper for Pokemon games (Ruby/Sapphire/Emerald, Diamond/Pearl/Platinum).

    Provides high-level interface for Pokemon game mechanics including
    Pokemon management, battle system, and world navigation.
    """

    def __init__(self, pynds: PyNDS):
        """Initialize Pokemon wrapper.

        Parameters
        ----------
        pynds : PyNDS
            PyNDS emulator instance
        """
        super().__init__(pynds)
        self.game_name = "Pokemon"

        # Memory addresses (these are examples - real addresses vary by game)
        self._memory_addresses = {
            "player_x": 0x02000000,
            "player_y": 0x02000004,
            "money": 0x02000008,
            "badges": 0x0200000C,
            "pokemon_count": 0x02000010,
            "current_hp": 0x02000014,
            "max_hp": 0x02000018,
            "level": 0x0200001C,
            "experience": 0x02000020,
        }

    def get_game_state(self) -> Dict[str, Any]:
        """Get current Pokemon game state.

        Returns
        -------
        Dict[str, Any]
            Current game state including player position, money, badges, etc.
        """
        try:
            # Read numeric values with precise types
            x = self.pynds.memory.read_ram_u16(self._memory_addresses["player_x"])
            y = self.pynds.memory.read_ram_u16(self._memory_addresses["player_y"])
            money = self.pynds.memory.read_ram_u32(self._memory_addresses["money"])
            badges = self.pynds.memory.read_ram_u8(self._memory_addresses["badges"])
            pokemon_count = self.pynds.memory.read_ram_u8(
                self._memory_addresses["pokemon_count"]
            )
            current_hp = self.pynds.memory.read_ram_u16(
                self._memory_addresses["current_hp"]
            )
            max_hp = self.pynds.memory.read_ram_u16(self._memory_addresses["max_hp"])
            level = self.pynds.memory.read_ram_u8(self._memory_addresses["level"])
            experience = self.pynds.memory.read_ram_u32(
                self._memory_addresses["experience"]
            )

            health_percentage: float = (current_hp / max_hp) if max_hp > 0 else 0.0

            state: Dict[str, Any] = {
                "player_position": {"x": x, "y": y},
                "money": money,
                "badges": badges,
                "pokemon_count": pokemon_count,
                "current_hp": current_hp,
                "max_hp": max_hp,
                "level": level,
                "experience": experience,
                "health_percentage": health_percentage,
            }

            return state
        except Exception as e:
            logger.error(f"Failed to get Pokemon game state: {e}")
            return {"error": str(e)}

    def get_available_actions(self) -> List[str]:
        """Get available Pokemon game actions.

        Returns
        -------
        List[str]
            List of available actions
        """
        return [
            "move_up",
            "move_down",
            "move_left",
            "move_right",
            "interact",
            "open_menu",
            "run",
            "bike",
            "use_item",
            "throw_pokeball",
            "battle",
            "save_game",
        ]

    def perform_action(self, action: str, **kwargs) -> bool:
        """Perform a Pokemon game action.

        Parameters
        ----------
        action : str
            Action to perform
        **kwargs
            Additional parameters

        Returns
        -------
        bool
            True if action was successful
        """
        try:
            if action == "move_up":
                self.pynds.button.press_key("up")
            elif action == "move_down":
                self.pynds.button.press_key("down")
            elif action == "move_left":
                self.pynds.button.press_key("left")
            elif action == "move_right":
                self.pynds.button.press_key("right")
            elif action == "interact":
                self.pynds.button.press_key("a")
            elif action == "open_menu":
                self.pynds.button.press_key("start")
            elif action == "run":
                self.pynds.button.press_key("b")
            elif action == "bike":
                # Bike toggle (example)
                self.pynds.button.press_key("select")
            elif action == "use_item":
                # Open item menu
                self.pynds.button.press_key("start")
                self.pynds.tick(30)  # Wait for menu
                self.pynds.button.press_key("a")
            elif action == "throw_pokeball":
                # Use pokeball (example)
                self.pynds.button.press_key("a")
            elif action == "battle":
                # Start battle (example)
                self.pynds.button.press_key("a")
            elif action == "save_game":
                self.pynds.write_save_file("pokemon_save.sav")
            else:
                logger.warning(f"Unknown Pokemon action: {action}")
                return False

            # Advance emulation
            self.pynds.tick(1)
            return True

        except Exception as e:
            logger.error(f"Failed to perform Pokemon action {action}: {e}")
            return False

    def catch_pokemon(self) -> bool:
        """Attempt to catch a Pokemon.

        Returns
        -------
        bool
            True if catch attempt was made
        """
        try:
            # Use pokeball
            self.pynds.button.press_key("a")
            self.pynds.tick(60)  # Wait for animation
            return True
        except Exception as e:
            logger.error(f"Failed to catch Pokemon: {e}")
            return False

    def heal_pokemon(self) -> bool:
        """Heal Pokemon at a Pokemon Center.

        Returns
        -------
        bool
            True if heal attempt was made
        """
        try:
            # Interact with nurse
            self.pynds.button.press_key("a")
            self.pynds.tick(30)
            self.pynds.button.press_key("a")  # Confirm heal
            self.pynds.tick(60)
            return True
        except Exception as e:
            logger.error(f"Failed to heal Pokemon: {e}")
            return False


class MarioWrapper(BaseGameWrapper):
    """Wrapper for Mario games (Super Mario 64 DS, New Super Mario Bros).

    Provides high-level interface for Mario game mechanics including
    movement, jumping, collecting items, and level progression.
    """

    def __init__(self, pynds: PyNDS):
        """Initialize Mario wrapper.

        Parameters
        ----------
        pynds : PyNDS
            PyNDS emulator instance
        """
        super().__init__(pynds)
        self.game_name = "Mario"

        # Memory addresses (examples)
        self._memory_addresses = {
            "mario_x": 0x02000000,
            "mario_y": 0x02000004,
            "mario_z": 0x02000008,
            "lives": 0x0200000C,
            "coins": 0x02000010,
            "score": 0x02000014,
            "level": 0x02000018,
            "stars": 0x0200001C,
        }

    def get_game_state(self) -> Dict[str, Any]:
        """Get current Mario game state.

        Returns
        -------
        Dict[str, Any]
            Current game state including position, lives, coins, etc.
        """
        try:
            return {
                "position": {
                    "x": self.pynds.memory.read_ram_f32(
                        self._memory_addresses["mario_x"]
                    ),
                    "y": self.pynds.memory.read_ram_f32(
                        self._memory_addresses["mario_y"]
                    ),
                    "z": self.pynds.memory.read_ram_f32(
                        self._memory_addresses["mario_z"]
                    ),
                },
                "lives": self.pynds.memory.read_ram_u8(self._memory_addresses["lives"]),
                "coins": self.pynds.memory.read_ram_u16(
                    self._memory_addresses["coins"]
                ),
                "score": self.pynds.memory.read_ram_u32(
                    self._memory_addresses["score"]
                ),
                "level": self.pynds.memory.read_ram_u8(self._memory_addresses["level"]),
                "stars": self.pynds.memory.read_ram_u8(self._memory_addresses["stars"]),
            }
        except Exception as e:
            logger.error(f"Failed to get Mario game state: {e}")
            return {"error": str(e)}

    def get_available_actions(self) -> List[str]:
        """Get available Mario game actions.

        Returns
        -------
        List[str]
            List of available actions
        """
        return [
            "move_left",
            "move_right",
            "jump",
            "crouch",
            "run",
            "punch",
            "kick",
            "collect_item",
            "pause",
            "camera_up",
            "camera_down",
            "camera_left",
            "camera_right",
        ]

    def perform_action(self, action: str, **kwargs) -> bool:
        """Perform a Mario game action.

        Parameters
        ----------
        action : str
            Action to perform
        **kwargs
            Additional parameters

        Returns
        -------
        bool
            True if action was successful
        """
        try:
            if action == "move_left":
                self.pynds.button.press_key("left")
            elif action == "move_right":
                self.pynds.button.press_key("right")
            elif action == "jump":
                self.pynds.button.press_key("a")
            elif action == "crouch":
                self.pynds.button.press_key("down")
            elif action == "run":
                self.pynds.button.press_key("b")
            elif action == "punch":
                self.pynds.button.press_key("y")
            elif action == "kick":
                self.pynds.button.press_key("x")
            elif action == "collect_item":
                self.pynds.button.press_key("a")
            elif action == "pause":
                self.pynds.button.press_key("start")
            elif action == "camera_up":
                self.pynds.button.press_key("up")
            elif action == "camera_down":
                self.pynds.button.press_key("down")
            elif action == "camera_left":
                self.pynds.button.press_key("left")
            elif action == "camera_right":
                self.pynds.button.press_key("right")
            else:
                logger.warning(f"Unknown Mario action: {action}")
                return False

            self.pynds.tick(1)
            return True

        except Exception as e:
            logger.error(f"Failed to perform Mario action {action}: {e}")
            return False

    def collect_coin(self) -> bool:
        """Attempt to collect a coin.

        Returns
        -------
        bool
            True if collection attempt was made
        """
        try:
            self.pynds.button.press_key("a")
            self.pynds.tick(10)
            return True
        except Exception as e:
            logger.error(f"Failed to collect coin: {e}")
            return False

    def jump_on_enemy(self) -> bool:
        """Attempt to jump on an enemy.

        Returns
        -------
        bool
            True if jump attempt was made
        """
        try:
            self.pynds.button.press_key("a")  # Jump
            self.pynds.tick(30)
            return True
        except Exception as e:
            logger.error(f"Failed to jump on enemy: {e}")
            return False


class ZeldaWrapper(BaseGameWrapper):
    """Wrapper for Zelda games (Phantom Hourglass, Spirit Tracks).

    Provides high-level interface for Zelda game mechanics including
    movement, sword fighting, item usage, and puzzle solving.
    """

    def __init__(self, pynds: PyNDS):
        """Initialize Zelda wrapper.

        Parameters
        ----------
        pynds : PyNDS
            PyNDS emulator instance
        """
        super().__init__(pynds)
        self.game_name = "Zelda"

        # Memory addresses (examples)
        self._memory_addresses = {
            "link_x": 0x02000000,
            "link_y": 0x02000004,
            "hearts": 0x02000008,
            "max_hearts": 0x0200000C,
            "rupees": 0x02000010,
            "bombs": 0x02000014,
            "arrows": 0x02000018,
            "keys": 0x0200001C,
        }

    def get_game_state(self) -> Dict[str, Any]:
        """Get current Zelda game state.

        Returns
        -------
        Dict[str, Any]
            Current game state including position, health, items, etc.
        """
        try:
            hearts = self.pynds.memory.read_ram_u8(self._memory_addresses["hearts"])
            max_hearts = self.pynds.memory.read_ram_u8(
                self._memory_addresses["max_hearts"]
            )

            return {
                "position": {
                    "x": self.pynds.memory.read_ram_f32(
                        self._memory_addresses["link_x"]
                    ),
                    "y": self.pynds.memory.read_ram_f32(
                        self._memory_addresses["link_y"]
                    ),
                },
                "hearts": hearts,
                "max_hearts": max_hearts,
                "health_percentage": hearts / max_hearts if max_hearts > 0 else 0,
                "rupees": self.pynds.memory.read_ram_u16(
                    self._memory_addresses["rupees"]
                ),
                "bombs": self.pynds.memory.read_ram_u8(self._memory_addresses["bombs"]),
                "arrows": self.pynds.memory.read_ram_u8(
                    self._memory_addresses["arrows"]
                ),
                "keys": self.pynds.memory.read_ram_u8(self._memory_addresses["keys"]),
            }
        except Exception as e:
            logger.error(f"Failed to get Zelda game state: {e}")
            return {"error": str(e)}

    def get_available_actions(self) -> List[str]:
        """Get available Zelda game actions.

        Returns
        -------
        List[str]
            List of available actions
        """
        return [
            "move_up",
            "move_down",
            "move_left",
            "move_right",
            "sword_attack",
            "use_item",
            "interact",
            "roll",
            "use_bomb",
            "use_arrow",
            "open_map",
            "pause",
        ]

    def perform_action(self, action: str, **kwargs) -> bool:
        """Perform a Zelda game action.

        Parameters
        ----------
        action : str
            Action to perform
        **kwargs
            Additional parameters

        Returns
        -------
        bool
            True if action was successful
        """
        try:
            if action == "move_up":
                self.pynds.button.press_key("up")
            elif action == "move_down":
                self.pynds.button.press_key("down")
            elif action == "move_left":
                self.pynds.button.press_key("left")
            elif action == "move_right":
                self.pynds.button.press_key("right")
            elif action == "sword_attack":
                self.pynds.button.press_key("a")
            elif action == "use_item":
                self.pynds.button.press_key("b")
            elif action == "interact":
                self.pynds.button.press_key("a")
            elif action == "roll":
                self.pynds.button.press_key("b")
            elif action == "use_bomb":
                self.pynds.button.press_key("b")
            elif action == "use_arrow":
                self.pynds.button.press_key("b")
            elif action == "open_map":
                self.pynds.button.press_key("start")
            elif action == "pause":
                self.pynds.button.press_key("start")
            else:
                logger.warning(f"Unknown Zelda action: {action}")
                return False

            self.pynds.tick(1)
            return True

        except Exception as e:
            logger.error(f"Failed to perform Zelda action {action}: {e}")
            return False

    def attack_enemy(self) -> bool:
        """Attack with sword.

        Returns
        -------
        bool
            True if attack was performed
        """
        try:
            self.pynds.button.press_key("a")
            self.pynds.tick(20)
            return True
        except Exception as e:
            logger.error(f"Failed to attack enemy: {e}")
            return False

    def collect_rupee(self) -> bool:
        """Attempt to collect a rupee.

        Returns
        -------
        bool
            True if collection attempt was made
        """
        try:
            self.pynds.button.press_key("a")
            self.pynds.tick(10)
            return True
        except Exception as e:
            logger.error(f"Failed to collect rupee: {e}")
            return False


def create_game_wrapper(pynds: PyNDS, game_type: str) -> Optional[BaseGameWrapper]:
    """Create a game wrapper for the specified game type.

    Parameters
    ----------
    pynds : PyNDS
        PyNDS emulator instance
    game_type : str
        Type of game wrapper to create ('pokemon', 'mario', 'zelda')

    Returns
    -------
    Optional[BaseGameWrapper]
        Game wrapper instance or None if type not supported
    """
    game_type = game_type.lower()

    if game_type == "pokemon":
        return PokemonWrapper(pynds)
    elif game_type == "mario":
        return MarioWrapper(pynds)
    elif game_type == "zelda":
        return ZeldaWrapper(pynds)
    else:
        logger.warning(f"Unknown game type: {game_type}")
        return None

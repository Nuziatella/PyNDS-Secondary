"""PyNDS Gym Environment for Reinforcement Learning.

This module provides a Gym-compatible wrapper for PyNDS, enabling seamless
integration with popular RL frameworks like Stable-Baselines3, Ray RLlib,
and OpenAI Gym. Perfect for training AI agents to play Nintendo DS games!

The environment supports both discrete and continuous action spaces, flexible
observation spaces, and customizable reward functions.

Examples
--------
>>> import gym
>>> from pynds.gym_env import PyNDSGymEnv
>>>
>>> # Create environment
>>> env = PyNDSGymEnv("game.nds", render_mode="human")
>>>
>>> # Standard Gym interface
>>> obs, info = env.reset()
>>> action = env.action_space.sample()
>>> obs, reward, terminated, truncated, info = env.step(action)
>>> env.close()
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import RenderFrame

from .pynds import PyNDS

logger = logging.getLogger(__name__)


class PyNDSGymEnv(gym.Env):
    """Gym-compatible wrapper for PyNDS emulator.

    This class wraps PyNDS in a standard Gym environment interface, making it
    compatible with popular RL frameworks. It supports both Nintendo DS and
    Game Boy Advance games with flexible action and observation spaces.

    Parameters
    ----------
    rom_path : str
        Path to the ROM file (.nds or .gba)
    action_type : str, optional
        Type of action space: 'discrete', 'multi_discrete', or 'continuous', by default 'discrete'
    observation_type : str, optional
        Type of observation: 'rgb', 'grayscale', or 'raw', by default 'rgb'
    frame_skip : int, optional
        Number of frames to skip between actions, by default 1
    max_episode_steps : int, optional
        Maximum steps per episode, by default 10000
    reward_type : str, optional
        Type of reward function: 'custom', 'frame_diff', or 'none', by default 'custom'
    render_mode : Optional[str], optional
        Render mode for visualization, by default None
    **kwargs
        Additional arguments passed to PyNDS constructor

    Attributes
    ----------
    action_space : gym.Space
        Action space defining valid actions
    observation_space : gym.Space
        Observation space defining valid observations
    pynds : PyNDS
        Underlying PyNDS emulator instance

    Examples
    --------
    >>> # Basic usage
    >>> env = PyNDSGymEnv("pokemon.nds")
    >>> obs, info = env.reset()
    >>> action = env.action_space.sample()
    >>> obs, reward, done, truncated, info = env.step(action)
    >>> env.close()

    >>> # With custom settings
    >>> env = PyNDSGymEnv(
    ...     "game.nds",
    ...     action_type="multi_discrete",
    ...     observation_type="grayscale",
    ...     frame_skip=4,
    ...     max_episode_steps=5000
    ... )
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        rom_path: str,
        action_type: str = "discrete",
        observation_type: str = "rgb",
        frame_skip: int = 1,
        max_episode_steps: int = 10000,
        reward_type: str = "custom",
        render_mode: Optional[str] = None,
        **kwargs,
    ):
        """Initialize PyNDS Gym environment.

        Parameters
        ----------
        rom_path : str
            Path to the ROM file
        action_type : str, optional
            Action space type, by default 'discrete'
        observation_type : str, optional
            Observation type, by default 'rgb'
        frame_skip : int, optional
            Frames to skip between actions, by default 1
        max_episode_steps : int, optional
            Maximum episode length, by default 10000
        reward_type : str, optional
            Reward function type, by default 'custom'
        render_mode : Optional[str], optional
            Render mode, by default None
        **kwargs
            Additional PyNDS arguments
        """
        super().__init__()

        self.rom_path = rom_path
        self.action_type = action_type
        self.observation_type = observation_type
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.reward_type = reward_type
        self.render_mode = render_mode

        # Initialize PyNDS emulator
        self.pynds = PyNDS(rom_path, **kwargs)

        # Episode tracking
        self.episode_steps = 0
        self.episode_reward: float = 0.0
        self.last_frame: Optional[np.ndarray] = None

        # Define action and observation spaces
        self._setup_action_space()
        self._setup_observation_space()

        # Initialize reward function
        self._setup_reward_function()

        logger.info(f"PyNDS Gym environment initialized: {rom_path}")
        logger.info(
            f"Action space: {self.action_type}, Observation: {self.observation_type}"
        )

    def _setup_action_space(self) -> None:
        """Set up the action space based on ``action_type``."""
        if self.action_type == "discrete":
            # Simple discrete actions: no-op, A, B, Start, Select, D-pad, L, R
            self.action_space = spaces.Discrete(9)
            self._action_meanings = [
                "noop",
                "A",
                "B",
                "Start",
                "Select",
                "Up",
                "Down",
                "Left",
                "Right",
            ]
        elif self.action_type == "multi_discrete":
            # Multi-discrete: buttons + d-pad + touch
            self.action_space = spaces.MultiDiscrete(
                [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
            )  # 12 buttons
            self._action_meanings = [
                "A",
                "B",
                "X",
                "Y",
                "L",
                "R",
                "Start",
                "Select",
                "Up",
                "Down",
                "Left",
                "Right",
            ]
        elif self.action_type == "continuous":
            # Continuous: button pressures + touch coordinates
            self.action_space = spaces.Box(
                low=0.0, high=1.0, shape=(14,), dtype=np.float32
            )  # 12 buttons + 2 touch coordinates
        else:
            raise ValueError(f"Unsupported action_type: {self.action_type}")

    def _setup_observation_space(self) -> None:
        """Set up the observation space based on ``observation_type``."""
        if self.observation_type == "rgb":
            if self.pynds.is_gba:
                # GBA: single screen, RGB
                self.observation_space = spaces.Box(
                    low=0, high=255, shape=(240, 160, 3), dtype=np.uint8
                )
            else:
                # NDS: dual screen, stacked vertically
                self.observation_space = spaces.Box(
                    low=0, high=255, shape=(480, 256, 3), dtype=np.uint8
                )
        elif self.observation_type == "grayscale":
            if self.pynds.is_gba:
                self.observation_space = spaces.Box(
                    low=0, high=255, shape=(240, 160, 1), dtype=np.uint8
                )
            else:
                self.observation_space = spaces.Box(
                    low=0, high=255, shape=(480, 256, 1), dtype=np.uint8
                )
        elif self.observation_type == "raw":
            # Raw frame data as-is
            if self.pynds.is_gba:
                self.observation_space = spaces.Box(
                    low=0, high=255, shape=(240, 160, 4), dtype=np.uint8
                )
            else:
                self.observation_space = spaces.Box(
                    low=0, high=255, shape=(480, 256, 4), dtype=np.uint8
                )
        else:
            raise ValueError(f"Unsupported observation_type: {self.observation_type}")

    def _setup_reward_function(self) -> None:
        """Set up the reward function based on ``reward_type``."""
        if self.reward_type == "custom":
            self._reward_function = self._custom_reward
        elif self.reward_type == "frame_diff":
            self._reward_function = self._frame_diff_reward
        elif self.reward_type == "none":
            self._reward_function = self._no_reward
        else:
            raise ValueError(f"Unsupported reward_type: {self.reward_type}")

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state.

        Parameters
        ----------
        seed : Optional[int], optional
            Random seed for reproducibility, by default None
        options : Optional[Dict[str, Any]], optional
            Additional reset options, by default None

        Returns
        -------
        Tuple[np.ndarray, Dict[str, Any]]
            Initial observation and info dictionary
        """
        super().reset(seed=seed)

        # Reset emulator
        self.pynds.reset()

        # Reset episode tracking
        self.episode_steps = 0
        self.episode_reward = 0.0
        self.last_frame = None

        # Get initial observation
        obs = self._get_observation()

        info = {
            "episode_steps": self.episode_steps,
            "episode_reward": self.episode_reward,
            "is_gba": self.pynds.is_gba,
            "frame_count": self.pynds.get_frame_count(),
        }

        logger.debug(f"Environment reset: episode_steps={self.episode_steps}")
        return obs, info

    def step(
        self, action: Union[int, np.ndarray, List]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute action and return next observation.

        Parameters
        ----------
        action : Union[int, np.ndarray, List]
            Action to execute

        Returns
        -------
        Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]
            Observation, reward, terminated, truncated, info
        """
        # Execute action
        self._execute_action(action)

        # Advance emulation
        for _ in range(self.frame_skip):
            self.pynds.tick()

        # Get observation
        obs = self._get_observation()

        # Calculate reward
        reward = self._reward_function(obs)
        self.episode_reward += reward

        # Update episode tracking
        self.episode_steps += 1

        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self.episode_steps >= self.max_episode_steps

        info = {
            "episode_steps": self.episode_steps,
            "episode_reward": self.episode_reward,
            "is_gba": self.pynds.is_gba,
            "frame_count": self.pynds.get_frame_count(),
            "action": action,
            "reward": reward,
        }

        logger.debug(f"Step {self.episode_steps}: action={action}, reward={reward:.3f}")
        return obs, reward, terminated, truncated, info

    def _execute_action(self, action: Union[int, np.ndarray, List]) -> None:
        """Execute action in the emulator."""
        if self.action_type == "discrete":
            self._execute_discrete_action(action)
        elif self.action_type == "multi_discrete":
            self._execute_multi_discrete_action(action)
        elif self.action_type == "continuous":
            self._execute_continuous_action(action)

    def _execute_discrete_action(self, action: int) -> None:
        """Execute discrete action."""
        if action == 0:  # no-op
            return
        elif action == 1:  # A
            self.pynds.button.press_key("a")
        elif action == 2:  # B
            self.pynds.button.press_key("b")
        elif action == 3:  # Start
            self.pynds.button.press_key("start")
        elif action == 4:  # Select
            self.pynds.button.press_key("select")
        elif action == 5:  # Up
            self.pynds.button.press_key("up")
        elif action == 6:  # Down
            self.pynds.button.press_key("down")
        elif action == 7:  # Left
            self.pynds.button.press_key("left")
        elif action == 8:  # Right
            self.pynds.button.press_key("right")

    def _execute_multi_discrete_action(self, action: List[int]) -> None:
        """Execute multi-discrete action."""
        button_actions = [
            "a",
            "b",
            "x",
            "y",
            "l",
            "r",
            "start",
            "select",
            "up",
            "down",
            "left",
            "right",
        ]

        for i, pressed in enumerate(action):
            if pressed and i < len(button_actions):
                self.pynds.button.press_key(button_actions[i])

    def _execute_continuous_action(self, action: np.ndarray) -> None:
        """Execute continuous action."""
        # First 12 elements are button pressures
        button_actions = [
            "a",
            "b",
            "x",
            "y",
            "l",
            "r",
            "start",
            "select",
            "up",
            "down",
            "left",
            "right",
        ]

        for i, pressure in enumerate(action[:12]):
            if pressure > 0.5:  # Threshold for button press
                self.pynds.button.press_key(button_actions[i])

        # Last 2 elements are touch coordinates (if NDS)
        if not self.pynds.is_gba and len(action) >= 14:
            touch_x = int(action[12] * 255)  # Scale to screen width
            touch_y = int(action[13] * 191)  # Scale to screen height
            self.pynds.button.set_touch(touch_x, touch_y)
            self.pynds.button.touch()

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        frame = self.pynds.get_frame()

        # Narrow union type from (ndarray | tuple[ndarray, ndarray])
        obs: np.ndarray
        if isinstance(frame, tuple):
            # NDS: combine top and bottom screens
            top_frame, bottom_frame = frame
            obs = np.vstack([top_frame, bottom_frame])
        else:
            # GBA: single frame
            obs = frame

        # Convert to desired observation type
        if self.observation_type == "grayscale":
            if len(obs.shape) == 3:
                # Convert RGB to grayscale
                obs = np.dot(obs[..., :3], [0.299, 0.587, 0.114])
                obs = obs.astype(np.uint8)
                obs = np.expand_dims(obs, axis=-1)
        elif self.observation_type == "rgb":
            if len(obs.shape) == 3 and obs.shape[2] == 4:
                # Convert RGBA to RGB
                obs = obs[..., :3]

        return obs

    def _custom_reward(self, obs: np.ndarray) -> float:
        """Compute the custom reward; override for specific games."""
        # Simple reward based on frame changes
        if self.last_frame is not None:
            frame_diff = np.mean(
                np.abs(obs.astype(float) - self.last_frame.astype(float))
            )
            reward = min(frame_diff / 100.0, 1.0)  # Normalize to [0, 1]
        else:
            reward = 0.0

        self.last_frame = obs.copy()
        return reward

    def _frame_diff_reward(self, obs: np.ndarray) -> float:
        """Reward based on frame differences."""
        if self.last_frame is not None:
            diff = np.mean(np.abs(obs.astype(float) - self.last_frame.astype(float)))
            return float(diff)
        return 0.0

    def _no_reward(self, obs: np.ndarray) -> float:
        """No reward function."""
        return 0.0

    def _is_terminated(self) -> bool:
        """Check if episode is terminated."""
        # Override this for game-specific termination conditions
        return False

    def render(self) -> Union[RenderFrame, List[RenderFrame], None]:
        """Render the environment."""
        if self.render_mode == "human":
            # Use PyNDS window for human rendering
            if not hasattr(self, "_window_initialized"):
                self.pynds.window.init_window()
                self._window_initialized = True
            self.pynds.window.render()
            return None
        elif self.render_mode == "rgb_array":
            # Return RGB array
            return self._get_observation()
        else:
            return None

    def close(self) -> None:
        """Close the environment."""
        if hasattr(self, "pynds"):
            self.pynds.close()
        super().close()

    def seed(self, seed: Optional[int] = None) -> None:
        """Set random seed for reproducibility."""
        if seed is not None:
            np.random.seed(seed)

    def get_action_meanings(self) -> List[str]:
        """Get human-readable action meanings."""
        return getattr(self, "_action_meanings", [])

    def get_episode_info(self) -> Dict[str, Any]:
        """Get current episode information."""
        return {
            "episode_steps": self.episode_steps,
            "episode_reward": self.episode_reward,
            "is_gba": self.pynds.is_gba,
            "frame_count": self.pynds.get_frame_count(),
        }


class CustomPyNDSGymEnv(PyNDSGymEnv):
    """PyNDS Gym Environment with custom reward functions.

    This class extends PyNDSGymEnv with enhanced reward functionality,
    including memory-based rewards, progress tracking, and exploration
    incentives. Perfect for fine-tuning your RL agents!

    Parameters
    ----------
    rom_path : str
        Path to the ROM file (.nds or .gba)
    **kwargs
        Additional arguments passed to PyNDSGymEnv

    Examples
    --------
    >>> env = CustomPyNDSGymEnv("game.nds")
    >>> obs, info = env.reset()
    >>> action = env.action_space.sample()
    >>> obs, reward, done, truncated, info = env.step(action)
    >>> analysis = env.get_reward_analysis()
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Custom reward tracking
        self.reward_history = []
        self.last_memory_values = {}
        self.progress_tracker: float = 0.0
        self.survival_steps = 0
        self.exploration_map = set()

        # Game-specific parameters (customize these!)
        self.target_memory_addresses = {
            "score": 0x02000000,  # Example: score address
            "lives": 0x02000004,  # Example: lives address
            "level": 0x02000008,  # Example: level address
        }

    def _custom_reward(self, obs: np.ndarray) -> float:
        """Enhanced custom reward function."""
        total_reward = 0.0

        # 1. Memory-based reward
        memory_reward = self._memory_based_reward()
        total_reward += memory_reward

        # 2. Progress-based reward
        progress_reward = self._progress_based_reward()
        total_reward += progress_reward

        # 3. Survival reward
        survival_reward = self._survival_reward()
        total_reward += survival_reward

        # 4. Exploration reward
        exploration_reward = self._exploration_reward(obs)
        total_reward += exploration_reward

        # 5. Frame change reward
        frame_reward = self._frame_diff_reward(obs)
        total_reward += frame_reward

        # Store reward components for analysis
        self.reward_history.append(
            {
                "total": total_reward,
                "memory": memory_reward,
                "progress": progress_reward,
                "survival": survival_reward,
                "exploration": exploration_reward,
                "frame": frame_reward,
            }
        )

        return total_reward

    def _memory_based_reward(self) -> float:
        """Reward based on specific memory values."""
        reward = 0.0

        try:
            for name, address in self.target_memory_addresses.items():
                # Read memory value (this is game-specific!)
                if name == "score":
                    # Example: reward for increasing score
                    current_score = self.pynds.memory.read_ram_u32(address)
                    if name in self.last_memory_values:
                        score_diff = current_score - self.last_memory_values[name]
                        if score_diff > 0:
                            reward += score_diff * 0.01  # Scale reward
                    self.last_memory_values[name] = current_score

                elif name == "lives":
                    # Example: penalty for losing lives
                    current_lives = self.pynds.memory.read_ram_u8(address)
                    if name in self.last_memory_values:
                        lives_diff = current_lives - self.last_memory_values[name]
                        if lives_diff < 0:
                            reward += lives_diff * 10.0  # Penalty for losing lives
                    self.last_memory_values[name] = current_lives

                elif name == "level":
                    # Example: reward for level progression
                    current_level = self.pynds.memory.read_ram_u8(address)
                    if name in self.last_memory_values:
                        level_diff = current_level - self.last_memory_values[name]
                        if level_diff > 0:
                            reward += level_diff * 100.0  # Big reward for level up!
                    self.last_memory_values[name] = current_level

        except Exception as e:  # nosec B110
            # If memory reading fails, continue without memory rewards
            logger.debug("memory-based reward read failed: %s", e)

        return reward

    def _progress_based_reward(self) -> float:
        """Reward based on game progress."""
        reward = 0.0

        # Example: reward for advancing in the game
        # This could be based on level, area, or other progress indicators
        try:
            # Check if we've made progress (customize this logic!)
            current_progress = self._get_game_progress()
            if current_progress > self.progress_tracker:
                progress_diff = current_progress - self.progress_tracker
                reward += progress_diff * 50.0  # Reward for progress
                self.progress_tracker = current_progress
        except Exception as e:  # nosec B110
            logger.debug("progress-based reward read failed: %s", e)

        return reward

    def _survival_reward(self) -> float:
        """Reward for staying alive."""
        self.survival_steps += 1

        # Small positive reward for each step survived
        survival_reward = 0.1

        # Bonus for surviving longer
        if self.survival_steps % 100 == 0:
            survival_reward += 1.0

        return survival_reward

    def _exploration_reward(self, obs: np.ndarray) -> float:
        """Reward for exploring new areas."""
        reward = 0.0

        # Create a simple hash of the observation for exploration tracking
        obs_hash = hash(obs.tobytes())

        if obs_hash not in self.exploration_map:
            self.exploration_map.add(obs_hash)
            reward += 1.0  # Reward for seeing new states

        return reward

    def _get_game_progress(self) -> float:
        """Get current game progress (customize for your game!)."""
        # This is a placeholder - implement based on your specific game
        # Examples:
        # - Level number
        # - Area/zone progress
        # - Score milestones
        # - Items collected

        try:
            # Example: use level as progress indicator
            level = self.pynds.memory.read_ram_u8(0x02000008)
            return float(level)
        except Exception:
            return 0.0

    def get_reward_analysis(self) -> Dict[str, Any]:
        """Get analysis of reward components."""
        if not self.reward_history:
            return {}

        # Calculate statistics for each reward component
        analysis = {}
        for component in [
            "total",
            "memory",
            "progress",
            "survival",
            "exploration",
            "frame",
        ]:
            values = [r[component] for r in self.reward_history]
            analysis[component] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "total": np.sum(values),
            }

        return analysis

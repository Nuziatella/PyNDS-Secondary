"""Tests for PyNDS Gym Environment wrapper.

This module contains comprehensive tests for the Gym-compatible wrapper,
ensuring it works correctly with different action spaces, observation types,
and reward functions. Perfect for validating RL integration!
"""

from unittest.mock import Mock, patch

import gymnasium as gym
import numpy as np
import pytest

from pynds.gym_env import CustomPyNDSGymEnv, PyNDSGymEnv


@pytest.fixture
def mock_pynds_instance():
    """Create a real PyNDS instance with mocked C++ components."""
    with patch("pynds.gym_env.PyNDS") as mock_pynds_class:
        with patch("os.path.isfile", return_value=True):
            # Mock the PyNDS instance
            mock_pynds = Mock()
            mock_pynds.is_gba = False
            mock_pynds.get_frame.return_value = (
                np.random.randint(0, 255, (256, 192, 4), dtype=np.uint8),
                np.random.randint(0, 255, (256, 192, 4), dtype=np.uint8),
            )
            mock_pynds.get_frame_count.return_value = 100
            mock_pynds.reset.return_value = None
            mock_pynds.close.return_value = None
            mock_pynds.button = Mock()
            mock_pynds.memory = Mock()
            mock_pynds.window = Mock()

            mock_pynds_class.return_value = mock_pynds

            yield mock_pynds


class TestPyNDSGymEnvInitialization:
    """Test Gym environment initialization."""

    def test_init_discrete_actions(self, mock_pynds_instance):
        """Test initialization with discrete actions."""
        env = PyNDSGymEnv(
            rom_path="test.nds", action_type="discrete", observation_type="rgb"
        )

        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n == 9
        assert len(env.get_action_meanings()) == 9
        env.close()

    def test_init_multi_discrete_actions(self, mock_pynds_instance):
        """Test initialization with multi-discrete actions."""
        env = PyNDSGymEnv(
            rom_path="test.nds", action_type="multi_discrete", observation_type="rgb"
        )

        assert isinstance(env.action_space, gym.spaces.MultiDiscrete)
        assert len(env.action_space.nvec) == 12
        env.close()

    def test_init_continuous_actions(self, mock_pynds_instance):
        """Test initialization with continuous actions."""
        env = PyNDSGymEnv(
            rom_path="test.nds", action_type="continuous", observation_type="rgb"
        )

        assert isinstance(env.action_space, gym.spaces.Box)
        assert env.action_space.shape == (14,)
        env.close()

    def test_init_rgb_observations(self, mock_pynds_instance):
        """Test initialization with RGB observations."""
        env = PyNDSGymEnv(
            rom_path="test.nds", action_type="discrete", observation_type="rgb"
        )

        assert isinstance(env.observation_space, gym.spaces.Box)
        assert env.observation_space.shape == (480, 256, 3)  # NDS stacked
        env.close()

    def test_init_grayscale_observations(self, mock_pynds_instance):
        """Test initialization with grayscale observations."""
        env = PyNDSGymEnv(
            rom_path="test.nds", action_type="discrete", observation_type="grayscale"
        )

        assert isinstance(env.observation_space, gym.spaces.Box)
        assert env.observation_space.shape == (480, 256, 1)  # NDS grayscale
        env.close()

    def test_init_gba_mode(self, mock_pynds_instance):
        """Test initialization with GBA mode."""
        mock_pynds_instance.is_gba = True
        mock_pynds_instance.get_frame.return_value = np.random.randint(
            0, 255, (160, 240, 4), dtype=np.uint8
        )

        env = PyNDSGymEnv(
            rom_path="test.gba", action_type="discrete", observation_type="rgb"
        )

        assert env.observation_space.shape == (240, 160, 3)  # GBA RGB
        env.close()

    def test_invalid_action_type(self, mock_pynds_instance):
        """Test initialization with invalid action type."""
        with pytest.raises(ValueError, match="Unsupported action_type"):
            PyNDSGymEnv(
                rom_path="test.nds", action_type="invalid", observation_type="rgb"
            )

    def test_invalid_observation_type(self, mock_pynds_instance):
        """Test initialization with invalid observation type."""
        with pytest.raises(ValueError, match="Unsupported observation_type"):
            PyNDSGymEnv(
                rom_path="test.nds", action_type="discrete", observation_type="invalid"
            )


class TestPyNDSGymEnvBasicFunctionality:
    """Test basic Gym environment functionality."""

    def test_reset(self, mock_pynds_instance):
        """Test environment reset."""
        env = PyNDSGymEnv("test.nds")

        obs, info = env.reset()

        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)
        assert "episode_steps" in info
        assert "episode_reward" in info
        assert "is_gba" in info
        assert "frame_count" in info
        assert env.episode_steps == 0
        assert env.episode_reward == 0.0

        env.close()

    def test_step_discrete(self, mock_pynds_instance):
        """Test step with discrete actions."""
        env = PyNDSGymEnv("test.nds", action_type="discrete")

        obs, info = env.reset()
        action = 1  # Press A button
        obs, reward, terminated, truncated, info = env.step(action)

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert env.episode_steps == 1

        env.close()

    def test_step_multi_discrete(self, mock_pynds_instance):
        """Test step with multi-discrete actions."""
        env = PyNDSGymEnv("test.nds", action_type="multi_discrete")

        obs, info = env.reset()
        action = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Press A button
        obs, reward, terminated, truncated, info = env.step(action)

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert env.episode_steps == 1

        env.close()

    def test_step_continuous(self, mock_pynds_instance):
        """Test step with continuous actions."""
        env = PyNDSGymEnv("test.nds", action_type="continuous")

        obs, info = env.reset()
        action = np.random.random(14)  # Random continuous action
        obs, reward, terminated, truncated, info = env.step(action)

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert env.episode_steps == 1

        env.close()

    def test_episode_termination(self, mock_pynds_instance):
        """Test episode termination conditions."""
        env = PyNDSGymEnv("test.nds", max_episode_steps=5)

        obs, info = env.reset()

        # Run until max steps
        for _ in range(6):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            if truncated:
                break

        assert truncated
        assert env.episode_steps >= 5

        env.close()


class TestPyNDSGymEnvObservationTypes:
    """Test different observation types."""

    def test_rgb_observation(self, mock_pynds_instance):
        """Test RGB observation processing."""
        env = PyNDSGymEnv("test.nds", observation_type="rgb")

        obs, _ = env.reset()

        assert obs.shape == (512, 192, 3)  # NDS stacked RGB
        assert obs.dtype == np.uint8
        assert obs.min() >= 0
        assert obs.max() <= 255

        env.close()

    def test_grayscale_observation(self, mock_pynds_instance):
        """Test grayscale observation processing."""
        env = PyNDSGymEnv("test.nds", observation_type="grayscale")

        obs, _ = env.reset()

        assert obs.shape == (512, 192, 1)  # NDS grayscale
        assert obs.dtype == np.uint8
        assert obs.min() >= 0
        assert obs.max() <= 255

        env.close()

    def test_raw_observation(self, mock_pynds_instance):
        """Test raw observation processing."""
        env = PyNDSGymEnv("test.nds", observation_type="raw")

        obs, _ = env.reset()

        assert obs.shape == (512, 192, 4)  # NDS raw RGBA
        assert obs.dtype == np.uint8

        env.close()

    def test_gba_observation(self, mock_pynds_instance):
        """Test GBA observation processing."""
        mock_pynds_instance.is_gba = True
        mock_pynds_instance.get_frame.return_value = np.random.randint(
            0, 255, (160, 240, 4), dtype=np.uint8
        )

        env = PyNDSGymEnv("test.gba", observation_type="rgb")

        obs, _ = env.reset()

        assert obs.shape == (160, 240, 3)  # GBA RGB
        assert obs.dtype == np.uint8

        env.close()


class TestPyNDSGymEnvRewardFunctions:
    """Test different reward functions."""

    def test_custom_reward(self, mock_pynds_instance):
        """Test custom reward function."""
        env = PyNDSGymEnv("test.nds", reward_type="custom")

        obs, _ = env.reset()
        obs, reward, _, _, _ = env.step(0)

        assert isinstance(reward, float)
        assert reward >= 0.0  # Custom reward should be non-negative

        env.close()

    def test_frame_diff_reward(self, mock_pynds_instance):
        """Test frame difference reward function."""
        env = PyNDSGymEnv("test.nds", reward_type="frame_diff")

        obs, _ = env.reset()
        obs, reward, _, _, _ = env.step(0)

        assert isinstance(reward, float)
        assert reward >= 0.0

        env.close()

    def test_no_reward(self, mock_pynds_instance):
        """Test no reward function."""
        env = PyNDSGymEnv("test.nds", reward_type="none")

        obs, _ = env.reset()
        obs, reward, _, _, _ = env.step(0)

        assert reward == 0.0

        env.close()

    def test_invalid_reward_type(self, mock_pynds_instance):
        """Test invalid reward type."""
        with pytest.raises(ValueError, match="Unsupported reward_type"):
            PyNDSGymEnv("test.nds", reward_type="invalid")


class TestPyNDSGymEnvRendering:
    """Test rendering functionality."""

    def test_render_human_mode(self, mock_pynds_instance):
        """Test human rendering mode."""
        env = PyNDSGymEnv("test.nds", render_mode="human")

        obs, _ = env.reset()
        result = env.render()

        assert result is None  # Human mode returns None
        mock_pynds_instance.window.render.assert_called()

        env.close()

    def test_render_rgb_array_mode(self, mock_pynds_instance):
        """Test RGB array rendering mode."""
        env = PyNDSGymEnv("test.nds", render_mode="rgb_array")

        obs, _ = env.reset()
        result = env.render()

        assert isinstance(result, np.ndarray)
        assert result.shape == (512, 192, 3)

        env.close()

    def test_render_none_mode(self, mock_pynds_instance):
        """Test no rendering mode."""
        env = PyNDSGymEnv("test.nds", render_mode=None)

        obs, _ = env.reset()
        result = env.render()

        assert result is None

        env.close()


class TestCustomPyNDSGymEnv:
    """Test custom Gym environment with enhanced rewards."""

    def test_custom_env_initialization(self, mock_pynds_instance):
        """Test custom environment initialization."""
        env = CustomPyNDSGymEnv("test.nds")

        assert hasattr(env, "reward_history")
        assert hasattr(env, "last_memory_values")
        assert hasattr(env, "progress_tracker")
        assert hasattr(env, "survival_steps")
        assert hasattr(env, "exploration_map")

        env.close()

    def test_custom_reward_components(self, mock_pynds_instance):
        """Test custom reward components."""
        env = CustomPyNDSGymEnv("test.nds")

        obs, _ = env.reset()
        obs, reward, _, _, _ = env.step(0)

        assert isinstance(reward, float)
        assert len(env.reward_history) == 1

        # Check reward components
        reward_components = env.reward_history[0]
        expected_components = [
            "total",
            "memory",
            "progress",
            "survival",
            "exploration",
            "frame",
        ]
        for component in expected_components:
            assert component in reward_components

        env.close()

    def test_reward_analysis(self, mock_pynds_instance):
        """Test reward analysis functionality."""
        env = CustomPyNDSGymEnv("test.nds")

        # Run a few steps
        obs, _ = env.reset()
        for _ in range(5):
            obs, reward, _, _, _ = env.step(0)

        analysis = env.get_reward_analysis()

        assert isinstance(analysis, dict)
        assert "total" in analysis
        assert "mean" in analysis["total"]
        assert "std" in analysis["total"]
        assert "min" in analysis["total"]
        assert "max" in analysis["total"]

        env.close()


class TestPyNDSGymEnvIntegration:
    """Test integration scenarios."""

    def test_gym_compatibility(self, mock_pynds_instance):
        """Test full Gym interface compatibility."""
        env = PyNDSGymEnv("test.nds")

        # Test all required Gym methods
        obs, info = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # Test optional methods
        env.render()
        env.close()

        # Test metadata
        assert hasattr(env, "metadata")
        assert "render_modes" in env.metadata
        assert "render_fps" in env.metadata

        # Test action meanings
        meanings = env.get_action_meanings()
        assert isinstance(meanings, list)
        assert len(meanings) > 0

    def test_episode_info_tracking(self, mock_pynds_instance):
        """Test episode information tracking."""
        env = PyNDSGymEnv("test.nds")

        obs, info = env.reset()

        # Run a few steps
        for _ in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

        # Check episode info
        episode_info = env.get_episode_info()
        assert episode_info["episode_steps"] == 3
        assert episode_info["episode_reward"] >= 0
        assert "is_gba" in episode_info
        assert "frame_count" in episode_info

        env.close()

    def test_frame_skip_functionality(self, mock_pynds_instance):
        """Test frame skip functionality."""
        env = PyNDSGymEnv("test.nds", frame_skip=4)

        obs, _ = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # Frame skip should call tick() multiple times
        assert mock_pynds_instance.tick.call_count == 4

        env.close()

    def test_seed_functionality(self, mock_pynds_instance):
        """Test seed functionality for reproducibility."""
        env = PyNDSGymEnv("test.nds")

        # Test seeding
        env.seed(42)

        # Should not raise any errors
        obs, _ = env.reset(seed=42)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        env.close()

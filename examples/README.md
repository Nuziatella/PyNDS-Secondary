# PyNDS Examples

Welcome to the PyNDS examples directory! This collection of examples demonstrates how to use PyNDS for reinforcement learning, showcasing integration with popular RL frameworks and advanced techniques.

## Quick Start

### Basic Gym Usage
```python
from pynds.gym_env import PyNDSGymEnv

# Create environment
env = PyNDSGymEnv("game.nds", render_mode="human")

# Standard Gym interface
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, done, truncated, info = env.step(action)
env.close()
```

### Stable-Baselines3 Training
```python
from stable_baselines3 import PPO
from pynds.gym_env import PyNDSGymEnv

env = PyNDSGymEnv("game.nds")
model = PPO("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
model.save("pynds_agent")
```

## Example Files

### `basic_gym_usage.py`
**Perfect for beginners!** Demonstrates the basic Gym interface with PyNDS:
- Environment creation and configuration
- Different action spaces (discrete, multi-discrete, continuous)
- Different observation types (RGB, grayscale, raw)
- Basic episode execution

**Run it:**
```bash
python examples/basic_gym_usage.py
```

### `stable_baselines3_training.py`
**For RL practitioners!** Complete training pipeline with Stable-Baselines3:
- PPO and DQN agent training
- Model evaluation and saving
- Training with periodic evaluation
- Tensorboard logging integration

**Run it:**
```bash
python examples/stable_baselines3_training.py
```

### `custom_reward_functions.py`
**For advanced users!** Custom reward function development:
- Memory-based rewards (tracking game state)
- Progress-based rewards (measuring advancement)
- Survival and exploration rewards
- Game-specific reward functions (Pokemon, Mario, Zelda)

**Run it:**
```bash
python examples/custom_reward_functions.py
```

## Supported Games

PyNDS works with any Nintendo DS or Game Boy Advance ROM, but these examples are optimized for:

- **Pokemon Games**: Experience gain, Pokemon catching rewards
- **Mario Games**: Coin collection, score-based rewards
- **Zelda Games**: Rupee collection, health management
- **Any Game**: Customizable reward functions for your specific needs

## Prerequisites

### Required Dependencies
```bash
pip install pynds
pip install gym
pip install numpy
pip install matplotlib
```

### For Stable-Baselines3 Examples
```bash
pip install stable-baselines3[extra]
pip install tensorboard
```

### For Custom Reward Functions
```bash
pip install pynds  # Already includes memory access
```

## Action Spaces

### Discrete Actions (9 actions)
- `0`: No-op
- `1`: A button
- `2`: B button
- `3`: Start
- `4`: Select
- `5-8`: D-pad directions

### Multi-Discrete Actions (12 buttons)
- `[A, B, X, Y, L, R, Start, Select, Up, Down, Left, Right]`
- Each button can be pressed (1) or not pressed (0)

### Continuous Actions (14 values)
- `[0-11]`: Button pressures (0.0 to 1.0)
- `[12-13]`: Touch screen coordinates (0.0 to 1.0)

## Observation Spaces

### RGB Observations
- **NDS**: `(480, 256, 3)` - Dual screen stacked vertically
- **GBA**: `(240, 160, 3)` - Single screen

### Grayscale Observations
- **NDS**: `(480, 256, 1)` - Dual screen grayscale
- **GBA**: `(240, 160, 1)` - Single screen grayscale

### Raw Observations
- **NDS**: `(480, 256, 4)` - Dual screen with alpha channel
- **GBA**: `(240, 160, 4)` - Single screen with alpha channel

## Reward Functions

### Built-in Reward Types
- **`custom`**: Frame change-based rewards (default)
- **`frame_diff`**: Raw frame difference rewards
- **`none`**: No rewards (for custom implementations)

### Custom Reward Components
- **Memory-based**: Track specific game values (score, lives, level)
- **Progress-based**: Reward for game advancement
- **Survival**: Reward for staying alive
- **Exploration**: Reward for discovering new states

## Configuration Examples

### Basic Configuration
```python
env = PyNDSGymEnv(
    rom_path="game.nds",
    action_type="discrete",
    observation_type="rgb",
    frame_skip=1,
    max_episode_steps=10000,
    render_mode="human"
)
```

### RL Training Configuration
```python
env = PyNDSGymEnv(
    rom_path="game.nds",
    action_type="discrete",
    observation_type="grayscale",  # Better for RL
    frame_skip=4,                  # Faster training
    max_episode_steps=5000,
    render_mode=None               # Headless training
)
```

### Advanced Configuration
```python
env = CustomPyNDSGymEnv(
    rom_path="game.nds",
    action_type="multi_discrete",
    observation_type="rgb",
    frame_skip=2,
    max_episode_steps=10000,
    reward_type="custom"
)
```

## Troubleshooting

### Common Issues

**ROM file not found:**
- Make sure you have a valid `.nds` or `.gba` file
- Update the `rom_path` in the examples
- Check file permissions

**Import errors:**
- Install missing dependencies: `pip install stable-baselines3[extra]`
- Make sure PyNDS is properly installed: `pip install pynds`

**Memory access errors:**
- Custom reward functions require game-specific memory addresses
- Update the `target_memory_addresses` in `CustomPyNDSGymEnv`
- Use a memory editor to find the correct addresses

**Performance issues:**
- Use `frame_skip > 1` for faster training
- Use grayscale observations for better performance
- Set `render_mode=None` for headless training

### Getting Help

1. Check the [main README](../README.md) for installation instructions
2. Look at the [test files](../tests/) for usage examples
3. Check the [PyNDS documentation](../docs/) for API reference
4. Open an issue on GitHub for bugs or feature requests

## Next Steps

1. **Start with `basic_gym_usage.py`** to understand the interface
2. **Try `stable_baselines3_training.py`** for RL training
3. **Experiment with `custom_reward_functions.py`** for your specific game
4. **Create your own reward functions** based on your game's mechanics
5. **Share your results** with the community!

## Additional Resources

- [PyNDS Documentation](../README.md)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [OpenAI Gym Documentation](https://gym.openai.com/)
- [Reinforcement Learning Resources](https://spinningup.openai.com/)

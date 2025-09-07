# PyNDS: Your Digital Time Machine to the Nintendo DS Universe

Welcome to **PyNDS** - the Python interface that brings Nintendo DS and Game Boy Advance emulation to your fingertips! Built on [NooDS](https://github.com/Hydr8gon/NooDS) emulator and inspired by [PyBoy](https://github.com/Baekalfen/PyBoy), PyNDS is your gateway to digital nostalgia, reinforcement learning adventures, and automated gaming magic.

Think of it as having a Nintendo DS that lives inside your Python code and does whatever you tell it to do. Perfect for AI researchers, bot developers, and anyone who wants to explore the digital worlds of the past with the power of modern Python!

## What Makes PyNDS Special?

- **Dual Platform Support**: Nintendo DS (.nds) and Game Boy Advance (.gba) ROMs
- **RL-Ready**: Designed specifically for reinforcement learning and AI research
- **Full Control**: Button input, memory access, and state management
- **Visual Magic**: Dual-screen NDS display and single-screen GBA nostalgia
- **High Performance**: C++ backend with Python convenience
- **Personality**: Because emulation should be fun, not boring!

## Quick Start: Your First Digital Adventure


```python
import pynds

# Load your digital treasure (LEGALLY OPTAINED ROM file)
nds = pynds.PyNDS("pokemon.nds")  # Auto-detects NDS format
# or
gba = pynds.PyNDS("game.gba")     # Auto-detects GBA format

# Run the emulation for one frame (watch the magic happen!)
nds.tick()

# Capture the digital moment (dual screens for NDS!)
top_frame, bottom_frame = nds.get_frame()

# For GBA, you get a single screen of pure retro magic
frame = gba.get_frame()

# Press some buttons
nds.button.press_key('a')
nds.button.set_touch(100, 150)  # Point at the touch screen
nds.button.touch()              # Make contact with the digital world

# Read the game's thoughts (memory access)
value = nds.memory.read_ram_u32(0x02000000)
nds.memory.write_ram_u32(0x02000000, 0x12345678)  # Rewrite memory!

# Save your progress
nds.save_state_to_file("checkpoint.dat")
nds.load_state_from_file("checkpoint.dat")
```

## Reinforcement Learning Integration

PyNDS is designed to play nicely with your favorite RL frameworks:

```python
import pynds
import gym
import numpy as np

class PyNDSEnv(gym.Env):
    """A Gym environment for Nintendo DS games - because AI needs to play too!"""

    def __init__(self, rom_path):
        super().__init__()
        self.nds = pynds.PyNDS(rom_path)
        self.action_space = gym.spaces.Discrete(12)  # 12 buttons
        self.observation_space = gym.spaces.Box(0, 255, (192, 256, 4), dtype=np.uint8)

    def step(self, action):
        # Convert action to button press
        button_map = ['a', 'b', 'select', 'start', 'right', 'left', 'up', 'down', 'r', 'l', 'x', 'y']
        if action < 12:
            self.nds.button.press_key(button_map[action])

        # Run emulation
        self.nds.tick()

        # Get observation (top screen)
        top_frame, _ = self.nds.get_frame()

        # Your reward logic here (the AI's motivation!)
        reward = self.calculate_reward(top_frame)

        return top_frame, reward, False, {}

    def reset(self):
        # Reset to beginning or load save state
        self.nds.load_state_from_file("start_state.dat")
        top_frame, _ = self.nds.get_frame()
        return top_frame

# Now train your AI to become a Nintendo DS master!
env = PyNDSEnv("your_game.nds")
```

## Installation: Bringing the Magic to Your Machine

### Option 1: The Easy Way (PyPI)
```bash
pip install pynds
```

### Option 2: Build from Source (For the Adventurous)
```bash
# Clone the repository (get the source code)
git clone https://github.com/unexploredtest/PyNDS.git
cd PyNDS

# Initialize the submodules (get the emulator engine)
git submodule update --init --recursive

# Build and install (compile the magic)
python setup.py install
```

**System Requirements:**
- Python 3.9+ (because we like modern Python!)
- CMake (for building the C++ magic)
- A C++ compiler (your system's digital blacksmith)
- Some ROM files (your digital adventures)

## Features: The Digital Arsenal

### Core Emulation
- **Dual-screen NDS support** - Experience games as they were meant to be played
- **Single-screen GBA support** - Pure retro magic in one screen
- **Automatic format detection** - We're smart enough to figure out your ROMs
- **High-resolution rendering** - Because pixels deserve to look good

### Input Control
- **Button simulation** - Press any button with code
- **Touch screen support** - Point and click in the digital world
- **Keyboard mapping** - Use your computer keyboard as a gamepad

### Memory Access
- **Read/write RAM** - Peek into the game's thoughts
- **Multiple data types** - u8, u16, u32, u64, i8, i16, i32, i64, f32, f64
- **Memory mapping** - Download chunks of digital consciousness

### State Management
- **Save states** - Create checkpoints in time
- **Load states** - Travel back to any moment
- **Game saves** - Save the traditional way too

### Visual Display
- **Pygame integration** - Watch your games in real-time
- **Automatic scaling** - Fits any window size
- **Dual-screen layout** - NDS games displayed properly

## Advanced Usage: Becoming a Digital Wizard

### Context Manager Magic
```python
# Automatic cleanup (no memory leaks on your watch!)
with pynds.PyNDS("game.nds") as nds:
    nds.tick()
    frame = nds.get_frame()
    # Automatically cleaned up when exiting the 'with' block
```

### Window Display
```python
# Open your portal to the digital world
nds.open_window(1024, 768)  # Custom size
nds.render()  # Watch the magic happen frame by frame
```

### Memory Hacking Adventures (not actual ram values!)
```python
# Read the game's health value (if you know where it is!)
health = nds.memory.read_ram_u16(0x02000000)

# Give yourself infinite lives (use responsibly!)
nds.memory.write_ram_u16(0x02000000, 999)

# Download a chunk of the game's memory
memory_chunk = nds.memory.read_map(0x02000000, 0x02001000)
```

## Troubleshooting: When the Digital World Gets Glitchy

**"My ROM won't load!"**
- Make sure it's a valid .nds or .gba file
- Check that the file path is correct
- Try a different ROM to test if the issue is file-specific

**"The emulator crashes!"**
- Ensure you have the latest version
- Check that all dependencies are installed
- Try running with a verified ROM first in the stand-alone NooDS emulator

**"Performance is slow!"**
- Close other applications
- Try reducing the window size
- Check if your system meets the requirements

## Contributing: Join the Digital Revolution!

We love contributors! Whether you're fixing bugs, adding features, or just improving documentation, your help makes PyNDS better for everyone.

**How to contribute:**
1. Fork the repository
2. Create a feature branch
3. Make your changes (with tests if possible!)
4. Submit a pull request
5. Celebrate your contribution to digital history!

## License & Credits

PyNDS is built on the amazing work of:
- [NooDS](https://github.com/Hydr8gon/NooDS) - The emulation engine
- [PyBoy](https://github.com/Baekalfen/PyBoy) - The inspiration
- [nanobind](https://github.com/wjakob/nanobind) - The Python-C++ bridge

## ⚠️ Legal Disclaimer

**Important:** PyNDS is for educational and research purposes. Make sure you own the ROM files you're using, and respect the intellectual property of game developers. We're here to explore and learn, not to pirate digital adventures!

---

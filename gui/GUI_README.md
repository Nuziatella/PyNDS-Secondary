# PyNDS Simple GUI

A GUI application that showcases all PyNDS functionality, including emulation, state management, frame control, and screen export. Perfect for testing the wheel in production!

### **Method 1: Using the Launcher (Recommended)**
```bash
# Navigate to gui directory
cd gui

# Run the launcher (checks dependencies and installs if needed)
python run_gui.py
```

### **Method 2: Manual Setup**
```bash
# Install dependencies
pip install -r gui/gui_requirements.txt

# Install PyNDS wheel
pip install dist/pynds-*.whl

# Run the GUI
python gui/simple_gui.py
```

## Requirements

### System Requirements
- Python 3.8 or higher
- Windows, macOS, or Linux
- At least 4GB RAM (for emulation)
- Graphics card with OpenGL support

### Python Dependencies
- `pynds` - Core PyNDS package (from wheel)
- `Pillow` - Image processing
- `numpy` - Scientific computing
- `tkinter` - GUI framework (usually included with Python)

## Usage Guide

### 1. Loading a ROM
1. Click "Load ROM" button
2. Select a `.nds` or `.gba` file
3. Wait for ROM to load
4. Check status in the display panel

### 2. Basic Emulation
1. Click "Start Emulation" to begin continuous emulation
2. Use "Stop Emulation" to pause
3. Use "Reset" to restart the game
4. Watch the live display in the center panel

### 3. Frame Control
1. Use "Step (1 frame)" for precise control
2. Use "Step (10 frames)" or "Step (60 frames)" for faster advancement
3. Use "Run 1 second" or "Run 5 seconds" for time-based control
4. Monitor frame count in the display panel

### 4. State Management
1. Click "Save State" to save current state to memory
2. Click "Save State to File" to save to disk
3. Click "Load State from File" to restore from disk
4. Check logs for state size and validation info

### 5. Screen Export
1. Click "Export Current Frame" to save current frame
2. Click "Export 10 Frames" to save multiple frames
3. Click "Get Frame as Image" to get PIL Image object
4. Check logs for export status

### 6. Button Input
1. Click any button in the "Button Input" section
2. Buttons will be pressed in the emulation
3. Check logs for button press confirmation
4. Use for testing game controls

### 7. Memory Access
1. Enter hexadecimal address in "Address" field (e.g., 0x02000000)
2. Click "Read U32" to read 32-bit value
3. Enter value in "Value" field
4. Click "Write U32" to write value
5. Check logs for memory operations

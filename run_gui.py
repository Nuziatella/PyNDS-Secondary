#!/usr/bin/env python3
"""PyNDS GUI Launcher.

Convenience script to launch the PyNDS GUI from the project root.
This script simply redirects to the GUI in the gui/ directory.

Usage:
    python run_gui.py
"""

import os
import sys
from pathlib import Path

# Add the gui directory to the Python path
gui_dir = Path(__file__).parent / "gui"
sys.path.insert(0, str(gui_dir))

# Change to the gui directory
os.chdir(gui_dir)

# Import and run the GUI
if __name__ == "__main__":
    try:
        from run_gui import main

        main()
    except ImportError as e:
        print(f"Failed to import GUI: {e}")
        print("   Make sure you're running from the project root directory")
        sys.exit(1)
    except Exception as e:
        print(f"GUI error: {e}")
        sys.exit(1)

"""PyNDS GUI Launcher.

This script launches the PyNDS GUI application with dependency checking.
It ensures all required packages are available before starting the GUI.

Run with: python gui/run_gui.py
"""

import importlib
import subprocess
import sys


def check_dependency(package_name, import_name=None):
    """Check if a package is available."""
    if import_name is None:
        import_name = package_name

    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False


def install_package(package_name):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        return False


def main():
    """Run the GUI launcher."""
    print("PyNDS GUI Launcher")
    print("=" * 40)

    # Check Python version
    if sys.version_info < (3, 8):
        print("Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False

    print(f"Python version: {sys.version.split()[0]}")

    # Required packages
    required_packages = [
        ("pynds", "pynds"),
        ("Pillow", "PIL"),
        ("numpy", "numpy"),
    ]

    missing_packages = []

    # Check each package
    for package_name, import_name in required_packages:
        if check_dependency(package_name, import_name):
            print(f"{package_name}: true")
        else:
            print(f"{package_name}: false")
            missing_packages.append(package_name)

    # Install missing packages
    if missing_packages:
        print(f"\nInstalling missing packages: {', '.join(missing_packages)}")

        for package in missing_packages:
            print(f"Installing {package}...")
            if install_package(package):
                print(f"{package} installed successfully")
            else:
                print(f"Failed to install {package}")
                print(f"   Please install manually: pip install {package}")
                return False

    # Check if PyNDS wheel is installed
    try:
        import pynds

        print(f"PyNDS version: {getattr(pynds, '__version__', 'unknown')}")
    except ImportError:
        print("PyNDS: false")
        print("   Please install the PyNDS wheel first:")
        print("   pip install dist/pynds-*.whl")
        return False

    # Launch GUI
    print("\nStarting PyNDS GUI...")
    print("=" * 40)

    try:
        # Import and run the GUI
        from simple_gui import main as gui_main

        gui_main()
    except ImportError as e:
        print(f"Failed to import GUI: {e}")
        print("   Make sure you're running from the gui directory")
        return False
    except Exception as e:
        print(f"GUI error: {e}")
        return False

    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)

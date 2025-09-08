"""Simple PyNDS GUI Application.

A comprehensive GUI application that showcases all PyNDS functionality
from Roadmaps 1-3, including emulation, state management, frame control,
and screen export. Perfect for testing the wheel in production!

Features:
- ROM loading and emulation display
- Button input controls
- Memory access and manipulation
- State save/load functionality
- Frame control and export
- Real-time emulation controls

Run with: python examples/simple_gui.py
"""

import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

import numpy as np
from PIL import Image, ImageTk

# Import PyNDS components
from pynds import PyNDS


class PyNDSGUI:
    """Simple GUI application for PyNDS testing and demonstration."""

    def __init__(self, root):
        """Initialize the GUI application."""
        self.root = root
        self.root.title("PyNDS - Nintendo DS Emulator")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f4f8")
        self.root.minsize(1000, 700)

        # PyNDS instance
        self.pynds = None
        self.emulation_running = False
        self.emulation_thread = None

        # GUI state
        self.current_rom_path = None
        self.frame_image = None

        # Button state tracking for holdable buttons
        self.button_states = {}
        self.button_widgets = {}
        self.last_frame_data = None

        # Initialize UI elements that will be created later
        self.rom_label = None

        # Display scaling
        self.display_scale = 1

        # Create GUI
        self.create_widgets()
        self.setup_styles()

        # Start GUI update loop
        self.update_gui()

    def create_widgets(self):
        """Create all GUI widgets."""
        # Create menu bar
        self.create_menu_bar()

        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=8, pady=8)

        # Configure grid weights
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # Left panel - Input controls
        self.create_input_panel(main_frame)

        # Center panel - Display
        self.create_display_panel(main_frame)

        # Bottom panel - Info and memory
        self.create_bottom_panel(main_frame)

    def create_menu_bar(self):
        """Create the menu bar with all controls."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load ROM...", command=self.load_rom)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)

        # Emulation menu
        emulation_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Emulation", menu=emulation_menu)
        emulation_menu.add_command(
            label="Start Emulation", command=self.start_emulation
        )
        emulation_menu.add_command(label="Stop Emulation", command=self.stop_emulation)
        emulation_menu.add_separator()
        emulation_menu.add_command(label="Reset", command=self.reset_emulation)

        # State menu
        state_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="State", menu=state_menu)
        state_menu.add_command(label="Save State", command=self.save_state)
        state_menu.add_command(label="Load State", command=self.load_state)
        state_menu.add_separator()
        state_menu.add_command(
            label="Save State to File...", command=self.save_state_to_file
        )
        state_menu.add_command(
            label="Load State from File...", command=self.load_state_from_file
        )

        # Frame menu
        frame_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Frame", menu=frame_menu)
        frame_menu.add_command(label="Step 1 Frame", command=self.step_frame)
        frame_menu.add_command(label="Step 10 Frames", command=self.step_10_frames)
        frame_menu.add_command(label="Run 1 Second", command=self.run_1_second)
        frame_menu.add_separator()
        frame_menu.add_command(
            label="Export Current Frame...", command=self.export_current_frame
        )
        frame_menu.add_command(
            label="Export 10 Frames...", command=self.export_10_frames
        )

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(
            label="1x (Native)", command=lambda: self.set_display_scale(1)
        )
        view_menu.add_command(label="2x", command=lambda: self.set_display_scale(2))
        view_menu.add_command(label="3x", command=lambda: self.set_display_scale(3))
        view_menu.add_separator()
        view_menu.add_command(label="Reset Display", command=self.reset_display)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)

    def create_input_panel(self, parent):
        """Create the input panel with buttons and touch controls."""
        input_frame = ttk.LabelFrame(parent, text="Input Controls", padding="8")
        input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 8))

        # Button controls
        button_frame = ttk.LabelFrame(input_frame, text="Buttons", padding="6")
        button_frame.pack(fill="x", pady=(0, 8))

        # Create button grid
        buttons = [
            ("Up", "up"),
            ("Down", "down"),
            ("Left", "left"),
            ("Right", "right"),
            ("A", "a"),
            ("B", "b"),
            ("X", "x"),
            ("Y", "y"),
            ("L", "l"),
            ("R", "r"),
            ("Start", "start"),
            ("Select", "select"),
        ]

        for i, (label, key) in enumerate(buttons):
            btn = ttk.Button(button_frame, text=label, width=8)
            btn.grid(row=i // 4, column=i % 4, padx=2, pady=2)

            # Store button widget for state tracking
            self.button_widgets[key] = btn

            # Bind mouse events for holdable buttons
            btn.bind("<Button-1>", lambda e, k=key: self.on_button_press(k))
            btn.bind("<ButtonRelease-1>", lambda e, k=key: self.on_button_release(k))
            btn.bind("<Leave>", lambda e, k=key: self.on_button_leave(k))

            # Initialize button state
            self.button_states[key] = False

        # Touch screen controls
        touch_frame = ttk.LabelFrame(input_frame, text="Touch Screen", padding="6")
        touch_frame.pack(fill="both", expand=True)

        # Touch instructions
        ttk.Label(
            touch_frame,
            text="Click and drag on the display\nto use touch screen",
            font=("Segoe UI", 9),
            justify="center",
        ).pack(pady=(0, 8))

        # Touch status
        self.touch_status = ttk.Label(
            touch_frame, text="Touch: Inactive", font=("Segoe UI", 9)
        )
        self.touch_status.pack()

    def create_bottom_panel(self, parent):
        """Create the bottom panel with info and memory access."""
        bottom_frame = ttk.Frame(parent)
        bottom_frame.grid(
            row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(8, 0)
        )

        # Left side - Info panel
        info_frame = ttk.LabelFrame(bottom_frame, text="Information", padding="6")
        info_frame.pack(side="left", fill="both", expand=True, padx=(0, 4))

        # Log text area
        self.log_text = scrolledtext.ScrolledText(info_frame, height=8, width=50)
        self.log_text.pack(fill="both", expand=True, pady=(0, 8))

        # ROM status
        self.rom_label = ttk.Label(
            info_frame, text="No ROM loaded", font=("Segoe UI", 9), foreground="gray"
        )
        self.rom_label.pack(anchor="w", pady=(0, 4))

        # System info
        self.system_info = ttk.Label(
            info_frame, text="No emulator loaded", font=("Segoe UI", 9)
        )
        self.system_info.pack(anchor="w")

        # Right side - Memory access
        memory_frame = ttk.LabelFrame(bottom_frame, text="Memory Access", padding="6")
        memory_frame.pack(side="right", fill="y", padx=(4, 0))

        # Memory address input
        addr_frame = ttk.Frame(memory_frame)
        addr_frame.pack(fill="x", pady=(0, 6))

        ttk.Label(addr_frame, text="Address:").pack(side="left")
        self.memory_address = ttk.Entry(addr_frame, width=12)
        self.memory_address.pack(side="left", padx=(4, 0))
        self.memory_address.insert(0, "0x02000000")

        # Memory value input
        value_frame = ttk.Frame(memory_frame)
        value_frame.pack(fill="x", pady=(0, 6))

        ttk.Label(value_frame, text="Value:").pack(side="left")
        self.memory_value = ttk.Entry(value_frame, width=12)
        self.memory_value.pack(side="left", padx=(4, 0))
        self.memory_value.insert(0, "0x12345678")

        # Memory buttons
        mem_btn_frame = ttk.Frame(memory_frame)
        mem_btn_frame.pack(fill="x")

        ttk.Button(mem_btn_frame, text="Read", command=self.read_memory).pack(
            side="left", padx=(0, 4)
        )
        ttk.Button(mem_btn_frame, text="Write", command=self.write_memory).pack(
            side="left"
        )

    def toggle_button(self, key):
        """Toggle button state (press if not pressed, release if pressed)."""
        if not self.pynds or key not in self.button_states:
            return

        if not self.button_states[key]:
            self.on_button_press(key)
        else:
            self.on_button_release(key)

    def on_button_press(self, key):
        """Handle button press (mouse down)."""
        if not self.pynds or key not in self.button_states:
            return
        if not self.button_states[key]:
            try:
                self.pynds.button.press_key(key)
                self.button_states[key] = True
                self.log_message(f"Pressed {key.upper()} button")
                if key in self.button_widgets:
                    # Change background color for visual feedback
                    self.button_widgets[key].config(style="Pressed.TButton")
            except Exception as e:
                self.log_message(f"Error pressing {key} button: {str(e)}")

    def on_button_release(self, key):
        """Handle button release (mouse up)."""
        if key not in self.button_states or not self.button_states[key]:
            return
        try:
            self.pynds.button.release_key(key)
            self.button_states[key] = False
            self.log_message(f"Released {key.upper()} button")
            if key in self.button_widgets:
                # Reset to normal style
                self.button_widgets[key].config(style="TButton")
        except Exception as e:
            self.log_message(f"Error releasing {key} button: {str(e)}")

    def on_button_leave(self, key):
        """Handle mouse leaving button while pressed."""
        if key in self.button_states and self.button_states[key]:
            self.on_button_release(key)

    def step_frame(self):
        """Step one frame."""
        if not self.pynds:
            self.log_message("No ROM loaded")
            return
        try:
            self.pynds.step(1)
            self.log_message("Stepped 1 frame")
        except Exception as e:
            self.log_message(f"Error stepping frame: {str(e)}")

    def step_10_frames(self):
        """Step 10 frames."""
        if not self.pynds:
            self.log_message("No ROM loaded")
            return
        try:
            self.pynds.step(10)
            self.log_message("Stepped 10 frames")
        except Exception as e:
            self.log_message(f"Error stepping frames: {str(e)}")

    def run_1_second(self):
        """Run for 1 second."""
        if not self.pynds:
            self.log_message("No ROM loaded")
            return
        try:
            self.pynds.run_seconds(1.0)
            self.log_message("Ran for 1 second")
        except Exception as e:
            self.log_message(f"Error running: {str(e)}")

    def export_current_frame(self):
        """Export current frame."""
        if not self.pynds:
            self.log_message("No ROM loaded")
            return
        try:
            from tkinter import filedialog

            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            )
            if filename:
                self.pynds.export_frame(filename)
                self.log_message(f"Exported frame to {filename}")
        except Exception as e:
            self.log_message(f"Error exporting frame: {str(e)}")

    def export_10_frames(self):
        """Export 10 frames."""
        if not self.pynds:
            self.log_message("No ROM loaded")
            return
        try:
            from tkinter import filedialog

            directory = filedialog.askdirectory()
            if directory:
                self.pynds.export_frames(directory, count=10)
                self.log_message(f"Exported 10 frames to {directory}")
        except Exception as e:
            self.log_message(f"Error exporting frames: {str(e)}")

    def set_display_scale(self, scale):
        """Set the display scale factor."""
        if scale not in [1, 2, 3]:
            self.log_message(f"Invalid scale: {scale}. Must be 1, 2, or 3.")
            return

        self.display_scale = scale
        self.setup_display_mode()
        self.log_message(f"Display scale set to {scale}x")

    def reset_display(self):
        """Reset display to native resolution."""
        self.display_scale = 1
        self.setup_display_mode()
        self.log_message("Display reset to native resolution")

    def show_about(self):
        """Show about dialog."""
        from tkinter import messagebox

        messagebox.showinfo(
            "About PyNDS",
            "PyNDS - Nintendo DS Emulator\n\n"
            "A Python wrapper for NooDS emulator\n"
            "Perfect for AI research and bot development",
        )

    def read_memory(self):
        """Read memory value from address."""
        if not self.pynds:
            self.log_message("No ROM loaded")
            return

        try:
            address_str = self.memory_address.get().strip()
            if address_str.startswith("0x"):
                address = int(address_str, 16)
            else:
                address = int(address_str)

            value = self.pynds.memory.read_ram_u32(address)
            self.memory_value.delete(0, tk.END)
            self.memory_value.insert(0, f"0x{value:08X}")
            self.log_message(f"Read memory 0x{address:08X}: 0x{value:08X}")
        except Exception as e:
            self.log_message(f"Error reading memory: {str(e)}")

    def write_memory(self):
        """Write value to memory address."""
        if not self.pynds:
            self.log_message("No ROM loaded")
            return

        try:
            address_str = self.memory_address.get().strip()
            value_str = self.memory_value.get().strip()

            if address_str.startswith("0x"):
                address = int(address_str, 16)
            else:
                address = int(address_str)

            if value_str.startswith("0x"):
                value = int(value_str, 16)
            else:
                value = int(value_str)

            self.pynds.memory.write_ram_u32(address, value)
            self.log_message(f"Wrote memory 0x{address:08X}: 0x{value:08X}")
        except Exception as e:
            self.log_message(f"Error writing memory: {str(e)}")

    # Old control panel method removed - now using menu bar and input panel

    def create_display_panel(self, parent):
        """Create the display panel for showing emulation."""
        display_frame = ttk.LabelFrame(parent, text="Display", padding="6")
        display_frame.grid(
            row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 8)
        )

        # Canvas for frame display (will resize based on game type)
        self.canvas = tk.Canvas(display_frame, bg="black", width=512, height=384)
        self.canvas.pack(expand=True, fill="both")

        # Display settings
        self.is_gba_mode = False

        # Bind mouse events for touch screen simulation
        self.canvas.bind("<Button-1>", self.on_canvas_click)  # Left click
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)  # Drag
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)  # Release

        # Touch state tracking
        self.touch_active = False
        self.last_touch_x = 0
        self.last_touch_y = 0

        # Status label
        self.status_label = ttk.Label(
            display_frame, text="No ROM loaded", font=("Arial", 12)
        )
        self.status_label.pack(pady=(10, 0))

        # Frame info
        self.frame_info_label = ttk.Label(
            display_frame, text="Frame: 0", font=("Arial", 10)
        )
        self.frame_info_label.pack()

    # Old info panel method removed - now using bottom panel

    def setup_styles(self):
        """Configure custom styles for the GUI."""
        style = ttk.Style()
        style.theme_use("clam")

        # Modern professional blue theme
        style.configure("TLabel", background="#f0f4f8", foreground="#1e3a8a")
        style.configure("TFrame", background="#f0f4f8")
        style.configure(
            "TLabelFrame",
            background="#e0f2fe",
            foreground="#0c4a6e",
            borderwidth=1,
            relief="solid",
        )
        style.configure(
            "TLabelFrame.Label",
            background="#e0f2fe",
            foreground="#0c4a6e",
            font=("Segoe UI", 9, "bold"),
        )
        style.configure(
            "TButton",
            background="#3b82f6",
            foreground="#ffffff",
            borderwidth=1,
            relief="raised",
            padding=(8, 4),
            font=("Segoe UI", 9),
        )
        style.map("TButton", background=[("active", "#2563eb"), ("pressed", "#1d4ed8")])

        # Pressed button style for visual feedback
        style.configure(
            "Pressed.TButton",
            background="#1d4ed8",
            foreground="#ffffff",
            borderwidth=1,
            relief="raised",
            padding=(8, 4),
            font=("Segoe UI", 9),
        )
        style.configure(
            "TEntry",
            fieldbackground="#ffffff",
            foreground="#1e3a8a",
            borderwidth=1,
            relief="solid",
        )
        style.configure(
            "TText",
            background="#ffffff",
            foreground="#1e3a8a",
            borderwidth=1,
            relief="solid",
        )

    def log_message(self, message):
        """Add a message to the log."""
        self.log_text.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def check_pynds_methods(self):
        """Check if PyNDS instance has all required methods."""
        if not self.pynds:
            return False

        required_methods = [
            "tick",
            "get_frame",
            "is_initialized",
            "close",
            "save_state",
            "load_state",
            "get_state_size",
            "validate_state",
            "step",
            "run_seconds",
            "reset",
            "export_frame",
            "export_frames",
            "get_frame_as_image",
        ]

        missing_methods = []
        for method in required_methods:
            if not hasattr(self.pynds, method):
                missing_methods.append(method)

        if missing_methods:
            self.log_message(f"Warning: Missing methods: {', '.join(missing_methods)}")
            return False

        return True

    def load_rom(self):
        """Load a ROM file."""
        file_path = filedialog.askopenfilename(
            title="Select ROM file",
            filetypes=[
                ("Nintendo DS ROMs", "*.nds"),
                ("Game Boy Advance ROMs", "*.gba"),
                ("All ROM files", "*.nds;*.gba"),
                ("All files", "*.*"),
            ],
        )

        if file_path:
            try:
                # Close existing ROM if any
                if self.pynds:
                    self.pynds.close()

                # Load new ROM
                self.pynds = PyNDS(file_path)
                self.current_rom_path = file_path

                # Check if all methods are available
                if not self.check_pynds_methods():
                    self.log_message("Warning: Some PyNDS methods may not be available")

                # Set display mode based on game type
                self.is_gba_mode = self.pynds.is_gba
                self.setup_display_mode()

                # Update GUI
                game_type = "GBA" if self.is_gba_mode else "NDS"
                self.rom_label.config(
                    text=f"Loaded: {Path(file_path).name} ({game_type})"
                )
                self.status_label.config(text="ROM loaded - Ready to emulate")
                self.system_info.config(text=f"Type: {game_type}")

                self.log_message(f"ROM loaded: {Path(file_path).name}")
                self.log_message(f"Emulator type: {game_type}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load ROM: {str(e)}")
                self.log_message(f"Error loading ROM: {str(e)}")

    def close_rom(self):
        """Close the current ROM."""
        if self.pynds:
            self.stop_emulation()
            self.pynds.close()
            self.pynds = None
            self.current_rom_path = None

            # Update GUI
            self.rom_label.config(text="No ROM loaded")
            self.status_label.config(text="No ROM loaded")
            self.system_info.config(text="No emulator loaded")
            self.canvas.delete("all")

            self.log_message("ROM closed")

    def start_emulation(self):
        """Start continuous emulation."""
        if not self.pynds:
            messagebox.showwarning("Warning", "Please load a ROM first")
            return

        if not self.emulation_running:
            self.emulation_running = True
            self.emulation_thread = threading.Thread(
                target=self.emulation_loop, daemon=True
            )
            self.emulation_thread.start()
            self.log_message("Emulation started")

    def stop_emulation(self):
        """Stop continuous emulation."""
        self.emulation_running = False
        self.log_message("Emulation stopped")

    def reset_emulation(self):
        """Reset the emulation."""
        if not self.pynds:
            messagebox.showwarning("Warning", "Please load a ROM first")
            return

        try:
            self.pynds.reset()
            self.log_message("Emulation reset")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to reset: {str(e)}")
            self.log_message(f"Error resetting: {str(e)}")

    def emulation_loop(self):
        """Run the emulation loop in a separate thread."""
        import time

        target_frame_time = 1.0 / 60.0  # 60 FPS target for both emulation and display

        while self.emulation_running and self.pynds:
            try:
                frame_start = time.time()

                # Run one frame at 60 FPS
                self.pynds.tick()

                # Update display at 60 FPS for true native experience
                self.root.after(0, self.update_display)

                # Maintain precise 60 FPS timing
                elapsed = time.time() - frame_start
                sleep_time = max(0, target_frame_time - elapsed)
                time.sleep(sleep_time)

            except Exception as e:
                error_msg = f"Emulation error: {str(e)}"
                self.root.after(0, lambda: self.log_message(error_msg))
                break

    def step_frames(self, count):
        """Step multiple frames."""
        if not self.pynds:
            messagebox.showwarning("Warning", "Please load a ROM first")
            return

        try:
            self.pynds.tick(count)
            self.update_display()
            self.log_message(f"Stepped {count} frames")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to step {count} frames: {str(e)}")
            self.log_message(f"Error stepping {count} frames: {str(e)}")

    def run_seconds(self, seconds):
        """Run emulation for specified seconds."""
        if not self.pynds:
            messagebox.showwarning("Warning", "Please load a ROM first")
            return

        try:
            if not hasattr(self.pynds, "run_seconds"):
                messagebox.showwarning(
                    "Warning", "Run seconds not available in this PyNDS version"
                )
                return

            self.pynds.run_seconds(seconds)
            self.update_display()
            self.log_message(f"Ran for {seconds} seconds")
        except Exception as e:
            messagebox.showerror(
                "Error", f"Failed to run for {seconds} seconds: {str(e)}"
            )
            self.log_message(f"Error running for {seconds} seconds: {str(e)}")

    # Duplicate button methods removed - using the correct ones defined earlier

    def press_button(self, key):
        """Press a button (legacy method for compatibility)."""
        self.on_button_press(key)

    def release_button(self, key):
        """Release a button (legacy method for compatibility)."""
        self.on_button_release(key)

    def on_canvas_click(self, event):
        """Handle mouse click on canvas for touch simulation."""
        if not self.pynds:
            return

        try:
            if self.is_gba_mode:
                # GBA doesn't have touch screen
                self.log_message("Touch not supported for GBA games")
                return
            else:
                # NDS touch screen: 256x192 (bottom screen only)
                # Scale coordinates based on display scale
                scaled_bottom_start = 192 * self.display_scale
                if event.y < scaled_bottom_start:  # Clicked on top screen
                    self.log_message("Touch only works on bottom screen")
                    return

                # Convert scaled coordinates back to native NDS coordinates
                nds_x = int(event.x / self.display_scale)
                nds_y = int((event.y - scaled_bottom_start) / self.display_scale)

            # Clamp to valid range
            nds_x = max(0, min(255, nds_x))
            nds_y = max(0, min(191, nds_y))

            # Set touch coordinates
            self.pynds.button.set_touch(nds_x, nds_y)
            self.pynds.button.touch()

            # Update state
            self.touch_active = True
            self.last_touch_x = nds_x
            self.last_touch_y = nds_y

            # Update UI
            self.touch_status.config(text=f"Touch: Active at ({nds_x}, {nds_y})")
            self.log_message(f"Touch activated at ({nds_x}, {nds_y})")

            # Draw touch indicator on canvas
            self.draw_touch_indicator(nds_x, nds_y)

        except Exception as e:
            self.log_message(f"Error with touch: {str(e)}")

    def on_canvas_drag(self, event):
        """Handle mouse drag on canvas for touch simulation."""
        if not self.pynds or not self.touch_active:
            return

        try:
            if self.is_gba_mode:
                return  # GBA doesn't have touch screen

            # NDS touch screen: 256x192 (bottom screen only)
            # Scale coordinates based on display scale
            scaled_bottom_start = 192 * self.display_scale
            if event.y < scaled_bottom_start:  # Dragging on top screen
                return

            # Convert scaled coordinates back to native NDS coordinates
            nds_x = int(event.x / self.display_scale)
            nds_y = int((event.y - scaled_bottom_start) / self.display_scale)

            # Clamp to valid range
            nds_x = max(0, min(255, nds_x))
            nds_y = max(0, min(191, nds_y))

            # Update touch coordinates
            self.pynds.button.set_touch(nds_x, nds_y)

            # Update state
            self.last_touch_x = nds_x
            self.last_touch_y = nds_y

            # Update UI
            self.touch_status.config(text=f"Touch: Dragging at ({nds_x}, {nds_y})")

            # Update touch indicator on canvas
            self.draw_touch_indicator(nds_x, nds_y)

        except Exception as e:
            self.log_message(f"Error with touch drag: {str(e)}")

    def on_canvas_release(self, event):
        """Handle mouse release on canvas for touch simulation."""
        if not self.pynds or not self.touch_active:
            return

        try:
            # Release touch
            self.pynds.button.release_touch()

            # Update state
            self.touch_active = False

            # Update UI
            self.touch_status.config(text="Touch: Inactive")
            self.log_message(
                f"Touch released at ({self.last_touch_x}, {self.last_touch_y})"
            )

            # Clear touch indicator
            self.clear_touch_indicator()

        except Exception as e:
            self.log_message(f"Error releasing touch: {str(e)}")

    def clear_touch(self):
        """Clear touch screen input."""
        if not self.pynds:
            messagebox.showwarning("Warning", "Please load a ROM first")
            return

        try:
            self.pynds.button.clear_touch()
            self.touch_active = False
            self.touch_status.config(text="Touch: Inactive")
            self.clear_touch_indicator()
            self.log_message("Touch input cleared")
        except Exception as e:
            self.log_message(f"Error clearing touch: {str(e)}")

    def draw_touch_indicator(self, nds_x, nds_y):
        """Draw a visual indicator on the canvas showing touch position."""
        try:
            # Convert NDS coordinates back to canvas coordinates
            # Scale coordinates based on display scale
            canvas_x = nds_x * self.display_scale
            canvas_y = (nds_y + 192) * self.display_scale  # Bottom half starts at y=192

            # Clear previous indicator
            self.canvas.delete("touch_indicator")

            # Scale indicator size with display scale
            size = 8 * self.display_scale
            line_width = max(2, 2 * self.display_scale)

            # Draw touch indicator (circle with crosshair)
            self.canvas.create_oval(
                canvas_x - size,
                canvas_y - size,
                canvas_x + size,
                canvas_y + size,
                outline="red",
                width=line_width,
                tags="touch_indicator",
            )
            self.canvas.create_line(
                canvas_x - size * 1.5,
                canvas_y,
                canvas_x + size * 1.5,
                canvas_y,
                fill="red",
                width=line_width,
                tags="touch_indicator",
            )
            self.canvas.create_line(
                canvas_x,
                canvas_y - size * 1.5,
                canvas_x,
                canvas_y + size * 1.5,
                fill="red",
                width=line_width,
                tags="touch_indicator",
            )
        except Exception as e:
            self.log_message(f"Error drawing touch indicator: {str(e)}")

    def clear_touch_indicator(self):
        """Clear the touch indicator from the canvas."""
        try:
            self.canvas.delete("touch_indicator")
        except Exception as e:
            self.log_message(f"Error clearing touch indicator: {str(e)}")

    def setup_display_mode(self):
        """Configure display mode based on game type and scale factor."""
        try:
            if self.is_gba_mode:
                # GBA: 160x240 native resolution
                base_width = 160
                base_height = 240
                canvas_width = base_width * self.display_scale
                canvas_height = base_height * self.display_scale
                self.log_message(
                    f"Display mode: GBA ({base_width}x{base_height}) at {self.display_scale}x scale = {canvas_width}x{canvas_height}"
                )
            else:
                # NDS: 256x384 native resolution (both screens stacked)
                base_width = 256
                base_height = 384
                canvas_width = base_width * self.display_scale
                canvas_height = base_height * self.display_scale
                self.log_message(
                    f"Display mode: NDS ({base_width}x{base_height}) at {self.display_scale}x scale = {canvas_width}x{canvas_height}"
                )

            # Set scaled canvas size
            self.canvas.config(width=canvas_width, height=canvas_height)

        except Exception as e:
            self.log_message(f"Error setting up display mode: {str(e)}")

    def save_state(self):
        """Save state to memory."""
        if not self.pynds:
            messagebox.showwarning("Warning", "Please load a ROM first")
            return

        try:
            if not hasattr(self.pynds, "save_state"):
                messagebox.showwarning(
                    "Warning", "Save state not available in this PyNDS version"
                )
                return

            state_data = self.pynds.save_state()
            if hasattr(self.pynds, "get_state_size"):
                state_size = self.pynds.get_state_size()
                self.log_message(f"State saved to memory ({state_size} bytes)")
            else:
                self.log_message(f"State saved to memory ({len(state_data)} bytes)")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save state: {str(e)}")
            self.log_message(f"Error saving state: {str(e)}")

    def load_state(self):
        """Load state from memory."""
        if not self.pynds:
            messagebox.showwarning("Warning", "Please load a ROM first")
            return

        try:
            # This would need the state data from save_state
            self.log_message("Load state from memory - requires state data")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load state: {str(e)}")
            self.log_message(f"Error loading state: {str(e)}")

    def save_state_to_file(self):
        """Save state to file."""
        if not self.pynds:
            messagebox.showwarning("Warning", "Please load a ROM first")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save state file",
            defaultextension=".state",
            filetypes=[("State files", "*.state"), ("All files", "*.*")],
        )

        if file_path:
            try:
                self.pynds.save_state_to_file(file_path)
                self.log_message(f"State saved to file: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save state to file: {str(e)}")
                self.log_message(f"Error saving state to file: {str(e)}")

    def load_state_from_file(self):
        """Load state from file."""
        if not self.pynds:
            messagebox.showwarning("Warning", "Please load a ROM first")
            return

        file_path = filedialog.askopenfilename(
            title="Load state file",
            filetypes=[("State files", "*.state"), ("All files", "*.*")],
        )

        if file_path:
            try:
                self.pynds.load_state_from_file(file_path)
                self.update_display()
                self.log_message(f"State loaded from file: {file_path}")
            except Exception as e:
                messagebox.showerror(
                    "Error", f"Failed to load state from file: {str(e)}"
                )
                self.log_message(f"Error loading state from file: {str(e)}")

    def export_frame(self):
        """Export current frame."""
        if not self.pynds:
            messagebox.showwarning("Warning", "Please load a ROM first")
            return

        if not hasattr(self.pynds, "export_frame"):
            messagebox.showwarning(
                "Warning", "Export frame not available in this PyNDS version"
            )
            return

        file_path = filedialog.asksaveasfilename(
            title="Export frame",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*"),
            ],
        )

        if file_path:
            try:
                self.pynds.export_frame(file_path)
                self.log_message(f"Frame exported to: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export frame: {str(e)}")
                self.log_message(f"Error exporting frame: {str(e)}")

    def export_frames(self, count):
        """Export multiple frames."""
        if not self.pynds:
            messagebox.showwarning("Warning", "Please load a ROM first")
            return

        if not hasattr(self.pynds, "export_frames"):
            messagebox.showwarning(
                "Warning", "Export frames not available in this PyNDS version"
            )
            return

        directory = filedialog.askdirectory(title="Select export directory")

        if directory:
            try:
                self.pynds.export_frames(directory, count=count)
                self.log_message(f"Exported {count} frames to: {directory}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export frames: {str(e)}")
                self.log_message(f"Error exporting frames: {str(e)}")

    def get_frame_as_image(self):
        """Get current frame as PIL Image."""
        if not self.pynds:
            messagebox.showwarning("Warning", "Please load a ROM first")
            return

        if not hasattr(self.pynds, "get_frame_as_image"):
            messagebox.showwarning(
                "Warning", "Get frame as image not available in this PyNDS version"
            )
            return

        try:
            image = self.pynds.get_frame_as_image()
            self.log_message(f"Got frame as PIL Image: {image.size}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to get frame as image: {str(e)}")
            self.log_message(f"Error getting frame as image: {str(e)}")

    def read_memory_u32(self):
        """Read 32-bit unsigned integer from memory."""
        if not self.pynds:
            messagebox.showwarning("Warning", "Please load a ROM first")
            return

        try:
            address = int(self.memory_address.get(), 16)
            value = self.pynds.memory.read_ram_u32(address)
            self.memory_value.delete(0, tk.END)
            self.memory_value.insert(0, str(value))
            self.log_message(f"Read U32 from 0x{address:08X}: {value}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read memory: {str(e)}")
            self.log_message(f"Error reading memory: {str(e)}")

    def write_memory_u32(self):
        """Write 32-bit unsigned integer to memory."""
        if not self.pynds:
            messagebox.showwarning("Warning", "Please load a ROM first")
            return

        try:
            address = int(self.memory_address.get(), 16)
            value = int(self.memory_value.get())
            self.pynds.memory.write_ram_u32(address, value)
            self.log_message(f"Wrote U32 to 0x{address:08X}: {value}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to write memory: {str(e)}")
            self.log_message(f"Error writing memory: {str(e)}")

    def update_display(self):
        """Update the display with current frame."""
        if not self.pynds:
            return

        try:
            # Get current frame
            frame_data = self.pynds.get_frame()

            # Convert to displayable format based on game type
            if self.is_gba_mode:
                # GBA: single frame at native 160x240
                display_frame = frame_data
            else:
                # NDS: stack both screens vertically (top + bottom)
                top_frame, bottom_frame = frame_data
                display_frame = np.vstack([top_frame, bottom_frame])  # 256x384 total

            # Convert to PIL Image
            if len(display_frame.shape) == 3 and display_frame.shape[2] == 4:
                # RGBA to RGB
                display_frame = display_frame[:, :, :3]

            image = Image.fromarray(display_frame.astype(np.uint8))

            # Scale image if needed
            if self.display_scale > 1:
                new_width = image.width * self.display_scale
                new_height = image.height * self.display_scale
                image = image.resize((new_width, new_height), Image.Resampling.NEAREST)

            # Convert to PhotoImage
            self.frame_image = ImageTk.PhotoImage(image)

            # Clear canvas and draw image at native resolution
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.frame_image)

            # Draw separator line between top and bottom screens for NDS
            if not self.is_gba_mode:
                separator_y = image.height // 2
                line_width = max(1, self.display_scale)  # Scale line width with display
                self.canvas.create_line(
                    0,
                    separator_y,
                    image.width,
                    separator_y,
                    fill="gray",
                    width=line_width,
                    tags="separator",
                )

            # Redraw touch indicator if active (only for NDS)
            if self.touch_active and not self.is_gba_mode:
                self.draw_touch_indicator(self.last_touch_x, self.last_touch_y)

            # Update frame info
            try:
                frame_count = self.pynds.get_frame_count()
                self.frame_info_label.config(text=f"Frame: {frame_count}")
            except AttributeError:
                # Fallback if get_frame_count is not available
                self.frame_info_label.config(text="Frame: N/A")
            except Exception:
                self.frame_info_label.config(text="Frame: Error")

        except Exception as e:
            self.log_message(f"Error updating display: {str(e)}")

    def update_gui(self):
        """Update GUI elements (called periodically)."""
        # Update status
        if self.pynds:
            if self.emulation_running:
                self.status_label.config(text="Emulation running")
            else:
                self.status_label.config(text="ROM loaded - Ready to emulate")
        else:
            self.status_label.config(text="No ROM loaded")

        # Schedule next update
        self.root.after(100, self.update_gui)

    def on_closing(self):
        """Handle application closing."""
        try:
            # Stop emulation first
            self.emulation_running = False

            # Wait a moment for emulation thread to stop
            if self.emulation_thread and self.emulation_thread.is_alive():
                self.emulation_thread.join(timeout=1.0)

            # Close ROM if loaded with proper cleanup
            if self.pynds:
                try:
                    # Stop any running emulation
                    if hasattr(self.pynds, "close"):
                        self.pynds.close()

                    # Clear references to help with garbage collection
                    if hasattr(self.pynds, "button"):
                        self.pynds.button = None
                    if hasattr(self.pynds, "memory"):
                        self.pynds.memory = None
                    if hasattr(self.pynds, "window"):
                        self.pynds.window = None

                    self.pynds = None
                except Exception as e:
                    print(f"Warning during PyNDS cleanup: {e}")

            # Clear button states
            self.button_states.clear()
            self.button_widgets.clear()

            # Destroy window
            self.root.destroy()
        except Exception as e:
            print(f"Error during cleanup: {e}")
            self.root.destroy()


def main():
    """Run the GUI application."""
    root = tk.Tk()
    app = PyNDSGUI(root)

    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)

    # Start the GUI
    root.mainloop()


if __name__ == "__main__":
    main()

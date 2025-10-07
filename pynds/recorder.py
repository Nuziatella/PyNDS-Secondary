"""PyNDS Recorder - PyBoy-inspired screenshot and recording system.

This module provides advanced screenshot and recording capabilities similar to
PyBoy's recording features, including frame capture, GIF creation, and video
export functionality. Perfect for creating gameplay videos, tutorials, and
documentation!

Classes:
    PyNDSRecorder: Main recording class for screenshots and videos
    FrameCapture: Individual frame capture and processing
    VideoExporter: Video export functionality
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

from .pynds import PyNDS

logger = logging.getLogger(__name__)


class FrameCapture:
    """Individual frame capture and processing utilities.

    Provides methods for capturing, processing, and analyzing individual
    frames from the emulator, similar to PyBoy's frame capture capabilities.
    """

    def __init__(self, pynds: PyNDS):
        """Initialize frame capture.

        Parameters
        ----------
        pynds : PyNDS
            PyNDS emulator instance
        """
        self.pynds = pynds
        self._frame_history: List[np.ndarray] = []
        self._max_history = 1000  # Maximum frames to keep in history

    def capture_frame(self, include_metadata: bool = True) -> Dict[str, Any]:
        """Capture the current frame with optional metadata.

        Parameters
        ----------
        include_metadata : bool, optional
            Whether to include frame metadata, by default True

        Returns
        -------
        Dict[str, Any]
            Frame data with optional metadata
        """
        try:
            frame = self.pynds.get_frame()

            # Convert to single frame if NDS
            if isinstance(frame, tuple):
                top_frame, bottom_frame = frame
                # Combine frames vertically
                combined_frame = np.vstack([top_frame, bottom_frame])
            else:
                combined_frame = frame

            result = {
                "frame": combined_frame,
                "timestamp": time.time(),
                "frame_count": self.pynds.get_frame_count(),
                "platform": self.pynds.get_platform(),
            }

            if include_metadata:
                result.update(
                    {
                        "shape": combined_frame.shape,
                        "dtype": str(combined_frame.dtype),
                        "mean_brightness": float(np.mean(combined_frame)),
                        "fps": self.pynds.get_fps(),
                    }
                )

            # Add to history
            self._frame_history.append(combined_frame.copy())
            if len(self._frame_history) > self._max_history:
                self._frame_history.pop(0)

            return result

        except Exception as e:
            logger.error(f"Frame capture failed: {e}")
            return {"error": str(e)}

    def capture_frame_sequence(
        self, count: int, interval: int = 1
    ) -> List[Dict[str, Any]]:
        """Capture a sequence of frames.

        Parameters
        ----------
        count : int
            Number of frames to capture
        interval : int, optional
            Frames to skip between captures, by default 1

        Returns
        -------
        List[Dict[str, Any]]
            List of captured frame data
        """
        frames = []

        for i in range(count):
            frame_data = self.capture_frame()
            frames.append(frame_data)

            if i < count - 1:  # Don't advance after the last frame
                for _ in range(interval):
                    self.pynds.tick(1)

        return frames

    def get_frame_difference(
        self, frame1: np.ndarray, frame2: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate difference between two frames.

        Parameters
        ----------
        frame1 : np.ndarray
            First frame
        frame2 : np.ndarray
            Second frame

        Returns
        -------
        Dict[str, Any]
            Frame difference analysis
        """
        try:
            # Ensure frames have the same shape
            if frame1.shape != frame2.shape:
                logger.warning("Frames have different shapes, resizing...")
                min_height = min(frame1.shape[0], frame2.shape[0])
                min_width = min(frame1.shape[1], frame2.shape[1])
                frame1 = frame1[:min_height, :min_width]
                frame2 = frame2[:min_height, :min_width]

            # Calculate differences
            diff = np.abs(frame1.astype(float) - frame2.astype(float))

            return {
                "mean_difference": float(np.mean(diff)),
                "max_difference": float(np.max(diff)),
                "changed_pixels": int(np.sum(diff > 0)),
                "total_pixels": int(np.prod(diff.shape)),
                "change_percentage": float(
                    np.sum(diff > 0) / np.prod(diff.shape) * 100
                ),
                "difference_image": diff.astype(np.uint8),
            }
        except Exception as e:
            logger.error(f"Frame difference calculation failed: {e}")
            return {"error": str(e)}

    def clear_history(self) -> None:
        """Clear frame history."""
        self._frame_history.clear()

    def get_history(self) -> List[np.ndarray]:
        """Get frame history.

        Returns
        -------
        List[np.ndarray]
            List of captured frames
        """
        return self._frame_history.copy()


class VideoExporter:
    """Video export functionality for PyNDS recordings.

    Provides methods for exporting recorded frames as video files,
    similar to PyBoy's video export capabilities.
    """

    def __init__(self):
        """Initialize video exporter."""
        self._supported_formats = ["gif", "mp4", "avi"]

    def export_gif(
        self,
        frames: List[np.ndarray],
        output_path: str,
        duration: float = 0.1,
        loop: int = 0,
    ) -> bool:
        """Export frames as an animated GIF.

        Parameters
        ----------
        frames : List[np.ndarray]
            List of frames to export
        output_path : str
            Output file path
        duration : float, optional
            Duration per frame in seconds, by default 0.1
        loop : int, optional
            Number of loops (0 = infinite), by default 0

        Returns
        -------
        bool
            True if export was successful
        """
        try:
            from PIL import Image

            # Convert frames to PIL Images
            pil_images = []
            for frame in frames:
                if len(frame.shape) == 3:
                    if frame.shape[2] == 4:  # RGBA
                        img = Image.fromarray(frame, "RGBA")
                    else:  # RGB
                        img = Image.fromarray(frame, "RGB")
                else:  # Grayscale
                    img = Image.fromarray(frame, "L")
                pil_images.append(img)

            # Save as GIF
            pil_images[0].save(
                output_path,
                save_all=True,
                append_images=pil_images[1:],
                duration=int(duration * 1000),  # Convert to milliseconds
                loop=loop,
            )

            logger.info(f"GIF exported to {output_path}")
            return True

        except ImportError:
            logger.error("PIL/Pillow is required for GIF export")
            return False
        except Exception as e:
            logger.error(f"GIF export failed: {e}")
            return False

    def export_mp4(
        self, frames: List[np.ndarray], output_path: str, fps: int = 30
    ) -> bool:
        """Export frames as MP4 video.

        Parameters
        ----------
        frames : List[np.ndarray]
            List of frames to export
        output_path : str
            Output file path
        fps : int, optional
            Frames per second, by default 30

        Returns
        -------
        bool
            True if export was successful
        """
        try:
            import cv2

            if not frames:
                logger.error("No frames to export")
                return False

            # Get frame dimensions
            height, width = frames[0].shape[:2]

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # Write frames
            for frame in frames:
                # Convert to BGR if needed
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame
                out.write(frame_bgr)

            out.release()
            logger.info(f"MP4 exported to {output_path}")
            return True

        except ImportError:
            logger.error("OpenCV is required for MP4 export")
            return False
        except Exception as e:
            logger.error(f"MP4 export failed: {e}")
            return False

    def get_supported_formats(self) -> List[str]:
        """Get list of supported export formats.

        Returns
        -------
        List[str]
            List of supported formats
        """
        return self._supported_formats.copy()


class PyNDSRecorder:
    """Main recording class for PyNDS screenshots and videos.

    Provides comprehensive recording capabilities similar to PyBoy's
    recording system, including frame capture, video export, and
    advanced recording features.
    """

    def __init__(self, pynds: PyNDS):
        """Initialize PyNDS recorder.

        Parameters
        ----------
        pynds : PyNDS
            PyNDS emulator instance
        """
        self.pynds = pynds
        self.frame_capture = FrameCapture(pynds)
        self.video_exporter = VideoExporter()
        self._recording = False
        self._recorded_frames: List[np.ndarray] = []
        self._recording_start_time: Optional[float] = None

    def start_recording(self) -> None:
        """Start recording frames."""
        self._recording = True
        self._recorded_frames = []
        self._recording_start_time = time.time()
        logger.info("Recording started")

    def stop_recording(self) -> None:
        """Stop recording frames."""
        self._recording = False
        duration = time.time() - (self._recording_start_time or 0)
        logger.info(
            f"Recording stopped. Duration: {duration:.2f}s, Frames: {len(self._recorded_frames)}"
        )

    def is_recording(self) -> bool:
        """Check if currently recording.

        Returns
        -------
        bool
            True if recording
        """
        return self._recording

    def record_frame(self) -> bool:
        """Record the current frame if recording is active.

        Returns
        -------
        bool
            True if frame was recorded
        """
        if not self._recording:
            return False

        try:
            frame_data = self.frame_capture.capture_frame()
            if "frame" in frame_data:
                self._recorded_frames.append(frame_data["frame"])
                return True
        except Exception as e:
            logger.error(f"Failed to record frame: {e}")

        return False

    def take_screenshot(
        self, path: str, format: str = "png", quality: int = 95
    ) -> bool:
        """Take a screenshot of the current frame.

        Parameters
        ----------
        path : str
            Output file path
        format : str, optional
            Image format, by default "png"
        quality : int, optional
            JPEG quality (1-100), by default 95

        Returns
        -------
        bool
            True if screenshot was successful
        """
        try:
            self.pynds.export_frame(path, format, quality)
            return True
        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            return False

    def take_screenshot_sequence(
        self, directory: str, count: int, prefix: str = "screenshot", interval: int = 1
    ) -> List[str]:
        """Take a sequence of screenshots.

        Parameters
        ----------
        directory : str
            Output directory
        count : int
            Number of screenshots to take
        prefix : str, optional
            Filename prefix, by default "screenshot"
        interval : int, optional
            Frames between screenshots, by default 1

        Returns
        -------
        List[str]
            List of created file paths
        """
        try:
            os.makedirs(directory, exist_ok=True)
            file_paths = []

            for i in range(count):
                filename = f"{prefix}_{i:06d}.png"
                filepath = os.path.join(directory, filename)

                if self.take_screenshot(filepath):
                    file_paths.append(filepath)

                if i < count - 1:  # Don't advance after the last screenshot
                    for _ in range(interval):
                        self.pynds.tick(1)

            logger.info(f"Screenshot sequence completed: {len(file_paths)} files")
            return file_paths

        except Exception as e:
            logger.error(f"Screenshot sequence failed: {e}")
            return []

    def export_recording(
        self, output_path: str, format: str = "gif", fps: int = 30
    ) -> bool:
        """Export recorded frames as video.

        Parameters
        ----------
        output_path : str
            Output file path
        format : str, optional
            Export format, by default "gif"
        fps : int, optional
            Frames per second for video formats, by default 30

        Returns
        -------
        bool
            True if export was successful
        """
        if not self._recorded_frames:
            logger.error("No frames recorded")
            return False

        try:
            if format.lower() == "gif":
                duration = 1.0 / fps
                return self.video_exporter.export_gif(
                    self._recorded_frames, output_path, duration
                )
            elif format.lower() == "mp4":
                return self.video_exporter.export_mp4(
                    self._recorded_frames, output_path, fps
                )
            else:
                logger.error(f"Unsupported format: {format}")
                return False

        except Exception as e:
            logger.error(f"Recording export failed: {e}")
            return False

    def get_recording_info(self) -> Dict[str, Any]:
        """Get information about the current recording.

        Returns
        -------
        Dict[str, Any]
            Recording information
        """
        return {
            "recording": self._recording,
            "frame_count": len(self._recorded_frames),
            "duration": (
                time.time() - (self._recording_start_time or 0)
                if self._recording
                else 0
            ),
            "start_time": self._recording_start_time,
            "supported_formats": self.video_exporter.get_supported_formats(),
        }

    def clear_recording(self) -> None:
        """Clear recorded frames."""
        self._recorded_frames = []
        self._recording_start_time = None
        logger.info("Recording cleared")

    def record_gameplay(
        self, duration: float, output_path: str, format: str = "gif", fps: int = 30
    ) -> bool:
        """Record gameplay for a specified duration.

        Parameters
        ----------
        duration : float
            Recording duration in seconds
        output_path : str
            Output file path
        format : str, optional
            Export format, by default "gif"
        fps : int, optional
            Frames per second, by default 30

        Returns
        -------
        bool
            True if recording was successful
        """
        try:
            self.start_recording()

            # Calculate frames needed
            frames_needed = int(duration * fps)
            frame_interval = max(1, 60 // fps)  # Adjust based on emulation speed

            for i in range(frames_needed):
                self.record_frame()
                if i < frames_needed - 1:  # Don't advance after the last frame
                    for _ in range(frame_interval):
                        self.pynds.tick(1)

            self.stop_recording()
            return self.export_recording(output_path, format, fps)

        except Exception as e:
            logger.error(f"Gameplay recording failed: {e}")
            self.stop_recording()
            return False

    def create_timelapse(
        self, frames: List[np.ndarray], output_path: str, speed_multiplier: float = 2.0
    ) -> bool:
        """Create a timelapse from frames.

        Parameters
        ----------
        frames : List[np.ndarray]
            List of frames
        output_path : str
            Output file path
        speed_multiplier : float, optional
            Speed multiplier for timelapse, by default 2.0

        Returns
        -------
        bool
            True if timelapse creation was successful
        """
        try:
            # Calculate duration based on speed multiplier
            duration = 1.0 / (30.0 * speed_multiplier)  # Base 30 FPS
            return self.video_exporter.export_gif(frames, output_path, duration)
        except Exception as e:
            logger.error(f"Timelapse creation failed: {e}")
            return False

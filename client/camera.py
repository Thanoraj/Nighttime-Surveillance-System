"""Example script demonstrating simple video recording using OpenCV.

This script relies on :mod:`camera_utils` to handle camera connection,
recording, and cleanup. It records five seconds of video from the first
available camera and saves it as ``output.mp4``.
"""

from camera_utils import open_camera, start_writer, record, release


def main() -> None:
    """Record a short video from the default camera."""
    cap = open_camera()  # Connect to the first camera
    writer = start_writer("output.mp4", (640, 480))
    try:
        record(cap, writer, duration=5)
    finally:
        release(cap, writer)


if __name__ == "__main__":
    main()

import cv2
import time
from typing import Tuple


def open_camera(index: int = 0, frame_size: Tuple[int, int] = (640, 480)) -> cv2.VideoCapture:
    """Open a camera device and set the desired frame size.

    Parameters
    ----------
    index : int
        Index of the camera device to open.
    frame_size : Tuple[int, int]
        Desired frame width and height.

    Returns
    -------
    cv2.VideoCapture
        The opened camera object.

    Raises
    ------
    RuntimeError
        If the camera cannot be opened.
    """
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video device {index}")

    width, height = frame_size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def start_writer(output_path: str, frame_size: Tuple[int, int], fps: float = 20.0) -> cv2.VideoWriter:
    """Create a :class:`cv2.VideoWriter` for recording frames.

    Parameters
    ----------
    output_path : str
        Path of the output video file.
    frame_size : Tuple[int, int]
        Width and height of the recorded frames.
    fps : float, optional
        Frames per second of the output video, by default ``20.0``.

    Returns
    -------
    cv2.VideoWriter
        Configured video writer instance.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, frame_size)


def record(cap: cv2.VideoCapture, writer: cv2.VideoWriter, duration: float = 5.0) -> None:
    """Record frames from ``cap`` to ``writer`` for ``duration`` seconds.

    Parameters
    ----------
    cap : cv2.VideoCapture
        Open camera object to read frames from.
    writer : cv2.VideoWriter
        Video writer used to store the frames.
    duration : float, optional
        Recording duration in seconds, by default ``5.0``.

    Raises
    ------
    RuntimeError
        If a frame cannot be captured.
    """
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Failed to capture image.")
        writer.write(frame)
        if time.time() - start_time > duration:
            break


def release(cap: cv2.VideoCapture | None, writer: cv2.VideoWriter | None) -> None:
    """Release camera and writer resources if they are not ``None``."""
    if cap is not None:
        cap.release()
    if writer is not None:
        writer.release()

import cv2 as cv
import threading

class Camera:
    """
    Camera class for initialization, frame handling, and cleanup.
    Supports optional background frame capture using threading.
    """

    DEFAULT_CAMERA_INDEX = 0
    DEFAULT_WIDTH = 640
    DEFAULT_HEIGHT = 480

    def __init__(
        self,
        camera_index=DEFAULT_CAMERA_INDEX,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT
    ):
        """Initialize the camera with specified parameters."""
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None

        # Threading-related attributes
        self.latest_frame = None
        self.is_running = False
        self.capture_thread = None
        self.frame_lock = threading.Lock()

    def open(self):
        """Open the camera and set frame properties."""
        self.cap = cv.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            error_msg = f"Cannot open camera with index {self.camera_index}"
            raise RuntimeError(error_msg)

        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.height)

    def update_frame(self):
        """Continuously capture frames in a background thread."""
        while self.is_running:
            self.capture_single_frame()

    def capture_single_frame(self):
        """Capture a single frame (helper for _update_frame)."""
        if self.cap is None or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        with self.frame_lock:
            self.latest_frame = frame

    def start(self):
        """Start background frame capture."""
        self.open()
        self.is_running = True
        self.capture_thread = threading.Thread(
            target=self.update_frame,
            daemon=True
        )
        self.capture_thread.start()

    def get_frame(self):
        """
        Get the latest frame captured by the background thread if running,
        otherwise capture a single frame.
        """
        if self.is_running:
            return self.get_latest_frame()

        return self.get_immediate_frame()

    def get_latest_frame(self):
        """Get the latest frame from background thread."""
        with self.frame_lock:
            if self.latest_frame is None:
                raise RuntimeError("No frame available yet.")

            return self.latest_frame.copy()

    def get_immediate_frame(self):
        """Capture and return an immediate single frame."""
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("Camera is not opened.")

        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from camera.")

        return frame

    def stop(self):
        """Stop background frame capture and release resources."""
        if not self.is_running:
            return

        self.is_running = False

        if self.capture_thread is not None:
            self.capture_thread.join()
            self.capture_thread = None

        self.release()

    def release(self):
        """Release the camera resources."""
        if self.cap is None:
            return

        self.cap.release()
        self.cap = None

    def enter(self):
        """Context manager entry point."""
        self.open()
        return self

    def exit(self, exc_type, exc_val, exc_tb):
        """Context manager exit point."""
        self.release()

        if exc_type is not None:
            print(f"An error occurred: {exc_val}")

        return False

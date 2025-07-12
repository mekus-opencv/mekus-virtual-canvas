import time
import cv2 as cv
import numpy as np

from modules.camera import Camera
from modules.hand_tracker import HandTracker
from modules.menu import Menu
from modules.drawing import (
    COLORS,
    WHITE,
)

class VirtualCanvas:
    """
    A virtual canvas application that allows users to draw on a
    live video feed using hand gestures.
    """

    ESC_KEY = 27  # Escape key to exit
    FRAME_DELAY = 1  # Delay in milliseconds for frame processing

    CANVAS_SIZE_MULTIPLIER = 0.79  # 80% of the camera frame
    CANVAS_OFFSET_X = 0.05  # 5% from left
    CANVAS_OFFSET_Y = 0.20  # 20% from top

    BOARD_TOGGLE = "Board"
    COLORS_TOGGLE = "Colors"
    BRUSH_SIZE_TOGGLE = "Pen Size"
    ERASER_TOGGLE = "Eraser"

    DEFAULT_BRUSH_SIZE = 10
    ERASER_BRUSH_MULTIPLIER = 2

    TOGGLE_OFFSET_Y = 0.1  # 10% from left

    BOARD_TOOGLE_OFFSET_X = 0.1  # 10% from top
    PEN_TOGGLE_OFFSET_X = 0.9  # 90% from Left

    FRAME_WEIGHT = 0.5  # Weight for blending canvas with frame
    CANVAS_WEIGHT = 0.5  # Weight for blending frame with canvas
    BRIGHTNESS_ADJUSTMENT = 0.5  # Brightness adjustment for blending

    def __init__(self):
        """Initialize the virtual canvas application."""
        self.camera = Camera()
        self.tracker = HandTracker()
        self.menu = None

        # Canvas properties
        self.canvas = None
        self.canvas_size = None
        self.canvas_origin = None

        # Drawing state
        self.current_color = COLORS[0]  # Start with red
        self.brush_size = self.DEFAULT_BRUSH_SIZE
        self.current_tool = ""
        self.is_drawing = False
        self.prev_pos = None

        # UI state
        self.show_canvas = False
        self.show_colors = False
        self.show_brush_sizes = False
        self.is_eraser = False

    def initialize(self):
        """Initialize camera and UI components."""
        self.camera.start()

        try:
            self.wait_for_camera()
        except RuntimeError as e:
            print(f"Error: {e}")
            return

        self.setup_menu()
        self.setup_canvas()

    def wait_for_camera(self):
        """Wait until the camera is ready."""
        TIMEOUT = 5  # seconds
        DELAY = 0.1  # seconds
        start_time = time.time()

        print("Waiting for camera to initialize...")

        while (
            self.camera.get_immediate_frame() is None
            and (time.time() - start_time) < TIMEOUT
        ):
            time.sleep(DELAY)

        if self.camera.get_immediate_frame() is None:
            raise RuntimeError("Camera initialization timed out.")

    def setup_canvas(self):
        """Initialize the drawing canvas."""
        self.canvas_size = (
            int(self.camera.height * self.CANVAS_SIZE_MULTIPLIER),
            int(self.camera.width * self.CANVAS_SIZE_MULTIPLIER),
        )
        self.canvas_origin = (
            int(self.camera.height * self.CANVAS_OFFSET_Y),  # 20% from top
            int(self.camera.width * self.CANVAS_OFFSET_X),  # 5% from left
        )

        height, width = self.canvas_size
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.canvas[:] = WHITE  # White background

    def setup_menu(self):
        """Initialize the menu with proper positioning."""

        # Calculate menu positions based on frame dimensions
        frame_width = self.camera.width
        frame_height = self.camera.height

        # Position toggles on the right side
        board_toggle_pos = (
            int(frame_width * self.BOARD_TOOGLE_OFFSET_X),
            int(frame_height * self.TOGGLE_OFFSET_Y),
        )
        pen_toggle_pos = (
            int(frame_width * self.PEN_TOGGLE_OFFSET_X),
            board_toggle_pos[1],  # Spaced vertically
        )

        self.menu = Menu(board_toggle_pos, pen_toggle_pos)
        self.menu.create_buttons()

    def process_frame(self):
        """Process each frame for drawing and UI interaction."""
        frame = self.camera.get_frame()

        # Check if the frame is valid
        if frame is None:
            return False

        frame = cv.flip(frame, 1)  # Flip horizontally for mirror effect

        # Process the frame with the hand tracker
        processed_frame = self.tracker.process_frame(frame)

        # Get finger position and drawing state
        index_pos, will_draw = self.get_finger_info(processed_frame)

        if not will_draw:
            self.prev_pos = None

        if self.tracker.is_drawing_mode and will_draw:
            self.handle_drawing(index_pos, processed_frame)

        self.handle_ui_interaction(index_pos, processed_frame)

        if self.show_canvas:
            self.blend_canvas_onto_frame(processed_frame)

        cv.imshow("Virtual Canvas", processed_frame)

        return cv.waitKey(self.FRAME_DELAY) != self.ESC_KEY

    def get_finger_info(self, frame):
        """Extract finger position and drawing state."""
        finger_pos = self.tracker.get_hand_position(frame)
        index_pos = finger_pos.get(self.tracker.INDEX_FINGER_KEY, None)

        draw_modes = self.tracker.draw_modes
        will_draw = draw_modes[0] if draw_modes else False

        return index_pos, will_draw

    def blend_canvas_onto_frame(self, frame):
        """Blend the canvas onto the live video frame."""
        canvas_y, canvas_x = self.canvas_origin
        height, width = self.canvas_size

        frame_region = frame[
            canvas_y : canvas_y + height, canvas_x : canvas_x + width
        ]
        blended = cv.addWeighted(
            src1=frame_region,
            alpha=self.FRAME_WEIGHT,
            src2=self.canvas,
            beta=self.CANVAS_WEIGHT,
            gamma=self.BRIGHTNESS_ADJUSTMENT,
        )
        frame[canvas_y : canvas_y + height, canvas_x : canvas_x + width] = (
            blended
        )

    def handle_drawing(self, finger_pos, frame):
        """Handle drawing operations on the canvas."""
        if not self.show_canvas or not finger_pos:
            return

        canvas_x = finger_pos[0] - self.canvas_origin[1]
        canvas_y = finger_pos[1] - self.canvas_origin[0]

        if not self.is_within_canvas(canvas_x, canvas_y):
            self.prev_pos = None
            return

        self.draw_on_canvas(canvas_x, canvas_y)

    def is_within_canvas(self, canvas_x, canvas_y):
        """Check if the canvas_x and canvas_y are within canvas bounds."""
        return (
            0 <= canvas_x < self.canvas_size[1]
            and 0 <= canvas_y < self.canvas_size[0]
        )

    def draw_on_canvas(self, canvas_x, canvas_y):
        """Draw or erase on the canvas at the given canvas coordinates."""
        if self.prev_pos is None:
            self.prev_pos = (canvas_x, canvas_y)

        if self.is_eraser:
            cv.circle(
                img=self.canvas,
                center=(canvas_x, canvas_y),
                radius=self.brush_size * self.ERASER_BRUSH_MULTIPLIER,
                color=WHITE,
                thickness=cv.FILLED,
            )

        if not self.is_eraser:
            cv.line(
                img=self.canvas,
                pt1=self.prev_pos,
                pt2=(canvas_x, canvas_y),
                color=self.current_color,
                thickness=self.brush_size,
            )

        self.prev_pos = (canvas_x, canvas_y)

    def handle_ui_interaction(self, finger_pos, frame):
        """Handle interactions with UI buttons."""
        if finger_pos is None:
            return

        # Check for button clicks
        clicked_button = self.menu.handle_interaction(
            finger_pos=finger_pos, is_clicking=self.tracker.detect_click()
        )

        if clicked_button:
            self.handle_button_action(clicked_button)

        # Draw UI elements
        self.menu.draw_ui(self, frame)

    def handle_button_action(self, button):
        """Execute actions based on button clicks."""
        self.toggle_ui_states(button)
        self.select_color(button)
        self.select_size(button)
        self.select_shape(button)
        self.clear_canvas(button)

    def toggle_ui_states(self, button):
        """Toggle UI element states based on the selected button."""

        if button.label == self.BOARD_TOGGLE:
            self.tracker.toggle_drawing_mode()
            self.show_canvas = not self.show_canvas
            return

        if button.label == self.COLORS_TOGGLE:
            self.show_colors = not self.show_colors
            return

        if button.label == self.BRUSH_SIZE_TOGGLE:
            self.show_brush_sizes = not self.show_brush_sizes
            return

        if button.label == self.ERASER_TOGGLE:
            self.is_eraser = not self.is_eraser
            return

    def select_color(self, button):
        """Select a new color from the color button pressed."""
        if button in self.menu.color_buttons:
            self.current_color = button.color
            self.is_eraser = False

    def select_size(self, button):
        """Change the brush size based on the selected pen size button."""
        if button in self.menu.pen_size_buttons:
            self.brush_size = button.value

    def select_shape(self, button):
        """Select the drawing tool based on the clicked shape button."""
        if button in self.menu.shape_buttons:
            self.current_tool = button.label
            self.is_eraser = False

    def clear_canvas(self, button):
        """Clear the entire canvas when the clear button is pressed."""
        if button == self.menu.clear_button:
            self.canvas[:] = WHITE

    def run(self):
        """Main application loop."""
        self.initialize()

        while self.process_frame():
            pass  # Keep processing frames

        self.camera.stop()
        cv.destroyAllWindows()

if __name__ == "__main__":
    app = VirtualCanvas()
    app.run()
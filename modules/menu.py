import cv2 as cv
from modules.drawing import COLORS, WHITE, LIGHT_GRAY, MID_GRAY, COLOR_OPTIONS

# Text Display Constants
FONT = cv.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.6
TEXT_THICKNESS = 1
LABEL_MARGIN = 10
LABEL_BOTTOM_MARGIN = 20
LABEL_OUTLINE_WIDTH = 4

class CircleButton:
    """Represents a circular button element in the UI."""

    OVERLAY_ALPHA = 0.8
    BLEND_TOTAL_WEIGHT = 1.0
    LABEL_COLOR = WHITE
    BRIGHTNESS_ADJUSTMENT = 0

    def __init__(
        self,
        center_x,
        center_y,
        radius,
        color,
        label="",
        is_pen=False,
        value=None,
    ):
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius
        self.color = color
        self.label = label
        self.is_pen = is_pen
        self.value = value

    def draw(self, frame):
        """Render the circular button with optional label."""
        overlay = frame.copy()
        center = (self.center_x, self.center_y)
        cv.circle(overlay, center, self.radius, self.color, cv.FILLED)
        cv.addWeighted(
            src1=overlay,
            alpha=self.OVERLAY_ALPHA,
            src2=frame,
            beta=self.BLEND_TOTAL_WEIGHT - self.OVERLAY_ALPHA,
            gamma=self.BRIGHTNESS_ADJUSTMENT,
            dst=frame,
        )

        if not self.label:
            return

        text_size = cv.getTextSize(
            self.label, FONT, TEXT_SCALE, TEXT_THICKNESS
        )[0]
        text_x = self.center_x - text_size[0] // 2
        text_y = self.center_y + self.radius + text_size[1] + LABEL_MARGIN
        cv.putText(
            img=frame,
            text=self.label,
            org=(text_x, text_y),
            fontFace=FONT,
            fontScale=TEXT_SCALE,
            color=self.LABEL_COLOR,
            thickness=TEXT_THICKNESS,
        )

    def is_over(self, x, y):
        """Return True if (x, y) is inside the button area."""
        dx, dy = x - self.center_x, y - self.center_y

        return dx**2 + dy**2 < self.radius**2  # Squared distance check

class Menu:
    """UI Manager for all on-screen buttons in the drawing app."""

    # UI Button Constants
    BUTTON_RADIUS = 25
    TOGGLE_RADIUS = 30
    BUTTON_SPACING = 75
    PEN_SIZE_OFFSET_Y = 65
    MAX_COLORS = 8
    PEN_SIZE_MIN = 5
    PEN_SIZE_MAX = 30
    PEN_SIZE_STEP = 5

    CLEAR_BUTTON = "Clear"
    BOARD_TOOGLE = "Board"
    PEN_TOGGLE = "Pen Size"
    COLOR_TOGGLE = "Colors"
    ERASER_TOGGLE = "Eraser"

    # Button Position Offsets
    CLEAR_OFFSET_X = 80
    ERASER_OFFSET_X = 80
    COLOR_OFFSET_X = 80

    def __init__(self, board_toggle_pos, pen_toggle_pos):
        """Initialize the menu with toggle positions and button configurations."""
        self.board_toggle_pos = board_toggle_pos
        self.pen_toggle_pos = pen_toggle_pos
        self.last_message = ""

        self.clear_button_pos = (
            pen_toggle_pos[0] - self.CLEAR_OFFSET_X,
            pen_toggle_pos[1],
        )
        self.eraser_button_pos = (
            self.clear_button_pos[0] - self.ERASER_OFFSET_X,
            pen_toggle_pos[1],
        )
        self.color_toggle_pos = (
            self.eraser_button_pos[0] - self.COLOR_OFFSET_X,
            pen_toggle_pos[1],
        )

        self.color_buttons = []
        self.pen_size_buttons = []
        self.shape_buttons = []
        self.clear_button = None
        self.current_hover = None
        self.create_buttons()

    def create_buttons(self):
        """Initialize all buttons in the menu."""
        self.create_color_buttons()
        self.create_pen_size_buttons()
        self.create_shape_buttons()
        self.create_clear_button()
        self.create_toggle_buttons()

    def create_color_buttons(self):
        """Create buttons for color selection."""
        self.color_buttons = [
            CircleButton(
                self.color_toggle_pos[0]
                - self.BUTTON_SPACING
                - index * self.BUTTON_SPACING,
                self.color_toggle_pos[1],
                self.BUTTON_RADIUS,
                color,
            )
            for index, color in enumerate(
                [c for c in COLORS if c != WHITE][: self.MAX_COLORS]
            )
        ]

    def create_pen_size_buttons(self):
        """Create buttons for different pen sizes."""
        sizes = list(
            range(self.PEN_SIZE_MIN, self.PEN_SIZE_MAX + 1, self.PEN_SIZE_STEP)
        )
        base_y = self.pen_toggle_pos[1] + self.PEN_SIZE_OFFSET_Y
        self.pen_size_buttons = [
            CircleButton(
                center_x=self.pen_toggle_pos[0],
                center_y=base_y + index * self.BUTTON_SPACING,
                radius=self.BUTTON_RADIUS,
                color=MID_GRAY,
                label=str(size),
                is_pen=True,
                value=size,
            )
            for index, size in enumerate(sizes)
        ]

    def create_shape_buttons(self):
        """Create buttons for different shapes."""
        self.shape_buttons = []

    def create_clear_button(self):
        """Create the clear button for the canvas."""
        self.clear_button = CircleButton(
            *self.clear_button_pos,
            self.TOGGLE_RADIUS,
            LIGHT_GRAY,
            self.CLEAR_BUTTON,
        )

    def create_toggle_buttons(self):
        """Create toggle buttons for board visibility, pen size, colors, and eraser."""
        self.board_toggle = CircleButton(
            *self.board_toggle_pos,
            self.TOGGLE_RADIUS,
            LIGHT_GRAY,
            self.BOARD_TOOGLE,
        )
        self.pen_toggle = CircleButton(
            *self.pen_toggle_pos,
            self.TOGGLE_RADIUS,
            LIGHT_GRAY,
            self.PEN_TOGGLE,
        )
        self.color_toggle = CircleButton(
            *self.color_toggle_pos,
            self.TOGGLE_RADIUS,
            LIGHT_GRAY,
            self.COLOR_TOGGLE,
        )
        self.eraser_toggle = CircleButton(
            *self.eraser_button_pos,
            self.TOGGLE_RADIUS,
            LIGHT_GRAY,
            self.ERASER_TOGGLE,
        )

    def draw_ui(self, app, frame):
        """Draw all UI elements on the provided frame."""
        self.board_toggle.draw(frame)
        if not app.show_canvas:
            return

        self.draw_main_controls(frame)

        if app.show_colors:
            self.draw_color_buttons(frame)

        if app.show_brush_sizes:
            self.draw_pen_size_buttons(frame)

        if self.last_message:
            self.draw_status_message(frame)

    def draw_main_controls(self, frame):
        """Draw the main control buttons on the frame."""
        for button in [
            self.clear_button,
            self.pen_toggle,
            self.color_toggle,
            self.eraser_toggle,
        ]:
            button.draw(frame)

    def draw_color_buttons(self, frame):
        """Draw the color selection buttons."""
        for button in self.color_buttons:
            button.draw(frame)

    def draw_pen_size_buttons(self, frame):
        """Draw the pen size buttons with labels."""
        for button in self.pen_size_buttons:
            cv.circle(
                img=frame,
                center=(button.center_x, button.center_y),
                radius=button.radius + LABEL_OUTLINE_WIDTH,
                color=WHITE,
                thickness=cv.FILLED,
            )
            cv.circle(
                img=frame,
                center=(button.center_x, button.center_y),
                radius=button.radius,
                color=button.color,
                thickness=cv.FILLED,
            )

            if button.label:
                self.draw_label(
                    frame, button.label, button.center_x, button.center_y
                )

    def draw_label(self, frame, text, center_x, center_y):
        text_size = cv.getTextSize(text, FONT, TEXT_SCALE, TEXT_THICKNESS)[0]
        text_x = center_x - text_size[0] // 2
        text_y = center_y + text_size[1] // 2
        cv.putText(
            frame,
            text,
            (text_x, text_y),
            FONT,
            TEXT_SCALE,
            WHITE,
            TEXT_THICKNESS,
            cv.LINE_AA,
        )

    def draw_status_message(self, frame):
        """Display the last message at the bottom of the frame."""
        text_size = cv.getTextSize(
            self.last_message, FONT, TEXT_SCALE, TEXT_THICKNESS
        )[0]
        text_x = frame.shape[1] - text_size[0] - LABEL_BOTTOM_MARGIN
        text_y = frame.shape[0] - LABEL_BOTTOM_MARGIN
        cv.putText(
            frame,
            self.last_message,
            (text_x, text_y),
            FONT,
            TEXT_SCALE,
            WHITE,
            TEXT_THICKNESS,
            cv.LINE_AA,
        )

    def handle_interaction(self, finger_pos, is_clicking):
        """Handle button interactions based on finger position."""
        if finger_pos is None:
            self.current_hover = None
            return None

        for button in self.get_all_buttons():
            if not button.is_over(finger_pos[0], finger_pos[1]):
                continue

            self.current_hover = button

            # If the button is not clicked, just exit the method
            if not is_clicking:
                return None

            # If the button is not in the list of buttons,
            # clear the last message
            if button not in self.get_all_buttons():
                self.last_message = ""

            # If the button clicked is a color button,
            # update the last message with the color name
            if button in self.color_buttons:
                name = COLOR_OPTIONS.get(tuple(button.color), str(button.color))
                self.last_message = f"Selected Color: {name}"

            # If the button clicked is a pen size button,
            # update the last message with the pen size
            if button in self.pen_size_buttons:
                self.last_message = f"Pen Thickness: {button.value}"

            # If the button clicked is a shape button,
            # update the last message with the shape label
            if button == self.eraser_toggle:
                self.last_message = "Eraser Selected"

            return button

        self.current_hover = None
        return None

    def get_all_buttons(self):
        return [
            self.clear_button,
            *self.color_buttons,
            *self.pen_size_buttons,
            *self.shape_buttons,
            self.board_toggle,
            self.pen_toggle,
            self.color_toggle,
            self.eraser_toggle,
        ]
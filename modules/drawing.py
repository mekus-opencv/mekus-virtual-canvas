import cv2 as cv

# COLOR CONSTANTS

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
PURPLE = (255, 0, 255)
CYAN = (255, 255, 0)
WHITE = (255, 255, 255)
LIGHT_GRAY = (211, 211, 211)
MID_GRAY = (169, 169, 169)
DARK_GRAY = (120, 120, 120)
DARKER_GRAY = (70, 70, 70)
BLACK = (0, 0, 0)
ORANGE = (0, 165, 255)
PINK = (203, 192, 255)
INDIGO = (130, 0, 75)
VIOLET = (238, 130, 238)

# SHAPES CONSTANTS

DRAW_CIRCLE = "Circle"
DRAW_RECTANGLE = "Rectangle"
DRAW_LINE = "Line"
SHAPE_ELLIPSE = "Ellipse"

# OPTIONS CONSTANTS
ERASER = "Eraser"
CLEAR_DRAWING = "Clear"
FREE_DRAW = "Free Draw"

# PEN SIZE CONSTANTS
PEN_SIZE_SMALL = 2
PEN_SIZE_MEDIUM = 5
PEN_SIZE_LARGE = 10
PEN_SIZE_HUGE = 20

# LISTS OF COLORS
COLORS = [
    RED,
    GREEN,
    BLUE,
    YELLOW,
    PURPLE,
    CYAN,
    WHITE,
    BLACK,
    ORANGE,
    PINK,
    INDIGO,
    VIOLET,
]

# LISTS OF SHAPES
# LISTS OF SHAPES (in desired label order)
SHAPES = [FREE_DRAW, DRAW_CIRCLE, DRAW_RECTANGLE, DRAW_LINE]


# LISTS OF OPTIONS
OPTIONS = [
    ERASER,
    CLEAR_DRAWING,
    FREE_DRAW,
]

# LISTS OF PEN SIZES
PEN_SIZE = [PEN_SIZE_SMALL, PEN_SIZE_MEDIUM, PEN_SIZE_LARGE, PEN_SIZE_HUGE]


class Shape:

    def __init__(self, color, thickness):

        self.color = color
        self.thickness = thickness

    def draw(self, canvas):
        raise NotImplementedError("Draw method must be implemented by subclass")


class Rectangle(Shape):
    def __init__(self, start_point, end_point, color, thickness):
        super().__init__(color, thickness)
        self.start_point = start_point
        self.end_point = end_point

    def draw(self, canvas):
        cv.rectangle(
            canvas, self.start_point, self.end_point, self.color, self.thickness
        )


class Circle(Shape):
    def __init__(self, center, radius, color, thickness):
        super().__init__(color, thickness)
        self.center = center
        self.radius = radius

    def draw(self, canvas):
        cv.circle(canvas, self.center, self.radius, self.color, self.thickness)


class Line(Shape):
    def __init__(self, start_point, end_point, color, thickness):
        super().__init__(color, thickness)
        self.start_point = start_point
        self.end_point = end_point

    def draw(self, canvas):
        cv.line(
            canvas, self.start_point, self.end_point, self.color, self.thickness
        )


class Ellipse(Shape):
    def __init__(
        self, center, axes, angle, start_angle, end_angle, color, thickness
    ):
        super().__init__(color, thickness)
        self.center = center
        self.axes = axes  # (major axis length, minor axis length)
        self.angle = angle
        self.start_angle = start_angle
        self.end_angle = end_angle

    def draw(self, canvas):
        cv.ellipse(
            canvas,
            self.center,
            self.axes,
            self.angle,
            self.start_angle,
            self.end_angle,
            self.color,
            self.thickness,
        )

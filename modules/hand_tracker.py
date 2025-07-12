import cv2 as cv
import mediapipe as mp

from modules.drawing import LIGHT_GRAY, DARKER_GRAY

class HandTracker:
    """Track hand landmarks and detect raised fingers using MediaPipe Hands."""

    # Drawing constants
    CIRCLE_TIP_RADIUS = 5
    INNER_CIRCLE_TIP_COLOR = LIGHT_GRAY
    OUTER_CIRCLE_TIP_COLOR = DARKER_GRAY
    INNER_CIRCLE_TIP_FILL = cv.FILLED
    OUTER_CIRCLE_TIP_THICKNESS = 2

    INDEX_FINGER_KEY = "INDEX_FINGER"
    MIDDLE_FINGER_KEY = "MIDDLE_FINGER"

    SELECTED_HAND_INDEX = 0  # Only one hand is selected for drawing

    # Finger landmark indices
    # These indices correspond to the MediaPipe Hands landmark model
    # https://chuoling.github.io/mediapipe/solutions/hands.html
    FINGER_LANDMARKS = {
        INDEX_FINGER_KEY: {"MCP": 5, "PIP": 6, "DIP": 7, "TIP": 8},
        MIDDLE_FINGER_KEY: {"MCP": 9, "PIP": 10, "DIP": 11, "TIP": 12},
    }

    def __init__(
        self,
        use_static_image_mode=False,
        max_num_hands=2,
        detection_conf=0.8,
        tracking_conf=0.85,
    ):
        """Initialize the hand tracker with MediaPipe Hands settings."""
        self.hand_detector = mp.solutions.hands.Hands(
            static_image_mode=use_static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )

        self.prev_index_up = False
        self.prev_middle_up = False

        self.is_clicked_prev = False

        self.hand_landmarks = None
        self.is_drawing_mode = False
        self.draw_modes = []

    def process_frame(self, image_bgr):
        """Process a frame and return it with drawn landmarks."""
        self.update_landmarks(image_bgr)

        # Draw raised fingers if hands are detected
        if self.hand_landmarks and self.hand_landmarks.multi_hand_landmarks:
            self.draw_raised_fingers(image_bgr)

        return image_bgr

    def get_hand_position(self, image_bgr):
        """Get current finger positions for all hands, separated by index."""

        if not self.has_landmarks():
            return {}  # No hands detected return empty dict

        selected_hand = self.hand_landmarks.multi_hand_landmarks[
            self.SELECTED_HAND_INDEX
        ]

        finger_positions = {}  # Dictionary to hold finger positions

        # Extract positions for each finger
        for finger_name, joint_names in self.FINGER_LANDMARKS.items():
            tip = selected_hand.landmark[joint_names["TIP"]]
            finger_positions[finger_name] = self.normalize_coordinates(
                tip, image_bgr.shape
            )

        return finger_positions

    def update_landmarks(self, image_bgr):
        """Update hand landmarks from the current frame."""
        image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
        self.hand_landmarks = self.hand_detector.process(image_rgb)
        self.hand_classifications = self.hand_landmarks.multi_handedness or []

    def has_landmarks(self):
        """Check if valid landmarks exist."""
        return self.hand_landmarks and self.hand_landmarks.multi_hand_landmarks

    def draw_raised_fingers(self, image_bgr):
        """Draw circles on all raised fingers for each detected hand."""
        raised_fingers = self.detect_raised_fingers()
        self.update_drawing_mode(raised_fingers)

        for hand_idx, landmark in enumerate(
            self.hand_landmarks.multi_hand_landmarks
        ):
            self.draw_finger_tips_for_hand(
                image_bgr, hand_idx, landmark, raised_fingers
            )

    def draw_finger_tips_for_hand(
        self, image_bgr, hand_idx, landmark, raised_fingers
    ):
        """Draw raised finger tips for a specific hand based on index."""
        is_draw_mode = self.get_draw_mode_for_hand(hand_idx)
        self.draw_finger_tips(
            image_bgr, landmark, raised_fingers[hand_idx], is_draw_mode
        )

    def get_draw_mode_for_hand(self, hand_idx):
        """Return whether the specific hand is in drawing mode."""
        return (
            self.draw_modes[hand_idx]
            if hand_idx < len(self.draw_modes)
            else False
        )

    def update_drawing_mode(self, raised_fingers):
        """Check if the drawing mode is active based on raised fingers."""
        self.draw_modes.clear()

        for fingers in raised_fingers:
            index_up = fingers.get(self.INDEX_FINGER_KEY)
            middle_up = fingers.get(self.MIDDLE_FINGER_KEY)

            is_drawing = index_up and not middle_up
            self.draw_modes.append(is_drawing)

    def draw_finger_tips(
        self,
        image_bgr,
        landmark,
        raised_fingers,
        is_draw_mode,
    ):
        """Draw tips for raised fingers of a single hand."""
        for finger_name, joint_names in self.FINGER_LANDMARKS.items():
            if raised_fingers.get(finger_name):
                tip = landmark.landmark[joint_names["TIP"]]
                x_coord, y_coord = self.normalize_coordinates(
                    tip, image_bgr.shape
                )
                self.draw_finger_tip(
                    image_bgr, finger_name, (x_coord, y_coord), is_draw_mode
                )

    def draw_finger_tip(self, image_bgr, finger_name, coord, is_draw_mode):
        """Draw a single finger tip marker based on its hand's draw mode."""
        cv.circle(
            image_bgr,
            coord,
            self.CIRCLE_TIP_RADIUS,
            self.OUTER_CIRCLE_TIP_COLOR,
            self.OUTER_CIRCLE_TIP_THICKNESS,
        )

        # Draw inner circle only if not in drawing mode and not the index finger
        if not (is_draw_mode and finger_name == self.INDEX_FINGER_KEY):
            cv.circle(
                image_bgr,
                coord,
                self.CIRCLE_TIP_RADIUS,
                self.INNER_CIRCLE_TIP_COLOR,
                self.INNER_CIRCLE_TIP_FILL,
            )

    def detect_raised_fingers(self):
        """Detect raised fingers for all hands."""
        if not self.has_landmarks():
            return []  # No hands detected return empty list

        return [
            self.check_hand_fingers(hand)
            for hand in self.hand_landmarks.multi_hand_landmarks
        ]

    def check_hand_fingers(self, hand_landmark):
        """Check which fingers are raised for a single hand."""
        return {
            finger_name: self.is_finger_raised(hand_landmark, joints)
            for finger_name, joints in self.FINGER_LANDMARKS.items()
        }

    def is_finger_raised(self, hand_landmark, joints):
        """Check if a single finger is raised."""
        y_coords = {
            key: hand_landmark.landmark[joints[key]].y
            for key in ["TIP", "DIP", "PIP", "MCP"]
        }

        return (
            y_coords["TIP"]
            < y_coords["DIP"]
            < y_coords["PIP"]
            < y_coords["MCP"]
        )

    def normalize_coordinates(self, landmark, image_shape):
        """Convert normalized coordinates to pixel coordinates."""
        image_height, image_width, _ = image_shape

        return (
            int(landmark.x * image_width),
            int(landmark.y * image_height),
        )

    def toggle_drawing_mode(self):
        self.is_drawing_mode = not self.is_drawing_mode

    def detect_click(self):
        """Detect click gesture based on index and middle finger positions."""
        if not self.has_landmarks():
            return False  # No hands detected return False

        hand = self.hand_landmarks.multi_hand_landmarks[
            self.SELECTED_HAND_INDEX
        ]
        index_up, middle_up = self.get_current_finger_states(hand)

        click_happened = self.is_click_happening(index_up, middle_up)
        self.update_previous_states(index_up, middle_up, click_happened)

        return click_happened

    def get_current_finger_states(self, hand):
        """Get the current raised states of index and middle fingers."""
        index_joints = self.FINGER_LANDMARKS[self.INDEX_FINGER_KEY]
        middle_joints = self.FINGER_LANDMARKS[self.MIDDLE_FINGER_KEY]
        return (
            self.is_finger_raised(hand, index_joints),
            self.is_finger_raised(hand, middle_joints),
        )

    def is_click_happening(self, index_up, middle_up):
        """Determine if a click gesture occurred based on finger states."""
        middle_curled = not middle_up
        was_ready = self.prev_index_up and self.prev_middle_up
        return (
            was_ready
            and index_up
            and middle_curled
            and not self.is_clicked_prev
        )

    def update_previous_states(self, index_up, middle_up, click_happened):
        """Update previous finger states for click detection."""
        self.prev_index_up = index_up
        self.prev_middle_up = middle_up
        self.is_clicked_prev = click_happened

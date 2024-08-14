import cv2
import numpy as np
from util.detect import get_landmarks, get_eye_points, create_shifted_points, add_intermediate_points, expand_points
from util.utils import get_color_from_json

eyeshadow_alpha = 0.7  # Opacity of the eyeshadow (0.0 to 1.0)

def hex_to_bgr(hex_color: str) -> tuple:
    """Convert HEX color code to BGR color format."""
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))  # Skip the '#'
    return (rgb[2], rgb[1], rgb[0])  # Convert RGB to BGR

def apply_eyeshadow(image: np.ndarray, prdCode: str, color: str) -> np.ndarray:
    """
    Apply eyeshadow to the image using specified product code and color.

    :param image: Input image (BGR format)
    :param prdCode: Product code to fetch color from JSON
    :param color: HEX color code for eyeshadow
    :return: Image with applied eyeshadow
    """
    # Get primary color from JSON and convert HEX to BGR
    bgr_color, _, option2 = get_color_from_json(prdCode)
    bgr_color = hex_to_bgr(color)
    
    bgr_color2 = None
    if option2 and option2 != "None":
        option2 = option2.lstrip('#')
        bgr_color2 = hex_to_bgr(option2)

    # Convert image to grayscale and detect landmarks
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    landmarks = get_landmarks(image)
    
    if landmarks is None:
        print("No faces detected.")
        return image

    landmarks = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])

    # Extract points for left and right eyes
    left_eye_indices = range(36, 40)
    right_eye_indices = range(42, 46)
    
    left_eye_points = landmarks[left_eye_indices]
    right_eye_points = landmarks[right_eye_indices]

    # Add intermediate points for smoother transition
    left_eye_points = add_intermediate_points(left_eye_points.tolist(), num_points=2)
    right_eye_points = add_intermediate_points(right_eye_points.tolist(), num_points=2)

    # Expand points to cover more area around the eyes
    additional_left_points = np.array([
        landmarks[37] + np.array([7, 0]),  # Extend left
        landmarks[39] + np.array([7, 0])   # Extend left
    ])
    additional_right_points = np.array([
        landmarks[43] + np.array([7, 0]),  # Extend right
        landmarks[45] + np.array([7, 0])   # Extend right
    ])

    left_eye_points = np.vstack((left_eye_points, additional_left_points))
    right_eye_points = np.vstack((right_eye_points, additional_right_points))

    # Add intermediate points for smoother coverage
    left_eye_points = add_intermediate_points(left_eye_points.tolist(), num_points=2)
    right_eye_points = add_intermediate_points(right_eye_points.tolist(), num_points=2)

    # Shift and expand points to cover the eye area
    y_shift1 = 18
    y_shift2 = 26
    left_shifted_points1 = create_shifted_points(left_eye_points, y_shift1)
    left_shifted_points2 = create_shifted_points(left_eye_points, y_shift2)
    right_shifted_points1 = create_shifted_points(right_eye_points, y_shift1)
    right_shifted_points2 = create_shifted_points(right_eye_points, y_shift2)

    left_shifted_points1 = expand_points(left_shifted_points1, factor=1.2)
    left_shifted_points2 = expand_points(left_shifted_points2, factor=1.4)
    right_shifted_points1 = expand_points(right_shifted_points1, factor=1.2)
    right_shifted_points2 = expand_points(right_shifted_points2, factor=1.4)

    # Define the eyeshadow areas for both colors
    left_eye_area1 = np.vstack((left_eye_points, left_shifted_points1[::-1]))
    left_eye_area2 = np.vstack((left_shifted_points1, left_shifted_points2[::-1]))
    right_eye_area1 = np.vstack((right_eye_points, right_shifted_points1[::-1]))
    right_eye_area2 = np.vstack((right_shifted_points1, right_shifted_points2[::-1]))

    # Convert image to BGRA (add alpha channel)
    image_bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    # Apply the first eyeshadow color
    for eye_area, color in zip([left_eye_area1, right_eye_area1], [bgr_color, bgr_color]):
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        cv2.fillPoly(mask, [eye_area], (1))

        # Create and blur alpha channel
        alpha_channel = (mask * eyeshadow_alpha * 255).astype(np.uint8)
        blurred_alpha_channel = cv2.GaussianBlur(alpha_channel, (45, 45), 0)
        blurred_alpha_channel = cv2.medianBlur(blurred_alpha_channel, 27)

        # Create and apply eyeshadow
        eyeshadow = np.zeros_like(image_bgra, dtype=np.uint8)
        eyeshadow[:, :, :3] = color
        eyeshadow[:, :, 3] = blurred_alpha_channel

        alpha_mask = blurred_alpha_channel / 255.0
        for c in range(0, 3):
            image[:, :, c] = (1.0 - alpha_mask) * image[:, :, c] + alpha_mask * eyeshadow[:, :, c]

    # Apply the second eyeshadow color if available
    if bgr_color2:
        for eye_area, color in zip([left_eye_area2, right_eye_area2], [bgr_color2, bgr_color2]):
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
            cv2.fillPoly(mask, [eye_area], (1))

            # Create and blur alpha channel
            alpha_channel = (mask * eyeshadow_alpha * 255).astype(np.uint8)
            blurred_alpha_channel = cv2.GaussianBlur(alpha_channel, (75, 75), 0)
            blurred_alpha_channel = cv2.medianBlur(blurred_alpha_channel, 21)

            # Create and apply eyeshadow
            eyeshadow = np.zeros_like(image_bgra, dtype=np.uint8)
            eyeshadow[:, :, :3] = color
            eyeshadow[:, :, 3] = blurred_alpha_channel

            alpha_mask = blurred_alpha_channel / 255.0
            for c in range(0, 3):
                image[:, :, c] = (1.0 - alpha_mask) * image[:, :, c] + alpha_mask * eyeshadow[:, :, c]

    return image

import cv2
import numpy as np
from util.detect import get_landmarks, get_cheek_centers
from util.utils import get_color_from_json

blush_alpha = 0.12  # Blush transparency (0.0 to 1.0)
blush_radius = 38  # Blush application radius (in pixels, adjustable)
blush_offset = -36  # Offset to move the blush position upwards (in pixels)
x_offset = 5  # Offset to move the blush position leftwards (in pixels)

def hex_to_bgr(hex_color: str) -> tuple:
    """Convert HEX color code to BGR color format."""
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))  # Skip the '#'
    return (rgb[2], rgb[1], rgb[0])  # Convert RGB to BGR

def apply_blush(image: np.ndarray, prdCode: str, color: str) -> np.ndarray:
    # Convert HEX color to BGR
    blush_color = hex_to_bgr(color)

    # Convert BGR to BGRA (add alpha channel)
    image_bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    
    # Detect faces in the image
    landmarks = get_landmarks(image)
    if landmarks is None:
        print("No faces detected.")
        return image

    # Process each face detected
    for face in [landmarks]:  # Currently handling only one face
        shape = landmarks

        # Calculate left and right cheek centers
        left_cheek_center, right_cheek_center = get_cheek_centers(shape, blush_offset, x_offset)
        
        for cheek_center in [left_cheek_center, right_cheek_center]:
            # Create a mask for the blush area
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
            cv2.circle(mask, (int(cheek_center[0]), int(cheek_center[1])), blush_radius, (1), -1, cv2.LINE_AA)

            # Create an alpha channel based on the mask
            alpha_channel = (mask * blush_alpha * 255).astype(np.uint8)
            
            # Apply Gaussian blur to alpha channel for a soft blush effect
            alpha_channel = cv2.GaussianBlur(alpha_channel, (45, 45), 0)
            
            # Apply median blur to alpha channel for noise reduction and smoothness
            alpha_channel = cv2.medianBlur(alpha_channel, 21)

            # Create blush image
            blush = np.zeros_like(image_bgra, dtype=np.uint8)
            blush[:, :, :3] = blush_color  # Color channels
            blush[:, :, 3] = alpha_channel  # Alpha channel
            
            # Update the final image considering the alpha channel
            alpha_mask = alpha_channel / 255.0  # Normalize alpha values between 0 and 1
            for c in range(0, 3):
                # Blend blush and original image for each color channel
                image[:, :, c] = (1.0 - alpha_mask) * image[:, :, c] + alpha_mask * blush[:, :, c]

    return image

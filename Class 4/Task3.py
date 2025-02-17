import numpy as np
import cv2

# Load an image
image = cv2.imread('image.jpg')  # Use your own image


# Scaling Transformation
def scale_image(image, scale_factor):
    # Create the scaling matrix
    scaling_matrix = np.array([[scale_factor, 0, 0],
                               [0, scale_factor, 0],
                               [0, 0, 1]])
    # Get the image dimensions
    rows, cols = image.shape[:2]
    # Apply the transformation
    scaled_image = cv2.warpPerspective(image, scaling_matrix, (cols, rows))
    return scaled_image


# Rotation Transformation
def rotate_image(image, angle):
    # Get the image dimensions
    rows, cols = image.shape[:2]
    # Get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    # Apply the rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    return rotated_image


# Translation Transformation
def translate_image(image, tx, ty):
    # Create the translation matrix (2×3 instead of 3×3)
    translation_matrix = np.float32([[1, 0, tx],
                                     [0, 1, ty]])
    # Get the image dimensions
    rows, cols = image.shape[:2]
    # Apply the transformation using warpAffine (not warpPerspective)
    translated_image = cv2.warpAffine(image, translation_matrix, (cols, rows))
    return translated_image

# Example transformations
scaled_image = scale_image(image, 1.5)  # Scale by a factor of 1.5
rotated_image = rotate_image(image, 45)  # Rotate by 45 degrees
translated_image = translate_image(image, 50, 30) # Translate by 50 pixels along x and 30 pixels along y

# Show the results
cv2.imshow('Original', image)
cv2.imshow('Scaled', scaled_image)
cv2.imshow('Rotated', rotated_image)
cv2.imshow('Translated', translated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

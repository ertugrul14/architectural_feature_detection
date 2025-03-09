import cv2
import numpy as np
import matplotlib.pyplot as plt
import svgwrite
import os

# Adjustable parameters - modified for black wall detection
MIN_CONTOUR_AREA = 500    # Increased to filter out smaller features
EPSILON_FACTOR = 0.01     # Increased for more significant corners
MIN_CORNER_DISTANCE = 10  # Increased to avoid duplicate corners
WALL_THRESHOLD = 50       # Lower threshold to detect dark walls
FILL_THRESHOLD = 0.85     # Higher threshold for filled areas
GAMMA = 0.5               # Darken the image

# Levels adjustment parameters
INPUT_BLACK = 50          # Lower to keep dark areas
INPUT_WHITE = 150         # Lower to make light areas white
OUTPUT_BLACK = 0         
OUTPUT_WHITE = 255       

def adjust_levels(image, in_black=0, in_white=255, out_black=0, out_white=255):
    """Adjust image levels similar to Photoshop's levels adjustment"""
    in_black = max(0, min(in_black, 255))
    in_white = max(0, min(in_white, 255))
    out_black = max(0, min(out_black, 255))
    out_white = max(0, min(out_white, 255))
    
    table = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        if i <= in_black:
            table[i] = out_black
        elif i >= in_white:
            table[i] = out_white
        else:
            scale = (i - in_black) / (in_white - in_black)
            table[i] = int(out_black + scale * (out_white - out_black))
    
    return cv2.LUT(image, table)

def adjust_gamma(image, gamma=1.0):
    """Apply gamma correction to the image"""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load and process image using relative path
image_path = os.path.join(current_dir, "plan_1.png")
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise ValueError(f"Could not load image from {image_path}")

# Apply levels adjustment and gamma correction
image = adjust_levels(image, in_black=INPUT_BLACK, in_white=INPUT_WHITE, out_black=OUTPUT_BLACK, out_white=OUTPUT_WHITE)
image = adjust_gamma(image, GAMMA)

# Apply threshold to isolate filled walls
_, binary = cv2.threshold(image, WALL_THRESHOLD, 255, cv2.THRESH_BINARY)

# Clean up the image
kernel = np.ones((5,5), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

# Invert the colors
binary = cv2.bitwise_not(binary)

# Find contours
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Remove nearby duplicates
def remove_nearby_points(points, min_distance):
    filtered_points = []
    for p1 in points:
        if not any(np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) < min_distance 
                  for p2 in filtered_points):
            filtered_points.append(p1)
    return filtered_points

# Visualization
plt.figure(figsize=(12, 8))
plt.imshow(binary, cmap='gray')

# Save the plot without graphical frames and text using relative path
output_png_filename = os.path.join(current_dir, "detected_walls.png")
plt.axis('off')  # Turn off the axis
plt.savefig(output_png_filename, bbox_inches='tight', pad_inches=0)
plt.close()

print(f"PNG saved as: {output_png_filename}")

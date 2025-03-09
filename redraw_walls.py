import cv2
import numpy as np
import matplotlib.pyplot as plt
import svgwrite
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the processed image using relative path
image_path = os.path.join(current_dir, "detected_walls.png")
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Improved corner detection parameters
max_corners = 2000  # Increased for more corners
quality_level = 0.0001  # Adjusted for better corner sensitivity
min_distance = 1  # Decreased to detect closer corners
block_size = 3  # Increased for better local analysis
k = 0.04

# Create SVG drawing with relative path
height, width = image.shape[:2]
svg_path = os.path.join(current_dir, "vector_walls.svg")
dwg = svgwrite.Drawing(svg_path, size=(width, height), profile='tiny')

# Improve binary image
_, binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
kernel = np.ones((3,3), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

# Find contours with improved parameters
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

# Create output image for PNG
output_image = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background

# Process each contour with refined parameters
for contour in contours:
    # More accurate corner detection
    epsilon = 0.001 * cv2.arcLength(contour, True)  # Decreased for higher precision
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    if len(approx) >= 3:
        # Filter points that are too close to each other
        filtered_points = []
        min_dist_threshold = 2  # Minimum distance between points
        
        for i, point in enumerate(approx):
            point = point[0]
            # Check distance with previously added points
            too_close = False
            for existing_point in filtered_points:
                dist = np.sqrt(np.sum((point - existing_point) ** 2))
                if dist < min_dist_threshold:
                    too_close = True
                    break
            if not too_close:
                filtered_points.append(point)
        
        # Create path for the wall using filtered points
        if len(filtered_points) >= 3:
            points = filtered_points  # No need to convert again, already in correct format
            
            # Draw walls in PNG
            points_array = np.array(points)
            cv2.polylines(output_image, [points_array], True, (0, 0, 0), 1)
            
            # Draw corner points in PNG
            for point in points:
                cv2.circle(output_image, tuple(point), 2, (255, 0, 0), -1)  # Red dots
            
            # Add to SVG
            path_data = f"M {points[0][0]},{points[0][1]}"
            for point in points[1:]:
                path_data += f" L {point[0]},{point[1]}"  # Remove extra [0] index
            path_data += " Z"  # Close the path
            
            # Add wall path to SVG
            path = dwg.path(d=path_data, stroke='black', fill='none', stroke_width=1)
            dwg.add(path)
            
            # Add corner points
            for point in points:
                dwg.add(dwg.circle(
                    center=(str(point[0]), str(point[1])),
                    r=2,
                    fill='red'
                ))

# Save both SVG and PNG using relative paths
output_vector_path = os.path.join(current_dir, "vector_walls.svg")
output_png_path = os.path.join(current_dir, "vector_walls.png")

dwg.save()
cv2.imwrite(output_png_path, output_image)

print(f"Vector drawing complete. Files saved as:")
print(f"SVG: {output_vector_path}")
print(f"PNG: {output_png_path}")
import cv2
import requests
import numpy as np
from inference_sdk import InferenceHTTPClient
from PIL import Image
import random
import svgwrite
import os

# Initialize the Roboflow Inference client
CLIENT = InferenceHTTPClient(
    api_url="https://outline.roboflow.com",
    api_key="XBnxrOb9awrSLw1QQWIf"
)

def generate_random_color():
    # Generate pastel colors for better visibility
    return (
        random.randint(100, 255),  # Blue
        random.randint(100, 255),  # Green
        random.randint(100, 255)   # Red
    )

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

try:
    # Load image using relative path
    image_path = os.path.join(current_dir, "vector_walls.png")
    image = Image.open(image_path)
    
    # Perform inference
    result = CLIENT.infer(image_path, model_id="room-detection-6nzte/1")
    
    # Load image using OpenCV for visualization
    image_cv = cv2.imread(image_path)
    
    if image_cv is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Create a mask for filling rooms
    mask = np.zeros_like(image_cv)
    
    # Parse and draw bounding boxes
    for prediction in result['predictions']:
        x, y, width, height = prediction['x'], prediction['y'], prediction['width'], prediction['height']
        class_name = prediction['class']
        
        # Convert center (x, y) to top-left (x1, y1)
        x1, y1 = int(x - width / 2), int(y - height / 2)
        x2, y2 = int(x + width / 2), int(y + height / 2)
        
        # Generate a random color for this room
        room_color = generate_random_color()
        
        # Create a polygon for the room
        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        
        # Fill the room with the random color
        cv2.fillPoly(mask, [pts], room_color)
        
        # Draw room boundaries
        cv2.polylines(mask, [pts], True, (0, 0, 0), 2)
        
        # Add room label
        cv2.putText(mask, class_name, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # Get the current working directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Combine original image with colored rooms
    result_image = cv2.addWeighted(image_cv, 0.3, mask, 0.7, 0)
    
    # Save as PNG using relative path
    png_output_path = os.path.join(current_dir, "detected_rooms_colored.png")
    cv2.imwrite(png_output_path, result_image)

    # Create SVG drawing using relative path
    svg_output_path = os.path.join(current_dir, "detected_rooms_colored.svg")
    dwg = svgwrite.Drawing(svg_output_path, size=(image_cv.shape[1], image_cv.shape[0]))
    
    # Add rooms to SVG
    for prediction in result['predictions']:
        x, y, width, height = prediction['x'], prediction['y'], prediction['width'], prediction['height']
        x1, y1 = int(x - width / 2), int(y - height / 2)
        x2, y2 = int(x + width / 2), int(y + height / 2)
        
        # Create room polygon
        points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        room_color = generate_random_color()
        
        # Add filled polygon with black stroke
        dwg.add(dwg.polygon(
            points=points,
            fill=svgwrite.rgb(*room_color, 'rgb'),
            stroke='black',
            stroke_width=2
        ))
        
        # Add room label
        dwg.add(dwg.text(
            prediction['class'],
            insert=(x1, y1 - 5),
            font_size='12px',
            fill='black'
        ))
    
    # Save SVG
    dwg.save()
    
    # Display the image with detections
    cv2.imshow("Room Detection", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Detection complete. Results saved as:")
    print(f"PNG: {png_output_path}")
    print(f"SVG: {svg_output_path}")

except Exception as e:
    print(f"Error: {str(e)}")

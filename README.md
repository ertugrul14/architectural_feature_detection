🏛️ Architectural Feature Detection Pipeline
This project provides a Python-based pipeline to detect architectural features—such as walls and rooms—from scanned floor plan images. The output includes both visual and vector (SVG) representations of detected structures.


📂 Overview
The pipeline consists of the following stages:

Image Preprocessing (image_adjustment.py): Enhances contrast, darkens the image, and extracts wall-like contours.

Wall Vectorization (redraw_walls.py): Detects corners, simplifies walls, and exports a clean vector drawing (SVG + PNG).

Room Detection (space_detection.py): Uses the Roboflow API to detect rooms inside the walls and generates a colored overlay and SVG layout.

Automation (pipeline.py): A runner script that sequentially executes the above scripts.

🛠️ Requirements
Python 3.7+

Packages:
opencv-python, numpy, matplotlib, svgwrite, Pillow, requests,
and optionally: inference-sdk (for Roboflow)

▶️ Usage
Place your floor plan image (e.g. plan_1.png) in the root directory.

Run the entire pipeline:

python pipeline.py

This will generate:

1.detected_walls.png
2.vector_walls.svg, vector_walls.png
3.detected_rooms_colored.png, detected_rooms_colored.svg

🤖 Roboflow Integration
Room detection uses the Roboflow Outline API.
To use this feature:

Get a free Roboflow API key

Replace the placeholder in space_detection.py:

api_key="YOUR_API_KEY"

🧱 File Structure
.
├── plan_1.png                  # Input floor plan
├── image_adjustment.py        # Wall preprocessing
├── redraw_walls.py            # Vectorization of walls
├── space_detection.py         # Room classification
├── pipeline.py                # Orchestrator
├── detected_walls.png         # Binary wall output
├── vector_walls.svg/.png      # Clean vector wall drawing
├── detected_rooms_colored.*   # Room detection output

🧪 Future Improvements
Detect doors/windows in addition to walls and rooms

Export layered DXF files for CAD use

Optimize wall vectorization for curved walls

Batch processing support

📜 License
This project is for academic and research use.
Please cite appropriately if used in publications.


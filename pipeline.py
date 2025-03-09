import os

scripts = ["image_adjustment.py", "redraw_walls.py", "space_detection.py"]

print("Starting the pipeline...\n")

for script in scripts:
    print(f"Running {script}...")
    exit_code = os.system(f"python {script}")
    
    if exit_code != 0:
        print(f"Error: {script} encountered an issue. Exiting pipeline.")
        break
    print(f"{script} completed successfully.\n")

print("Pipeline execution finished.")

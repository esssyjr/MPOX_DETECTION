
from fastapi import FastAPI, UploadFile
from typing import List
from ultralytics import YOLO
from PIL import Image
import io

# Initialize the FastAPI app
app = FastAPI()

# Load your YOLO model when the app starts
model = YOLO("MPOX.pt")  # Ensure your model path is correct

@app.post("/detect")
async def detect_mpox(files: List[UploadFile]):
    """
    Detect MPOX in uploaded images.

    Args:
        files (List[UploadFile]): List of images uploaded by the user.

    Returns:
        dict: Status and total detections.
    """
    total_detections = 0  # Initialize a counter for detections

    # Process each uploaded image
    for file in files:
        image_bytes = await file.read()  # Read image file as bytes
        image = Image.open(io.BytesIO(image_bytes))  # Convert bytes to an image

        # Perform detection using the YOLO model
        results = model(image)

        # Count the number of detections in the current image
        num_detections = len(results[0].boxes)  # Assuming results[0].boxes contains the detections
        total_detections += num_detections

    # Check if there are 2 or more detections to confirm the case
    if total_detections >= 2:
        return {"status": "Confirmed Case", "detections": total_detections}
    else:
        return {"status": "Not Confirmed", "detections": total_detections}

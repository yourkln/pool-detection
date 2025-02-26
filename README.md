# Swimming Pool Detection in Aerial Images

## Overview
This Python script detects swimming pools in aerial images, such as satellite or drone views of residential neighborhoods. It combines deep learning and traditional computer vision techniques for robust detection. Initially, I fine-tuned YOLOv11n model (`best.pt`) from the `ultralytics` library to identify potential pool regions, followed by OpenCV (`cv2`) and NumPy (`np`) for refined color segmentation, noise removal, and contour analysis. The script outlines detected pools in red on the input image and saves their boundary coordinates to a text file. It is designed to handle challenges like shadows, irregular pool shapes, and varying land colors, as seen in the provided aerial images.

## Process Explanation
The script processes an aerial image through the following steps:

1. **Input Handling**: The script accepts an aerial image path via a command-line argument and loads the image for processing.
2. **YOLO-Based Initial Detection**: A fine-tuned YOLOv11n model (`best.pt`) predicts potential swimming pool regions, providing bounding boxes around likely pool areas.
3. **Bounding Box Enlargement**: Each bounding box is enlarged by a factor of 0.3 (30%) to ensure the entire pool is captured, even if the initial detection misses parts of it. The enlarged region of interest (ROI) is then cropped for further processing.
4. **Color Segmentation on ROI**: The cropped ROI is converted to HSV color space, and a mask is created to isolate blue/turquoise pixels (hue: 80–120, saturation: 60–255, value: 80–255), characteristic of pool water.
5. **Noise Removal**: Morphological operations are applied to the mask:
   - **Opening**: Performed with a 3x3 kernel (2 iterations) to remove small noise (e.g., isolated blue pixels).
   - **Closing**: Performed with a 3x3 kernel (3 iterations) to fill gaps in pool areas (e.g., caused by shadows or reflections).
   - I found these values to be the best
6. **Contour Detection and Filtering**: Contours are extracted from the cleaned mask and filtered based on:
   - **Area**: Between 200 and 30,000 pixels to include small and large pools while excluding tiny noise or oversized non-pool regions.
   - **Shape**: Contours are smoothed using `cv2.approxPolyDP` with an epsilon of 0.001 times the contour’s arc length, ensuring accurate boundaries for irregular pool shapes.
7. **Output Generation**: Valid contours are drawn in red (1-pixel thickness) on the cropped ROI. These contours are then overlaid back onto the original image, with their coordinates adjusted to be relative to the full image. The annotated image and all pool boundary coordinates are saved to files.

### Changes Made
- **Integrated YOLO Model**: Replaced the original standalone color-based detection with a YOLOv11n model (`best.pt`) for initial pool detection, improving reliability by focusing on likely pool regions before applying traditional CV techniques.
- **Bounding Box Enlargement**: Added a 30% enlargement of YOLO bounding boxes to capture entire pools, addressing cases where initial detections might be too tight.
- **Refined Contour Processing**: Updated the `detect_pools` function to process ROIs from YOLO detections, applying color segmentation, morphological operations (with specific iteration counts: 2 for opening, 3 for closing), and contour smoothing with `cv2.approxPolyDP`.
- **Removed Texture and Shape Filters**: Eliminated previous texture filtering (Laplacian) and shape constraints (circularity, color uniformity) from the original pipeline, relying instead on YOLO’s initial detection and area-based contour filtering.

**Examples**:

| Sample Image | Output Image |
|-------------|-------------|
| ![000000079](https://yourkln.com/assets/000000079.jpg) | ![000000079_output](https://yourkln.com/assets/000000079_output.jpg) |
| ![000000136](https://yourkln.com/assets/000000136.jpg) | ![000000136_output](https://yourkln.com/assets/000000136_output.jpg) |
| ![000000216](https://yourkln.com/assets/000000216.jpg) | ![000000216_output](https://yourkln.com/assets/000000216_output.jpg) |

## How to Use

### Prerequisites

- **Python**: Ensure Python is installed on your system.
- **Libraries**: Install the required dependencies using pip:
  
  ```bash
  pip install opencv-python numpy ultralytics matplotlib
  ```
  
- **YOLO Model**: Place the fine-tuned `best.pt` model file in the same directory as the script (loaded as `./best.pt`).

### Running the Script via Command Line (CL)
1. **Save the Script**: Save the provided code as `detect_pools.py`.
2. **Prepare Your Image**: Have an aerial image ready (e.g., satellite or drone imagery of residential areas).
3. **Execute the Script**: Open a terminal or command prompt, navigate to the script’s directory, and run the script with the `--image` argument to specify the input image path:
   
   ```bash
   python detect_pools.py --image path/to/aerial_image.jpg
   ```
   
   - Replace `path/to/aerial_image.jpg` with the actual path to your image file.
   
4. **Check the Output**: After processing, the script will:
   - Save `coordinates.txt`: A text file listing the (x, y) coordinates of detected pool boundaries in the format "Point X: [x, y]".
   - Save `output_image.jpg`: The input image with red outlines around detected pools.
   - Optionally display the annotated image using Matplotlib if run in an environment that supports it (e.g., Jupyter notebook).
   - Print "No detections" if no pools are found.

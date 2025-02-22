# Swimming Pool Detection in Aerial Images

## Overview
This Python script detects swimming pools in aerial images, such as satellite or drone views of residential neighborhoods. It uses OpenCV (`cv2`) and NumPy (`np`) to analyze color, texture, and shape characteristics, identifying pools by their distinct blue/turquoise hues, smooth surfaces, and typical sizes/shapes. The script outlines detected pools in red on the input image and saves their boundary coordinates to a text file. It is designed to handle challenges like shadows, irregular pool shapes, and grey land, as seen in the provided aerial images.

## Process Explanation
The script processes an aerial image through the following steps:

1. **Input Handling**: The script accepts an aerial image path via a command-line argument and loads the image for processing.
2. **Color Segmentation**: The image is converted to HSV color space to isolate blue/turquoise pixels (hue: 80–120, saturation: 60–255, value: 80–255), typical of pool water. This step excludes grey land and shadows by requiring higher saturation levels.
3. **Noise Removal**: Morphological operations (opening and closing) with a 3x3 kernel (I found that this works the best) clean the mask by removing small noise (e.g., isolated blue pixels) and filling gaps in pool areas (e.g., caused by shadows or reflections).
4. **Texture Filtering**: A Laplacian filter identifies smooth areas with low texture variance (threshold < 150), ensuring only uniform surfaces like pools are retained, while textured regions like rooftops or lawns are filtered out.
5. **Contour Detection and Filtering**: Contours are extracted from the mask and filtered based on:
   - **Area**: Between 200 and 30,000 pixels to include small and large pools while excluding tiny noise or oversized non-pool regions.
   - **Shape**: Minimum circularity of 0.15 to accommodate irregular pool shapes (e.g., kidney-shaped pools in the images).
   - **Color Uniformity**: Saturation > 10 and hue variance < 200 to exclude grey land and ensure consistent pool-like colors.
6. **Output Generation**: Valid pool contours are drawn in red on the original image, and their precise (x, y) coordinates are collected. The annotated image and coordinates are then saved to files.

**Examples**:

| Sample Image | Output Image |
|-------------|-------------|
| ![000000079](https://github.com/user-attachments/assets/5ae01f72-aa1f-4149-a275-d16b7e0d72cb) | ![000000079_output](https://github.com/user-attachments/assets/57536561-666b-48dd-9070-0aab571feb83) |
| ![000000136](https://github.com/user-attachments/assets/4ffa0818-c67f-4938-904d-3088f3340d88) | ![000000136_output](https://github.com/user-attachments/assets/1655ffa6-ab45-42b4-9d8e-dd1b0f8ab501) |
| ![000000216](https://github.com/user-attachments/assets/27b4eaea-f3b3-4e44-9842-acf534228613) | ![000000216_output](https://github.com/user-attachments/assets/27a2160d-f091-46d1-83eb-344fef1a5b0b) |



This process successfully detects pools in images like the provided aerial views, where turquoise pools stand out against grey/brown roofs and green lawns, even with shadows or irregular shapes.

## How to Use

### Prerequisites
- **Python**: Ensure Python is installed on your system.
- **Libraries**: Install the required dependencies using pip:
  
  ```bash
  pip install opencv-python numpy
  ```

### Running the Script via Command Line (CL)
1. **Save the Script**: Save the provided code as `detect_pools.py`.
2. **Prepare Your Image**: Have an aerial image ready
3. **Execute the Script**: Open a terminal or command prompt, navigate to the script's directory, and run the script with the `--image` argument to specify the input image path:
   
   ```bash
   python detect_pools.py --image path/to/aerial_image.jpg
   ```
   
   - Replace `path/to/aerial_image.jpg` with the actual path to your image file.
     
5. **Check the Output**: After processing, the script will:
   - Save `coordinates.txt`: A text file listing the (x, y) coordinates of detected pool boundaries.
   - Save `output_image.jpg`: The input image with red outlines around detected pools.
   - Print a message indicating success (e.g., "Successfully processed...") or if no pools were detected.

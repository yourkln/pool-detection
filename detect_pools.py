import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import matplotlib.pyplot as plt

trained_model = YOLO('./best.pt')

parser = argparse.ArgumentParser(description="Detect swimming pools in aerial images")
parser.add_argument("--image", required=True, help="Path to aerial image")
args = parser.parse_args()

def detect_pools(image):
    """Applies refined pool detection relying on YOLO's initial detection."""

    filtered = cv2.edgePreservingFilter(image, flags=1, sigma_s=60, sigma_r=0.4)
    hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([80, 60, 80])
    upper_blue = np.array([120, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pool_coordinates = []
    output_image = image.copy()
    min_area = 200  
    max_area = 30000  
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            epsilon = 0.001 * cv2.arcLength(contour, True)
            smoothed_contour = cv2.approxPolyDP(contour, epsilon, True)
            cv2.drawContours(output_image, [smoothed_contour], -1, (0, 0, 255), 1)
            coordinates = contour.reshape(-1, 2).tolist()
            pool_coordinates.extend(coordinates)

    return pool_coordinates, output_image

def save_results(all_coordinates, final_image):
    """
    Save the detection results to files.
    - all_coordinates: List of all contour coordinates from all detected pools.
    - final_image: The final composite image with all contours overlaid.
    """
    
    coord_filename = "coordinates.txt"
    with open(coord_filename, 'w') as f:
        f.write("Pool boundary coordinates (x, y):\n")
        for i, coord in enumerate(all_coordinates):
            f.write(f"Point {i+1}: {coord}\n")

    image_filename = "output_image.jpg"
    cv2.imwrite(image_filename, final_image)

def process_image(image_path):
    """Detects pools using YOLO, applies contour refinement, and saves final results."""
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read image")

    results = trained_model.predict(image_path, save=False, verbose=False)
    
    final_image = image.copy()
    all_coordinates = []  
    enlargement_factor = 0.3

    status = 0

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  
            width = x2 - x1
            height = y2 - y1
            delta_x = int(width * enlargement_factor)
            delta_y = int(height * enlargement_factor)
        
            x1_enlarged = max(0, x1 - delta_x)  
            y1_enlarged = max(0, y1 - delta_y)  
            x2_enlarged = x2 + delta_x
            y2_enlarged = y2 + delta_y
            
            roi = image[y1_enlarged:y2_enlarged, x1_enlarged:x2_enlarged]
            coordinates, refined_image = detect_pools(roi)
            if coordinates:
                status=1
            
            final_image[y1_enlarged:y2_enlarged, x1_enlarged:x2_enlarged] = refined_image     
            adjusted_coordinates = [(x + x1_enlarged, y + y1_enlarged) for x, y in coordinates]
            all_coordinates.extend(adjusted_coordinates)

    if status:
        final_image_rgb = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
        save_results(all_coordinates, final_image)
    else:
        print("No detections")

process_image(args.image)
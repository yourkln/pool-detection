import cv2
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description="Detect swimming pools in aerial images")
parser.add_argument("--image", required=True, help="Put the Path to your aerial image")
args = parser.parse_args()

def detect_pools(image_path):
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read image")
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower_blue = np.array([80, 60, 80])  
    upper_blue = np.array([120, 255, 255])  
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    kernel = np.ones((3, 3), np.uint8)  
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)  
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)  
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.abs(laplacian)
    texture_mask = laplacian < 150  
    mask = cv2.bitwise_and(mask, mask, mask=texture_mask.astype(np.uint8) * 255)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  
    pool_coordinates = []
    output_image = image.copy()

    min_area = 200  
    max_area = 30000  
    min_circularity = 0.15  
    max_hue_variance = 200  
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            
            cnt_mask = np.zeros_like(mask)
            cv2.drawContours(cnt_mask, [contour], -1, 255, -1)
            hsv_inside = cv2.bitwise_and(hsv, hsv, mask=cnt_mask)
            s_inside = hsv_inside[:, :, 1][cnt_mask > 0]  
            if len(s_inside) > 0:
                avg_saturation = np.mean(s_inside)
                if avg_saturation < 10:  
                    continue
            
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > min_circularity:
                    h_inside = hsv_inside[:, :, 0][cnt_mask > 0]  
                    if len(h_inside) > 0:
                        hue_variance = np.var(h_inside)
                        if hue_variance < max_hue_variance:                            
                            cv2.drawContours(output_image, [contour], -1, (0, 0, 255), 1)  
                            coordinates = contour.reshape(-1, 2).tolist()
                            pool_coordinates.extend(coordinates)

    return pool_coordinates, output_image

def save_results(coordinates, output_image):
    
    coord_filename = "coordinates.txt"
    with open(coord_filename, 'w') as f:
        f.write("Pool boundary coordinates (x, y):\n")
        for i, coord in enumerate(coordinates):
            f.write(f"Point {i+1}: {coord}\n")
    
    image_filename = "output_image.jpg"
    cv2.imwrite(image_filename, output_image)

image_path = args.image

try:
    print(f"Processing {image_path}...")
    coordinates, output_image = detect_pools(image_path)
    
    if coordinates:
        save_results(coordinates, output_image)
        print(f"Successfully processed {image_path}")
    else:
        print(f"No pools detected in {image_path}")
        
except Exception as e:
    print(f"Error processing {image_path}: {str(e)}")
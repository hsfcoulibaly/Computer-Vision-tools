import cv2
import numpy as np
import os
from flask import current_app


def find_best_match_and_blur(image_path, templates_dir):
    """
    Finds the best matching template in the image, draws a bounding box,
    and then blurs the detected region.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None, "Error: Could not load source image."

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    best_match = {'max_val': -1, 'top_left': (0, 0), 'template_dims': (0, 0)}

    # Iterate over all 10 templates
    for i in range(1, 11):
        template_filename = f'template_{i}.jpg'  # Assuming .jpg, adjust if needed
        template_path = os.path.join(templates_dir, template_filename)

        # Check if the template file exists
        if not os.path.exists(template_path):
            continue

        template = cv2.imread(template_path, 0)  # Read template in grayscale
        w, h = template.shape[::-1]  # Template width and height

        # Perform Normalized Cross-Correlation
        # This generates a result map where brighter spots indicate a better match.
        result = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

        # Find the best match score and location for the current template
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Track the best match found across all templates
        if max_val > best_match['max_val']:
            best_match.update({
                'max_val': max_val,
                'top_left': max_loc,
                'template_dims': (w, h)
            })

    # --- Post-Processing: Blur the Detected Region (Task 3 requirement) ---

    # Set a robust threshold for detection (e.g., 80% correlation)
    DETECTION_THRESHOLD = 0.8

    if best_match['max_val'] >= DETECTION_THRESHOLD:
        top_left = best_match['top_left']
        w, h = best_match['template_dims']

        # Calculate bottom-right corner
        bottom_right = (top_left[0] + w, top_left[1] + h)

        # 1. Extract the Region of Interest (ROI)
        # Slicing the image array: [startY:endY, startX:endX]
        roi = img_bgr[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        # 2. Apply a Blur Filter (e.g., Gaussian Blur)
        # Using a large kernel size (e.g., 41x41) for a visible blur effect
        blurred_roi = cv2.GaussianBlur(roi, (41, 41), 0)

        # 3. Replace the original ROI with the blurred ROI
        img_bgr[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = blurred_roi

        # Optional: Draw the bounding box for visualization on the final output
        cv2.rectangle(img_bgr, top_left, bottom_right, (0, 255, 0), 2)

        return img_bgr, f"Detected object with score: {best_match['max_val']:.2f}. Region blurred."

    return img_bgr, f"No object detected above threshold ({DETECTION_THRESHOLD:.2f}). Max score: {best_match['max_val']:.2f}"
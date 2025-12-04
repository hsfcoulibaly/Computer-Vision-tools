import cv2
import numpy as np
import math
import os

def process_log_gradient(image_path, output_dir):
    """
    Calculates the Gradient Magnitude, Gradient Angle, and Laplacian of Gaussian
    for an image and saves the three resulting images to the output directory.

    Returns: A dictionary of the saved file paths.
    """
    try:
        # Load image and convert to grayscale
        img = cv2.imread(image_path)
        if img is None:
            return {"error": "Could not read input image."}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # --- 1. Sobel Gradient Calculation (Gradient Magnitude and Angle) ---

        # Calculate gradients in X and Y directions using Sobel filter (CV_64F for precision)
        # ksize=3 is the standard kernel size
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # a. Gradient Magnitude (G = sqrt(Gx^2 + Gy^2))
        magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        # Normalize magnitude to 8-bit image for visualization (0-255)
        magnitude_visual = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # b. Gradient Angle (Theta = atan2(Gy, Gx))
        # Note: atan2 returns angle in radians, from -pi to pi.
        angle = cv2.phase(sobelx, sobely, angleInDegrees=True)
        # Normalize angle to 8-bit image for visualization (0-255)
        # Angles range from 0 to 360 degrees. We map this range to 0-255.
        angle_visual = cv2.normalize(angle, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # --- 2. Laplacian of Gaussian (LoG) ---

        # c. LoG filtered image
        # Step 1: Apply Gaussian Blur to smooth the image and reduce noise
        # Note: Using a standard deviation (sigma) of 3.0
        gaussian_blur = cv2.GaussianBlur(gray, (0, 0), 3.0)

        # Step 2: Apply Laplacian filter (second derivative)
        # CV_64F for precision; ksize=5 is commonly used with LoG
        log_filter = cv2.Laplacian(gaussian_blur, cv2.CV_64F, ksize=5)

        # Normalize LoG for visualization (since values can be negative)
        log_visual = cv2.normalize(log_filter, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # --- 3. Save Results ---

        base_name = os.path.basename(image_path).rsplit('.', 1)[0]

        # 1. Magnitude
        mag_filename = f"{base_name}_mag.png"
        cv2.imwrite(os.path.join(output_dir, mag_filename), magnitude_visual)

        # 2. Angle
        angle_filename = f"{base_name}_angle.png"
        cv2.imwrite(os.path.join(output_dir, angle_filename), angle_visual)

        # 3. LoG
        log_filename = f"{base_name}_log.png"
        cv2.imwrite(os.path.join(output_dir, log_filename), log_visual)

        return {
            "original_file": os.path.basename(image_path),
            "magnitude_file": mag_filename,
            "angle_file": angle_filename,
            "log_file": log_filename
        }

    except Exception as e:
        print(f"Error during LoG/Gradient processing: {e}")
        return {"error": str(e)}


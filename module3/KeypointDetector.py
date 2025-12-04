import cv2
import numpy as np
import os


def detect_edge_keypoints(gray_img, output_path):
    """
    Detects edge keypoints using the Canny edge detector and draws them.

    Returns: The filename of the saved image.
    """
    # 1. Edge Detection using Canny
    # Canny is highly effective as an edge keypoint finder
    # Thresholds are chosen based on common practice (adjust if needed)
    edges = cv2.Canny(gray_img, 100, 200)

    # 2. Visualize: Convert the grayscale edge map back to color (BGR)
    edge_visual = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # 3. Mark Edges (Optional: if you wanted to draw circles instead of lines)
    # Since Canny returns a binary map, we just return the map as the visualization

    filename = os.path.basename(output_path)
    cv2.imwrite(output_path, edge_visual)
    return filename


def detect_corner_keypoints(gray_img, output_path):
    """
    Detects corner keypoints using the Harris Corner Detector.

    Returns: The filename of the saved image.
    """
    # 1. Harris Corner Detection
    # blockSize: size of the neighborhood considered for corner detection (2)
    # ksize: Aperture parameter for the Sobel operator (3)
    # k: Harris detector free parameter (0.04)
    dst = cv2.cornerHarris(gray_img, 2, 3, 0.04)

    # 2. Result Dilatation to mark corners easily (not necessary, but visually clearer)
    dst = cv2.dilate(dst, None)

    # 3. Visualize: Create a color image to draw the corners on
    corner_visual = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

    # 4. Threshold for an optimal value, marking the detected corners (RED dots)
    # The 'dst > 0.01 * dst.max()' finds the strongest 1% of the corner responses.
    corner_visual[dst > 0.01 * dst.max()] = [0, 0, 255]  # BGR color for Red

    filename = os.path.basename(output_path)
    cv2.imwrite(output_path, corner_visual)
    return filename


def process_keypoint_detection(image_path, output_dir):
    """
    Orchestrates the keypoint detection and returns the paths to the saved images.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"error": "Could not read input image."}

        # All keypoint detection is typically done on grayscale images
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        base_name = os.path.basename(image_path).rsplit('.', 1)[0]

        # --- Edge Detection ---
        edge_path = os.path.join(output_dir, f"{base_name}_edges.png")
        edge_filename = detect_edge_keypoints(gray, edge_path)

        # --- Corner Detection (Harris) ---
        corner_path = os.path.join(output_dir, f"{base_name}_corners.png")
        corner_filename = detect_corner_keypoints(gray, corner_path)

        return {
            "original_file": os.path.basename(image_path),
            "edge_file": edge_filename,
            "corner_file": corner_filename
        }

    except Exception as e:
        print(f"Error during Keypoint processing: {e}")
        return {"error": str(e)}


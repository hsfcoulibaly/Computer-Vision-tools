from flask import Blueprint, request, jsonify, render_template
import math
import numpy as np
import os
import sys

# Define the Blueprint. The 'module1' name is used for the URL prefix,
# and 'templates' specifies where its HTML files are located (module1/templates).
module1_bp = Blueprint(
    'module1',
    __name__,
    template_folder='templates',
    static_folder='static'
)

# --- Camera Calibration Dependency Setup ---

# The calibration file is expected to be in the main project root directory.
# We must adjust the path since this blueprint runs from a subdirectory.
# Using os.getcwd() gives the execution directory (the root).
calibration_file = os.path.join(os.getcwd(), 'camera_calibration.npz')

CALIBRATED_FX = None
CALIBRATED_FY = None

try:
    if not os.path.exists(calibration_file):
        # NOTE: Raising a FileNotFoundError here would crash the entire main app
        # We will set the values to 1.0 and print a warning instead,
        # allowing the main app to start.
        print(f"!!! WARNING: Calibration file '{calibration_file}' not found. Using default FX/FY of 1.0.",
              file=sys.stderr)
        CALIBRATED_FX = 1.0
        CALIBRATED_FY = 1.0
    else:
        calibration_data = np.load(calibration_file)
        camera_matrix = calibration_data['camera_matrix']
        CALIBRATED_FX = camera_matrix[0, 0]
        CALIBRATED_FY = camera_matrix[1, 1]

        print(f"--- Project Alpha Intrinsics Loaded (FX: {CALIBRATED_FX:.2f}, FY: {CALIBRATED_FY:.2f}) ---")

except Exception as e:
    print(f"!!! ERROR loading calibration data for Project Alpha: {e}", file=sys.stderr)
    CALIBRATED_FX = 1.0
    CALIBRATED_FY = 1.0


# --- Blueprint Routes ---

@module1_bp.route('/')
def index():
    """Main page for Project Alpha (accessible at /project/alpha)"""
    return render_template('module1/index.html', title="Project Alpha: Real-World Distance Calculator")


@module1_bp.route('/calculate', methods=['POST'])
def calculate_distance():
    """API endpoint to calculate real-world distance."""
    data = request.json
    points = data.get('points')
    displayed_dims = data.get('displayedDimensions')
    original_dims = data.get('originalDimensions')
    # Default camera distance in cm (original project default)
    camera_distance = float(data.get('cameraDistance', 61))

    if not points or len(points) < 2 or not displayed_dims or not original_dims:
        return jsonify(
            {'error': 'Insufficient data. Requires 2 points, displayed dimensions, and original dimensions.'}), 400

    if CALIBRATED_FX is None or CALIBRATED_FX == 0 or CALIBRATED_FY is None or CALIBRATED_FY == 0:
        return jsonify(
            {'error': 'Calibration data is missing or invalid (FX/FY = 0). Cannot calculate real distance.'}), 500

    # Calculate scaling factors between original image and displayed image
    scale_x = displayed_dims['width'] / original_dims['width']
    scale_y = displayed_dims['height'] / original_dims['height']

    p1_displayed = points[0]
    p2_displayed = points[1]

    # 1. Calculate pixel difference in displayed coordinates
    pixel_diff_x_displayed = p2_displayed['x'] - p1_displayed['x']
    pixel_diff_y_displayed = p2_displayed['y'] - p1_displayed['y']

    # 2. Scale back to the pixel difference on the ORIGINAL image/sensor size
    pixel_diff_x_original = pixel_diff_x_displayed / scale_x
    pixel_diff_y_original = pixel_diff_y_displayed / scale_y

    # 3. Use the Pinhole Camera Model formula (Real World Distance = Pixel Distance * Camera Distance / Focal Length)
    #     # This formula relates the pixel dimensions to the actual distance based on camera intrinsics.
    real_world_dist_x = (pixel_diff_x_original * camera_distance) / CALIBRATED_FX
    real_world_dist_y = (pixel_diff_y_original * camera_distance) / CALIBRATED_FY

    # 4. Calculate the total 3D distance (Pythagorean theorem)
    total_real_world_distance = math.sqrt(real_world_dist_x ** 2 + real_world_dist_y ** 2)

    result = {
        'realWorldDistanceX': f"{real_world_dist_x:.4f}",
        'realWorldDistanceY': f"{real_world_dist_y:.4f}",
        'totalRealWorldDistance': f"{total_real_world_distance:.4f}",
        'details': {
            'point1_displayed': p1_displayed,
            'point2_displayed': p2_displayed,
            'pixel_diff_original_x': f"{pixel_diff_x_original:.2f}",
            'pixel_diff_original_y': f"{pixel_diff_y_original:.2f}",
            'calibrated_fx': CALIBRATED_FX,
            'calibrated_fy': CALIBRATED_FY,
            'camera_distance': camera_distance,
            'scale_factors': {'x': f"{scale_x:.4f}", 'y': f"{scale_y:.4f}"}
        }
    }
    return jsonify(result)


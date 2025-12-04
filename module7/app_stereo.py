from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
import os

app = Flask(__name__)

# --- Global Stereo Projection Matrices ---
PROJ_MATRIX_L = None
PROJ_MATRIX_R = None
Q_MATRIX = None


def load_stereo_calibration():
    """Loads the required P_L, P_R, and Q matrices from the pose estimation script output."""
    global PROJ_MATRIX_L, PROJ_MATRIX_R, Q_MATRIX

    calibration_file = 'stereo_calibration.npz'

    if not os.path.exists(calibration_file):
        print("\n--- WARNING: STEREO CALIBRATION FILE NOT FOUND ---")
        print("Using DUMMY stereo matrices. Results will be highly inaccurate.")
        # Fallback DUMMY matrices for app startup demonstration
        B = 100.0
        PROJ_MATRIX_L = np.array([
            [1000, 0, 320, 0], [0, 1000, 240, 0], [0, 0, 1, 0]
        ], dtype=np.float64)
        PROJ_MATRIX_R = np.array([
            [1000, 0, 320, -1000 * B], [0, 1000, 240, 0], [0, 0, 1, 0]
        ], dtype=np.float64)
        Q_MATRIX = np.array([
            [1, 0, 0, -320], [0, 1, 0, -240], [0, 0, 0, 1000], [0, 0, 1 / B, 0]
        ], dtype=np.float64)

    else:
        try:
            calibration_data = np.load(calibration_file)
            PROJ_MATRIX_L = calibration_data['P_L']
            PROJ_MATRIX_R = calibration_data['P_R']
            Q_MATRIX = calibration_data['Q']
            print("\n--- Stereo Intrinsics Loaded from stereo_calibration.npz ---")
        except KeyError:
            print("\n--- ERROR: 'P_L', 'P_R', or 'Q' not found. Run 'estimate_pose.py'. ---")


load_stereo_calibration()


def triangulate_and_estimate(p1_L, p1_R, p2_L, p2_R, reference_depth_cm):
    """
    Performs stereo triangulation, then scales the 3D results based on a known reference depth.
    """
    if PROJ_MATRIX_L is None or PROJ_MATRIX_R is None:
        raise RuntimeError("Stereo projection matrices are not loaded. Cannot perform triangulation.")

    pts1 = np.array([[p1_L['x'], p2_L['x']], [p1_L['y'], p2_L['y']]], dtype=np.float64)
    pts2 = np.array([[p1_R['x'], p2_R['x']], [p1_R['y'], p2_R['y']]], dtype=np.float64)

    # 1. Triangulation (Results are in arbitrary, scaled units)
    points_4d = cv2.triangulatePoints(PROJ_MATRIX_L, PROJ_MATRIX_R, pts1, pts2)
    points_3d_scaled = (points_4d[:3] / points_4d[3]).T

    P1_3D_scaled = points_3d_scaled[0]
    P2_3D_scaled = points_3d_scaled[1]

    # 2. Determine Scale Factor (S) using the reference depth Z of Point 1
    Z1_scaled = P1_3D_scaled[2]

    if abs(Z1_scaled) < 1e-6:  # Check for near-zero depth, which causes division by zero
        raise ValueError("Calculated Z depth for P1 is near zero. Cannot establish scale.")

    # Calculate the scale factor S to convert arbitrary units to centimeters (cm)
    scale_factor_s = reference_depth_cm / Z1_scaled

    # 3. Apply Scale Factor to all 3D points
    P1_3D_cm = P1_3D_scaled * scale_factor_s
    P2_3D_cm = P2_3D_scaled * scale_factor_s

    # 4. Calculate Euclidean distance in centimeters
    distance_vector = P2_3D_cm - P1_3D_cm
    total_real_world_distance = np.linalg.norm(distance_vector)

    return {
        'P1_3D': P1_3D_cm.tolist(),
        'P2_3D': P2_3D_cm.tolist(),
        'realWorldDistanceX': distance_vector[0],
        'realWorldDistanceY': distance_vector[1],
        'realWorldDistanceZ': distance_vector[2],
        'totalRealWorldDistance': total_real_world_distance,
        'depth1': P1_3D_cm[2],
        'depth2': P2_3D_cm[2]
    }


@app.route('/calculate', methods=['POST'])
def calculate_distance():
    data = request.json
    points_L = data.get('points_L')
    points_R = data.get('points_R')
    displayed_dims = data.get('displayedDimensions')
    original_dims = data.get('originalDimensions')
    # New: Reference depth from the user
    reference_depth_cm = float(data.get('referenceDepthCm', 76.0))  # Default to 76 cm

    if not points_L or len(points_L) != 2 or not points_R or len(points_R) != 2:
        return jsonify({'error': 'Insufficient data. Requires 2 corresponding points for LEFT and RIGHT images.'}), 400

    # Convert displayed coordinates back to original pixel coordinates
    scale_x = original_dims['width'] / displayed_dims['width']
    scale_y = original_dims['height'] / displayed_dims['height']

    p1_L_orig = {'x': points_L[0]['x'] * scale_x, 'y': points_L[0]['y'] * scale_y}
    p2_L_orig = {'x': points_L[1]['x'] * scale_x, 'y': points_L[1]['y'] * scale_y}
    p1_R_orig = {'x': points_R[0]['x'] * scale_x, 'y': points_R[0]['y'] * scale_y}
    p2_R_orig = {'x': points_R[1]['x'] * scale_x, 'y': points_R[1]['y'] * scale_y}

    try:
        # Pass the reference depth for scaling
        result = triangulate_and_estimate(p1_L_orig, p1_R_orig, p2_L_orig, p2_R_orig, reference_depth_cm)
    except Exception as e:
        return jsonify({'error': f'Triangulation error: {str(e)}'}), 500

    output = {
        'totalRealWorldDistance': f"{result['totalRealWorldDistance']:.2f} cm",
        'realWorldDistanceX': f"{result['realWorldDistanceX']:.2f} cm",
        'realWorldDistanceY': f"{result['realWorldDistanceY']:.2f} cm",
        'realWorldDistanceZ': f"{result['realWorldDistanceZ']:.2f} cm",
        'point1_depth_Z': f"{result['depth1']:.2f} cm",
        'point2_depth_Z': f"{result['depth2']:.2f} cm",
        'P1_3D': f"({result['P1_3D'][0]:.2f}, {result['P1_3D'][1]:.2f}, {result['P1_3D'][2]:.2f}) cm",
        'P2_3D': f"({result['P2_3D'][0]:.2f}, {result['P2_3D'][1]:.2f}, {result['P2_3D'][2]:.2f}) cm",
    }
    return jsonify(output)


if __name__ == '__main__':
    app.run(debug=True)


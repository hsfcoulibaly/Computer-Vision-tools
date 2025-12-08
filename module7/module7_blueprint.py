from flask import Blueprint, request, jsonify, render_template
import numpy as np
import cv2
import os
import sys

# Define the Blueprint.
# The url_prefix is set in app.py when registering this blueprint.
module7_bp = Blueprint(
    'module7',
    __name__,
    template_folder='templates',
    static_folder='static'
)

# --- Global Stereo Projection Matrices ---
PROJ_MATRIX_L = None
PROJ_MATRIX_R = None
Q_MATRIX = None


def get_module_path():
    """Returns the absolute path to the module7 directory."""
    return os.path.dirname(os.path.abspath(__file__))


def load_stereo_calibration():
    """Loads the required P_L, P_R, and Q matrices from the pose estimation script output."""
    global PROJ_MATRIX_L, PROJ_MATRIX_R, Q_MATRIX

    # Use module-relative path for calibration file
    calibration_file = os.path.join(get_module_path(), 'stereo_calibration.npz')

    if not os.path.exists(calibration_file):
        print("\n--- WARNING: STEREO CALIBRATION FILE NOT FOUND ---", file=sys.stderr)
        print("Using DUMMY stereo matrices. Results will be highly inaccurate.", file=sys.stderr)
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
            print("\n--- Module7: Stereo Intrinsics Loaded from stereo_calibration.npz ---")
        except KeyError:
            print("\n--- ERROR: 'P_L', 'P_R', or 'Q' not found. Run 'estimate_pose.py'. ---", file=sys.stderr)


# Load calibration on module import
load_stereo_calibration()


def triangulate_points(points_L, points_R):
    """
    Triangulates multiple corresponding point pairs from stereo images.
    
    Args:
        points_L: List of {'x': x, 'y': y} dicts for left image
        points_R: List of {'x': x, 'y': y} dicts for right image
        
    Returns:
        numpy array of 3D points (N x 3)
    """
    if PROJ_MATRIX_L is None or PROJ_MATRIX_R is None:
        raise RuntimeError("Stereo projection matrices are not loaded.")
    
    n_points = len(points_L)
    pts_L = np.array([[p['x'] for p in points_L], [p['y'] for p in points_L]], dtype=np.float64)
    pts_R = np.array([[p['x'] for p in points_R], [p['y'] for p in points_R]], dtype=np.float64)
    
    # Triangulate all points at once
    points_4d = cv2.triangulatePoints(PROJ_MATRIX_L, PROJ_MATRIX_R, pts_L, pts_R)
    points_3d = (points_4d[:3] / points_4d[3]).T  # Convert to Nx3
    
    return points_3d


def scale_points_to_cm(points_3d, reference_depth_cm, reference_point_idx=0):
    """
    Scales 3D points from arbitrary units to centimeters using a reference depth.
    
    Args:
        points_3d: Nx3 array of 3D points
        reference_depth_cm: Known depth of reference point in cm
        reference_point_idx: Index of the reference point (default: 0)
        
    Returns:
        Nx3 array of scaled 3D points in cm
    """
    z_ref = points_3d[reference_point_idx, 2]
    
    if abs(z_ref) < 1e-6:
        raise ValueError("Reference point Z depth is near zero. Cannot establish scale.")
    
    scale_factor = reference_depth_cm / z_ref
    return points_3d * scale_factor


def calculate_distance_3d(p1, p2):
    """Calculate Euclidean distance between two 3D points."""
    return np.linalg.norm(np.array(p2) - np.array(p1))


def fit_circle_3d(points_3d):
    """
    Fits a circle to 3 or more 3D points (assuming they're roughly coplanar).
    Uses the circumcircle formula for 3 points.
    
    Args:
        points_3d: Nx3 array of 3D points (minimum 3)
        
    Returns:
        dict with 'center', 'radius', 'diameter'
    """
    if len(points_3d) < 3:
        raise ValueError("Need at least 3 points to fit a circle")
    
    # Use first 3 points for circle fitting
    A, B, C = points_3d[0], points_3d[1], points_3d[2]
    
    # Calculate side lengths
    a = calculate_distance_3d(B, C)  # opposite to A
    b = calculate_distance_3d(A, C)  # opposite to B
    c = calculate_distance_3d(A, B)  # opposite to C
    
    # Calculate circumradius using formula: R = (a*b*c) / (4*Area)
    # Area using Heron's formula
    s = (a + b + c) / 2
    area_squared = s * (s - a) * (s - b) * (s - c)
    
    if area_squared <= 0:
        raise ValueError("Points are collinear, cannot fit circle")
    
    area = np.sqrt(area_squared)
    radius = (a * b * c) / (4 * area)
    diameter = 2 * radius
    
    # Calculate center using barycentric coordinates
    # For circumcenter: weights are a², b², c² (opposite side squared)
    w_A = a * a * (b * b + c * c - a * a)
    w_B = b * b * (a * a + c * c - b * b)
    w_C = c * c * (a * a + b * b - c * c)
    w_sum = w_A + w_B + w_C
    
    if abs(w_sum) < 1e-10:
        center = (A + B + C) / 3  # Fallback to centroid
    else:
        center = (w_A * A + w_B * B + w_C * C) / w_sum
    
    return {
        'center': center.tolist(),
        'radius': radius,
        'diameter': diameter
    }


def calculate_polygon_edges(points_3d):
    """
    Calculates all edge lengths of a polygon defined by 3D vertices.
    
    Args:
        points_3d: Nx3 array of 3D vertices in order
        
    Returns:
        List of edge lengths
    """
    n = len(points_3d)
    edges = []
    
    for i in range(n):
        next_i = (i + 1) % n  # Wrap around to first point
        edge_length = calculate_distance_3d(points_3d[i], points_3d[next_i])
        edges.append(edge_length)
    
    return edges


def calculate_rectangle_dimensions(points_3d):
    """
    Calculates width and height of a rectangle from 4 corner points.
    Assumes points are in order (clockwise or counter-clockwise).
    
    Args:
        points_3d: 4x3 array of corner points
        
    Returns:
        dict with 'width', 'height', 'area', 'edges'
    """
    if len(points_3d) != 4:
        raise ValueError("Rectangle requires exactly 4 points")
    
    # Calculate all 4 edges
    edges = calculate_polygon_edges(points_3d)
    
    # For a rectangle, opposite edges should be equal
    # edges[0] and edges[2] are opposite, edges[1] and edges[3] are opposite
    width = (edges[0] + edges[2]) / 2  # Average of opposite edges
    height = (edges[1] + edges[3]) / 2
    
    # Ensure width >= height for consistency
    if width < height:
        width, height = height, width
    
    area = width * height
    
    return {
        'width': width,
        'height': height,
        'area': area,
        'edges': edges
    }


# --- Blueprint Routes ---

@module7_bp.route('/')
def index():
    """Main page for Module 6 - Stereo Object Size Estimation."""
    return render_template('module7/index_stereo.html')


@module7_bp.route('/calculate', methods=['POST'])
def calculate_distance():
    """API endpoint to calculate 3D distance from stereo image pairs (legacy 2-point)."""
    data = request.json
    points_L = data.get('points_L')
    points_R = data.get('points_R')
    displayed_dims = data.get('displayedDimensions')
    original_dims = data.get('originalDimensions')
    reference_depth_cm = float(data.get('referenceDepthCm', 76.0))

    if not points_L or len(points_L) != 2 or not points_R or len(points_R) != 2:
        return jsonify({'error': 'Requires 2 corresponding points for LEFT and RIGHT images.'}), 400

    # Scale coordinates from displayed to original
    scale_x = original_dims['width'] / displayed_dims['width']
    scale_y = original_dims['height'] / displayed_dims['height']

    points_L_scaled = [{'x': p['x'] * scale_x, 'y': p['y'] * scale_y} for p in points_L]
    points_R_scaled = [{'x': p['x'] * scale_x, 'y': p['y'] * scale_y} for p in points_R]

    try:
        points_3d = triangulate_points(points_L_scaled, points_R_scaled)
        points_3d_cm = scale_points_to_cm(points_3d, reference_depth_cm, 0)
        
        distance = calculate_distance_3d(points_3d_cm[0], points_3d_cm[1])
        distance_vector = points_3d_cm[1] - points_3d_cm[0]
        
    except Exception as e:
        return jsonify({'error': f'Triangulation error: {str(e)}'}), 500

    return jsonify({
        'totalRealWorldDistance': f"{distance:.2f} cm",
        'realWorldDistanceX': f"{distance_vector[0]:.2f} cm",
        'realWorldDistanceY': f"{distance_vector[1]:.2f} cm",
        'realWorldDistanceZ': f"{distance_vector[2]:.2f} cm",
        'point1_depth_Z': f"{points_3d_cm[0, 2]:.2f} cm",
        'point2_depth_Z': f"{points_3d_cm[1, 2]:.2f} cm",
        'P1_3D': f"({points_3d_cm[0, 0]:.2f}, {points_3d_cm[0, 1]:.2f}, {points_3d_cm[0, 2]:.2f}) cm",
        'P2_3D': f"({points_3d_cm[1, 0]:.2f}, {points_3d_cm[1, 1]:.2f}, {points_3d_cm[1, 2]:.2f}) cm",
    })


@module7_bp.route('/calculate_shape', methods=['POST'])
def calculate_shape():
    """
    API endpoint to calculate object dimensions based on shape type.
    
    Shape types:
    - 'rectangle': 4 corners → width, height, area
    - 'circle': 3 points on circumference → diameter, radius
    - 'polygon': N vertices → all edge lengths
    """
    data = request.json
    shape_type = data.get('shapeType', 'polygon')
    points_L = data.get('points_L', [])
    points_R = data.get('points_R', [])
    displayed_dims = data.get('displayedDimensions')
    original_dims = data.get('originalDimensions')
    reference_depth_cm = float(data.get('referenceDepthCm', 76.0))

    # Validate point counts
    min_points = {'rectangle': 4, 'circle': 3, 'polygon': 3}
    required = min_points.get(shape_type, 3)
    
    if len(points_L) < required or len(points_R) < required:
        return jsonify({
            'error': f'{shape_type.capitalize()} requires at least {required} points. Got {len(points_L)}.'
        }), 400

    if len(points_L) != len(points_R):
        return jsonify({'error': 'Left and Right point counts must match.'}), 400

    # Scale coordinates from displayed to original
    scale_x = original_dims['width'] / displayed_dims['width']
    scale_y = original_dims['height'] / displayed_dims['height']

    points_L_scaled = [{'x': p['x'] * scale_x, 'y': p['y'] * scale_y} for p in points_L]
    points_R_scaled = [{'x': p['x'] * scale_x, 'y': p['y'] * scale_y} for p in points_R]

    try:
        # Triangulate all points
        points_3d = triangulate_points(points_L_scaled, points_R_scaled)
        points_3d_cm = scale_points_to_cm(points_3d, reference_depth_cm, 0)
        
        # Calculate shape-specific measurements
        if shape_type == 'rectangle':
            if len(points_3d_cm) != 4:
                return jsonify({'error': 'Rectangle requires exactly 4 corner points.'}), 400
            
            result = calculate_rectangle_dimensions(points_3d_cm)
            
            return jsonify({
                'shapeType': 'rectangle',
                'width': f"{result['width']:.2f} cm",
                'height': f"{result['height']:.2f} cm",
                'area': f"{result['area']:.2f} cm²",
                'edges': [f"{e:.2f} cm" for e in result['edges']],
                'points3D': [
                    f"P{i+1}: ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}) cm" 
                    for i, p in enumerate(points_3d_cm)
                ],
                'avgDepth': f"{np.mean(points_3d_cm[:, 2]):.2f} cm"
            })
            
        elif shape_type == 'circle':
            result = fit_circle_3d(points_3d_cm)
            
            return jsonify({
                'shapeType': 'circle',
                'diameter': f"{result['diameter']:.2f} cm",
                'radius': f"{result['radius']:.2f} cm",
                'circumference': f"{np.pi * result['diameter']:.2f} cm",
                'area': f"{np.pi * result['radius']**2:.2f} cm²",
                'center': f"({result['center'][0]:.2f}, {result['center'][1]:.2f}, {result['center'][2]:.2f}) cm",
                'points3D': [
                    f"P{i+1}: ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}) cm" 
                    for i, p in enumerate(points_3d_cm)
                ],
                'avgDepth': f"{np.mean(points_3d_cm[:, 2]):.2f} cm"
            })
            
        else:  # polygon
            edges = calculate_polygon_edges(points_3d_cm)
            perimeter = sum(edges)
            
            return jsonify({
                'shapeType': 'polygon',
                'numVertices': len(points_3d_cm),
                'edges': [f"Edge {i+1}: {e:.2f} cm" for i, e in enumerate(edges)],
                'perimeter': f"{perimeter:.2f} cm",
                'points3D': [
                    f"P{i+1}: ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}) cm" 
                    for i, p in enumerate(points_3d_cm)
                ],
                'avgDepth': f"{np.mean(points_3d_cm[:, 2]):.2f} cm"
            })

    except Exception as e:
        return jsonify({'error': f'Calculation error: {str(e)}'}), 500

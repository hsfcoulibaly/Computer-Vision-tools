import cv2
import cv2.aruco as aruco
import numpy as np
import math
import base64
import os
from typing import List, Dict, Tuple

# --- OpenCV and ArUco Configuration ---
# Use the same dictionary ID you printed your markers from
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
ARUCO_PARAMS = aruco.DetectorParameters()


def calculate_centroid_and_angle(points):
    """
    Calculates the geometric centroid of a set of 2D points and
    sorts them based on the angle they form relative to the centroid.
    This ensures the polygon is drawn correctly without self-intersections.
    """
    if points.size == 0:
        return np.array([], dtype=np.int32)

    center_x = np.mean(points[:, 0])
    center_y = np.mean(points[:, 1])

    angles = []
    for point in points:
        angle = math.atan2(point[1] - center_y, point[0] - center_x)
        angles.append(angle)

    sorted_points = [point for _, point in sorted(zip(angles, points), key=lambda x: x[0])]

    return np.array(sorted_points, dtype=np.int32)


def process_image(image_data):
    """
    Processes the uploaded image data (as bytes), detects ArUco markers,
    draws the boundary, and returns the result as a Base64-encoded JPEG string
    and the number of markers found.
    """
    # Convert image data (bytes) to an OpenCV NumPy array
    np_array = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if frame is None:
        return "Error: Could not decode image. Please ensure it is a valid JPEG or PNG file.", 0

    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detector = aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)

    num_markers = 0

    if ids is not None:
        num_markers = len(ids)
        # Draw all detected markers (Green)
        frame = aruco.drawDetectedMarkers(frame, corners, ids, borderColor=(0, 255, 0))

        marker_centers = []
        for marker_corners in corners:
            center = np.mean(marker_corners[0], axis=0).astype(int)
            marker_centers.append(center)
            # Draw center point (Yellow)
            cv2.circle(frame, (center[0], center[1]), 8, (0, 255, 255), -1)

        marker_centers_np = np.array(marker_centers)
        ordered_points = calculate_centroid_and_angle(marker_centers_np)

        if ordered_points.size > 0:
            ordered_points_reshaped = ordered_points.reshape((-1, 1, 2))
            # Draw boundary polygon (Red)
            cv2.polylines(frame, [ordered_points_reshaped], isClosed=True, color=(0, 0, 255), thickness=5)

    # Encode the processed image back to Base64
    is_success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if is_success:
        jpg_as_text = base64.b64encode(buffer.tobytes()).decode('utf-8')
        return f"data:image/jpeg;base64,{jpg_as_text}", num_markers
    else:
        return "Error: Could not encode processed image.", 0


def process_image_from_path(image_path: str) -> Tuple[bool, int, np.ndarray, str]:
    """
    Processes an image from file path, detects ArUco markers, and returns results.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        tuple: (success, num_markers, processed_image, error_message)
    """
    try:
        frame = cv2.imread(image_path)
        if frame is None:
            return False, 0, None, f"Could not load image: {image_path}"
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        detector = aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
        corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
        
        num_markers = 0
        
        if ids is not None:
            num_markers = len(ids)
            # Draw all detected markers (Green)
            frame = aruco.drawDetectedMarkers(frame, corners, ids, borderColor=(0, 255, 0))
            
            marker_centers = []
            for marker_corners in corners:
                center = np.mean(marker_corners[0], axis=0).astype(int)
                marker_centers.append(center)
                # Draw center point (Yellow)
                cv2.circle(frame, (center[0], center[1]), 8, (0, 255, 255), -1)
            
            if len(marker_centers) >= 3:  # Need at least 3 points for a polygon
                marker_centers_np = np.array(marker_centers)
                ordered_points = calculate_centroid_and_angle(marker_centers_np)
                
                if ordered_points.size > 0:
                    ordered_points_reshaped = ordered_points.reshape((-1, 1, 2))
                    # Draw boundary polygon (Red)
                    cv2.polylines(frame, [ordered_points_reshaped], isClosed=True, color=(0, 0, 255), thickness=5)
        
        return True, num_markers, frame, ""
        
    except Exception as e:
        return False, 0, None, f"Error processing {image_path}: {str(e)}"


def batch_evaluate_images(image_paths: List[str], metadata: List[Dict] = None) -> Dict:
    """
    Evaluates ArUco marker detection on multiple images and computes statistics.
    
    Args:
        image_paths: List of paths to image files
        metadata: Optional list of dictionaries with metadata (distance, angle, etc.)
        
    Returns:
        dict: Evaluation results with statistics and per-image results
    """
    results = {
        'total_images': len(image_paths),
        'successful_detections': 0,
        'failed_detections': 0,
        'total_markers_detected': 0,
        'avg_markers_per_image': 0.0,
        'detection_rate': 0.0,
        'min_markers': float('inf'),
        'max_markers': 0,
        'image_results': []
    }
    
    if metadata is None:
        metadata = [{}] * len(image_paths)
    
    for idx, (image_path, meta) in enumerate(zip(image_paths, metadata)):
        filename = os.path.basename(image_path)
        success, num_markers, processed_img, error = process_image_from_path(image_path)
        
        # Save processed image if successful
        processed_path = None
        if success and processed_img is not None:
            # Save to same directory with 'processed_' prefix
            dir_path = os.path.dirname(image_path)
            base_name = os.path.splitext(filename)[0]
            ext = os.path.splitext(filename)[1]
            processed_filename = f"processed_{base_name}{ext}"
            processed_path = os.path.join(dir_path, processed_filename)
            try:
                cv2.imwrite(processed_path, processed_img)
            except Exception as e:
                print(f"Warning: Could not save processed image {processed_path}: {e}")
                processed_path = None
        
        image_result = {
            'index': idx + 1,
            'filename': filename,
            'success': success,
            'num_markers': num_markers if success else 0,
            'error': error if not success else None,
            'processed_path': processed_path,
            'distance': meta.get('distance', 'N/A'),
            'angle': meta.get('angle', 'N/A'),
            'notes': meta.get('notes', '')
        }
        
        results['image_results'].append(image_result)
        
        if success:
            results['successful_detections'] += 1
            results['total_markers_detected'] += num_markers
            results['min_markers'] = min(results['min_markers'], num_markers)
            results['max_markers'] = max(results['max_markers'], num_markers)
        else:
            results['failed_detections'] += 1
    
    # Calculate statistics
    if results['total_images'] > 0:
        results['detection_rate'] = (results['successful_detections'] / results['total_images']) * 100.0
    
    if results['successful_detections'] > 0:
        results['avg_markers_per_image'] = results['total_markers_detected'] / results['successful_detections']
    
    if results['min_markers'] == float('inf'):
        results['min_markers'] = 0
    
    return results


import cv2
import cv2.aruco as aruco
import numpy as np
import math
import base64

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


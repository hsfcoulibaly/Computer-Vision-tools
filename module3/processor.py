import cv2
import cv2.aruco as aruco
import numpy as np

# Dictionary to map exercise ID to a display name and the processing function
EXERCISE_MODES = {
    '1': ('Gradient & LoG', 'process_gradient_log'),
    '2': ('Canny & Harris Keypoints', 'process_keypoints'),
    '3': ('Object Boundary (Contours)', 'process_boundary'),
    '4': ('ArUco Segmentation', 'process_aruco_segmentation')
}

class VideoProcessor:
    def __init__(self, camera_index=0):
        self.video = cv2.VideoCapture(camera_index)
        if not self.video.isOpened():
            print("Error: Could not open video stream. Check camera index.")

        # ArUco Setup (for Exercise 4)
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_params = aruco.DetectorParameters()
        self.aruco_detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

    def __del__(self):
        self.video.release()

    # Exercise 1: Gradient & LoG [cite: 18, 19]
    def process_gradient_log(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

        magnitude, angle = cv2.cartToPolar(sobelx, sobely, angleInDegrees=True)
        cv2.normalize(magnitude, magnitude, 0, 255, cv2.NORM_MINMAX)
        magnitude_8u = np.uint8(magnitude)
        angle_8u = np.uint8(angle / 360 * 255)

        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        log = cv2.Laplacian(blur, cv2.CV_64F, ksize=5)
        cv2.normalize(np.absolute(log), log, 0, 255, cv2.NORM_MINMAX)
        log_8u = np.uint8(log)

        # Visualization setup (Stacking images)
        h, w = gray.shape[:2]
        new_h = int(h / 2)

        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        mag_3ch = cv2.cvtColor(magnitude_8u, cv2.COLOR_GRAY2BGR)
        angle_3ch = cv2.cvtColor(angle_8u, cv2.COLOR_GRAY2BGR)
        log_3ch = cv2.cvtColor(log_8u, cv2.COLOR_GRAY2BGR)

        top_row = cv2.resize(np.hstack([gray_3ch, mag_3ch]), (w, new_h))
        bottom_row = cv2.resize(np.hstack([angle_3ch, log_3ch]), (w, new_h))
        final_output = np.vstack([top_row, bottom_row])

        # Labels for verification
        cv2.putText(final_output, 'Original | Magnitude', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(final_output, 'Angle | LoG', (10, new_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return final_output

    # Exercise 2: Canny & Harris [cite: 21, 22]
    def process_keypoints(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- EDGE DETECTION (Canny) ---
        # The Canny algorithm is the standard simple, effective edge detector.
        canny = cv2.Canny(gray, threshold1=100, threshold2=200)

        # --- CORNER DETECTION (Harris) ---
        # Harris is a classic, simple corner detection algorithm.
        harris_dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

        # Dilate for visibility and normalize
        harris_dst = cv2.dilate(harris_dst, None)
        cv2.normalize(harris_dst, harris_dst, 0, 255, cv2.NORM_MINMAX)
        harris_8u = np.uint8(harris_dst)

        # Draw corners on the original color frame for better visualization
        corner_frame = frame.copy()
        # Threshold the Harris response to draw keypoints
        corner_frame[harris_8u > 0.01 * harris_8u.max()] = [0, 0, 255]  # Mark corners in red

        # Visualization setup
        canny_3ch = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)

        h, w = frame.shape[:2]
        new_h = int(h / 2)

        top_row = cv2.resize(np.hstack([frame, canny_3ch]), (w, new_h))
        bottom_row = cv2.resize(np.hstack([corner_frame, np.zeros_like(frame)]),
                                (w, new_h))  # Fill last quadrant with black
        final_output = np.vstack([top_row, bottom_row])

        # Labels
        cv2.putText(final_output, 'Original | Canny Edges', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(final_output, 'Harris Corners', (10, new_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return final_output

    # Exercise 3: Boundaries [cite: 23]
    def process_boundary(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Preprocessing: Use simple thresholding to get a binary mask of the object
        # Assuming the object is darker/lighter than the background
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour (most likely the main object)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

            # Draw the boundary on the original color frame
            contour_frame = frame.copy()
            # Draw the contour (boundary) in green with thickness 3
            cv2.drawContours(contour_frame, [largest_contour], -1, (0, 255, 0), 3)

            # Draw the bounding box (optional, for verification)
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(contour_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue box
        else:
            contour_frame = frame.copy()

        # Visualization
        cv2.putText(contour_frame, 'Object Boundary (Largest Contour)', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2)
        return contour_frame

    # Exercise 4: ArUco Segmentation [cite: 26, 29]
    def process_aruco_segmentation(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)

        segmentation_frame = frame.copy()

        # Check if markers were found
        if ids is not None:
            # Draw the detected markers (optional, but useful for verification)
            aruco.drawDetectedMarkers(segmentation_frame, corners, ids)

            # Extract marker centers to define the boundary points
            boundary_points = []

            # Assuming the markers are stuck on the object boundary and are ordered
            # We'll use the center of each detected marker as a boundary point
            for corner in corners:
                # Calculate the center (mean of the four corner points)
                center_x = int(np.mean(corner[0, :, 0]))
                center_y = int(np.mean(corner[0, :, 1]))
                boundary_points.append([center_x, center_y])

                # Draw a circle on the center point (keypoint)
                cv2.circle(segmentation_frame, (center_x, center_y), 5, (0, 0, 255), -1)  # Red circle

            # Convert points to NumPy array and reshape for cv2.polylines
            if len(boundary_points) > 2:
                points_np = np.array(boundary_points, dtype=np.int32).reshape((-1, 1, 2))

                # Draw the segmented boundary by connecting the points (closed polygon)
                # isClosed=True ensures the boundary is segmented completely
                cv2.polylines(segmentation_frame, [points_np], isClosed=True, color=(255, 0, 0),
                              thickness=3)  # Blue line

        # Labels
        cv2.putText(segmentation_frame, 'ArUco Boundary Segmentation', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 0), 2)

        return segmentation_frame

    def get_frame(self, exercise_mode):
        """Reads a frame and calls the corresponding processing function."""
        success, frame = self.video.read()
        if not success:
            return None

        # Call the processing function based on the mode ID
        try:
            func_name = EXERCISE_MODES.get(exercise_mode)[1]
            processing_func = getattr(self, func_name)
            processed_frame = processing_func(frame)
        except Exception as e:
            # Fallback to original frame if mode is invalid or error occurs
            processed_frame = frame
            cv2.putText(processed_frame, f'Error in mode {exercise_mode}: {e}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)
            print(f"Error processing frame: {e}")

        # Encode the processed frame to JPEG format for web streaming
        ret, jpeg = cv2.imencode('.jpg', processed_frame)
        return jpeg.tobytes()
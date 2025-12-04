# calibrate_camera.py
import numpy as np
import cv2
import glob
import os

# --- Define the dimensions of the checkerboard ---
# This is the number of INTERIOR corners. For a 10x7 grid of squares,
# you will have 9x6 interior corners.
CHECKERBOARD = (9, 6)

# --- Create a directory to save calibration images if it doesn't exist ---
IMAGE_DIR = 'calibration_images'
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)
    print(f"Created directory: {IMAGE_DIR}")
    exit()

# --- Calibration Setup ---
# Termination criteria for the corner sub-pixel algorithm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
# These are the 3D coordinates of the checkerboard corners in an ideal, flat world.
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob(f'{IMAGE_DIR}/*.jpg') # You can change to .png, etc.

if not images:
    print(f"Error: No images found in the '{IMAGE_DIR}' directory. Please add your calibration photos.")
    exit()

print(f"Found {len(images)} images. Processing...")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    # After
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)
        # Refine the corner locations for better accuracy
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        print(f"✓ Corners found in {os.path.basename(fname)}")
    else:
        print(f"✗ Corners not found in {os.path.basename(fname)}. Skipping this image.")

if not imgpoints:
    print("\nError: Could not find corners in any of the images. Calibration failed.")
    print("Tips: Ensure good lighting, a flat checkerboard, and clear, non-blurry photos.")
    exit()

print("\nPerforming camera calibration...")
# The core calibration function
# It returns the camera matrix, distortion coefficients, rotation and translation vectors
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# --- Save the calibration result ---
output_file = 'camera_calibration.npz'
np.savez(output_file, camera_matrix=mtx, dist_coeffs=dist)

print("\n--- Calibration Successful! ---")
print(f"Results saved to '{output_file}'")
print("\nCamera Matrix (mtx):\n", mtx)
print("\nFocal Length (fx, fy):", (mtx[0, 0], mtx[1, 1]))
print("\nOptical Center (cx, cy):", (mtx[0, 2], mtx[1, 2]))
print("\nDistortion Coefficients (dist):\n", dist)
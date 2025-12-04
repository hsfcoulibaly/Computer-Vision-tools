import os
import time
from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
from algorithms.sift import compute_sift_features, match_features
from algorithms.ransac import ransac_homography
from algorithms.stitching import custom_stitch_images

# --- Configuration ---
UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# --- Helper Function to Align and Stack Images (Fixes Black Bars) ---

def align_and_stack_images(img1, img2):
    """Resizes images to have equal height and stacks them horizontally."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 1. Standardize heights by resizing the smaller one to match the taller one
    if h1 != h2:
        target_h = max(h1, h2)

        # Resize img1
        if h1 != target_h:
            new_w1 = int(w1 * (target_h / h1))
            img1 = cv2.resize(img1, (new_w1, target_h), interpolation=cv2.INTER_LINEAR)

        # Resize img2
        if h2 != target_h:
            new_w2 = int(w2 * (target_h / h2))
            img2 = cv2.resize(img2, (new_w2, target_h), interpolation=cv2.INTER_LINEAR)

    # 2. Stack the now equally-sized images horizontally
    stacked_image = np.hstack((img1, img2))
    return stacked_image


def draw_matches_on_image(img1_color, kps1, img2_color, kps2, raw_matches, inlier_indices, output_path):
    """
    Draws keypoints and inlier match lines on a stacked image and saves it.
    kps1 and kps2 must be NumPy arrays of (x, y) coordinates.
    """
    # Align and stack the images first
    stacked_image = align_and_stack_images(img1_color, img2_color)
    h1, w1 = img1_color.shape[:2]  # Use original heights/widths for offset calculations

    # Ensure kps are float arrays (x, y)
    kps1 = np.float32(kps1)
    kps2 = np.float32(kps2)

    # Draw matches (only inliers)
    for i in inlier_indices:
        (idx1, idx2) = raw_matches[i]

        # Get coordinates for the match
        pt1 = tuple(map(int, kps1[idx1]))
        pt2 = tuple(map(int, kps2[idx2]))

        # Adjust pt2 x-coordinate by the width of the first image (w1)
        pt2_shifted = (pt2[0] + w1, pt2[1])

        # Draw line (random color)
        color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
        cv2.line(stacked_image, pt1, pt2_shifted, color, 1)

        # Draw circles for keypoints
        cv2.circle(stacked_image, pt1, 3, color, -1)
        cv2.circle(stacked_image, pt2_shifted, 3, color, -1)

    cv2.imwrite(output_path, stacked_image)
    return output_path


# --- SIFT/RANSAC Comparison Logic ---

def compare_sift_implementations(path1, path2):
    """
    Orchestrates the custom and open-source (OpenCV) SIFT/RANSAC comparison.
    """
    img1_color = cv2.imread(path1, cv2.IMREAD_COLOR)
    img2_color = cv2.imread(path2, cv2.IMREAD_COLOR)
    img1_gray = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

    # --- 1. CUSTOM SIFT & RANSAC ---
    start_time_custom = time.time()

    # 1. Feature Detection (calls algorithms/sift.py)
    kps1_custom_np, descs1_custom = compute_sift_features(path1)
    kps2_custom_np, descs2_custom = compute_sift_features(path2)

    # 2. Feature Matching (calls algorithms/sift.py)
    raw_matches_custom = match_features(descs1_custom, descs2_custom)

    # --- RANSAC outputs ---
    inliers_custom_indices = np.array([])
    H_custom = np.identity(3)

    if len(raw_matches_custom) >= 4:
        # 3. Extract matched coordinates for RANSAC
        src_pts_custom = np.float32([kps1_custom_np[m[0]] for m in raw_matches_custom]).reshape(-1, 2)
        dst_pts_custom = np.float32([kps2_custom_np[m[1]] for m in raw_matches_custom]).reshape(-1, 2)

        # 4. RANSAC Optimization (calls algorithms/ransac.py)
        H_custom, inliers_custom_indices = ransac_homography(src_pts_custom, dst_pts_custom)

    end_time_custom = time.time()
    time_custom = end_time_custom - start_time_custom

    # Draw Custom Matches
    custom_output_path = os.path.join(app.config['RESULTS_FOLDER'], 'custom_sift_matches_vis.jpg')
    # NOTE: raw_matches_custom contains the indices of the keypoints used in the match list,
    # not the full original keypoints list, but for visualization here we pass the keypoints
    # themselves and rely on the indices in the raw_matches list.
    draw_matches_on_image(img1_color, kps1_custom_np, img2_color, kps2_custom_np,
                          raw_matches_custom, inliers_custom_indices, custom_output_path)
    # --- END CUSTOM SIFT & RANSAC ---

    # --- 2. OPEN-SOURCE (OpenCV) SIFT & RANSAC ---
    start_time_cv = time.time()

    # 1. Feature Detection (OpenCV)
    sift_cv = cv2.SIFT_create()
    kps1_cv_raw, descs1_cv = sift_cv.detectAndCompute(img1_gray, None)
    kps2_cv_raw, descs2_cv = sift_cv.detectAndCompute(img2_gray, None)

    kps1_cv_np = np.float32([kp.pt for kp in kps1_cv_raw])
    kps2_cv_np = np.float32([kp.pt for kp in kps2_cv_raw])

    raw_matches_cv = []
    inliers_cv_indices = np.array([])

    # 2. Feature Matching (OpenCV Brute-Force with Ratio Test)
    if descs1_cv is not None and descs2_cv is not None and descs1_cv.shape[0] > 1 and descs2_cv.shape[0] > 1:
        bf = cv2.BFMatcher()
        raw_matches_cv_pre = bf.knnMatch(descs1_cv, descs2_cv, k=2)

        # Apply Lowe's Ratio Test
        for m, n in raw_matches_cv_pre:
            if m.distance < 0.75 * n.distance:
                raw_matches_cv.append((m.queryIdx, m.trainIdx))  # Store (kps1_idx, kps2_idx)

        # 3. RANSAC Optimization (OpenCV)
        if len(raw_matches_cv) >= 4:
            # Prepare points for cv2.findHomography (needs (N, 1, 2) shape for masks)
            src_pts_cv_in = np.float32([kps1_cv_np[m[0]] for m in raw_matches_cv]).reshape(-1, 1, 2)
            dst_pts_cv_in = np.float32([kps2_cv_np[m[1]] for m in raw_matches_cv]).reshape(-1, 1, 2)

            # Use OpenCV's RANSAC
            M_cv, mask_cv = cv2.findHomography(src_pts_cv_in, dst_pts_cv_in, cv2.RANSAC, 5.0)

            # Convert mask to a list of inlier indices
            inliers_cv_indices = np.where(mask_cv.ravel() == 1)[0]

    end_time_cv = time.time()
    time_cv = end_time_cv - start_time_cv
    # --- END OPEN-SOURCE SIFT & RANSAC ---

    # Draw OpenCV Matches
    cv_output_path = os.path.join(app.config['RESULTS_FOLDER'], 'cv_sift_matches_vis.jpg')
    # NOTE: The coordinates kps1_cv_np and kps2_cv_np are correct here
    draw_matches_on_image(img1_color, kps1_cv_np, img2_color, kps2_cv_np,
                          raw_matches_cv, inliers_cv_indices, cv_output_path)

    # 3. Collect Results
    results = {
        'custom_keypoints': len(kps1_custom_np),
        'open_cv_keypoints': len(kps1_cv_raw),
        'custom_matches': len(inliers_custom_indices),
        'open_cv_matches': len(inliers_cv_indices),
        'custom_time': time_custom,
        'open_cv_time': time_cv,
        'custom_match_img': url_for('static', filename=f'results/{os.path.basename(custom_output_path)}'),
        'open_cv_match_img': url_for('static', filename=f'results/{os.path.basename(cv_output_path)}')
    }
    return results


# --- Routes ---

@app.route('/')
def index():
    """Landing page with links to the two modules."""
    return render_template('index.html')


# --- Image Stitching Module ---

@app.route('/stitch', methods=['GET', 'POST'])
def stitch_images():
    if request.method == 'POST':
        camera_files = request.files.getlist('camera_images')
        mobile_file = request.files.get('mobile_panorama')

        uploaded_paths = []

        # 1. Save Camera Images
        for file in camera_files:
            if file and allowed_file(file.filename):
                filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filename)
                uploaded_paths.append(filename)

        # 2. Save Mobile Panorama Image
        mobile_path = None
        if mobile_file and allowed_file(mobile_file.filename):
            mobile_path = os.path.join(app.config['UPLOAD_FOLDER'], 'mobile_pano.jpg')
            mobile_file.save(mobile_path)

        if len(uploaded_paths) < 2:  # Changed from 4 to 2 to allow testing with fewer images, though 4/8 is required for the project
            return "Error: Please upload at least 2 camera images.", 400

        # CALL YOUR CUSTOM STITCHING FUNCTION HERE
        stitched_image_path = custom_stitch_images(uploaded_paths)

        # Path returned by custom_stitch_images is 'static/results/stitched_custom.jpg'
        custom_img_url_path = os.path.basename(stitched_image_path)

        return render_template('stitch_results.html',
                               custom_img_url=url_for('static', filename=f'results/{custom_img_url_path}'),
                               mobile_img_url=url_for('static',
                                                      filename=f'uploads/{os.path.basename(mobile_path)}' if mobile_path else ''))

    return render_template('stitch.html')


# --- SIFT Comparison Module ---

@app.route('/sift_compare', methods=['GET', 'POST'])
def sift_compare():
    if request.method == 'POST':
        img1 = request.files.get('image1')
        img2 = request.files.get('image2')

        path1, path2 = None, None

        # Save files using temporary names
        if img1 and allowed_file(img1.filename):
            path1 = os.path.join(app.config['UPLOAD_FOLDER'], 'sift_img1.jpg')
            img1.save(path1)

        if img2 and allowed_file(img2.filename):
            path2 = os.path.join(app.config['UPLOAD_FOLDER'], 'sift_img2.jpg')
            img2.save(path2)

        if not path1 or not path2:
            return "Error: Please upload two images for SIFT comparison.", 400

        # CALL THE COMPARISON ORCHESTRATOR
        results = compare_sift_implementations(path1, path2)

        return render_template('sift_results.html', results=results)

    return render_template('sift_compare.html')


if __name__ == '__main__':
    app.run(debug=True)

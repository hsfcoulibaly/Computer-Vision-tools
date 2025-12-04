import os
import sys
import time
from flask import Blueprint, render_template, request, url_for
import cv2
import numpy as np

# Import helper functions from the local algorithms directory
try:
    from .algorithms.sift import compute_sift_features, match_features
    from .algorithms.ransac import ransac_homography
    from .algorithms.stitching import custom_stitch_images
except ImportError:
    print(
        "!!! ERROR: Could not import algorithm modules. Make sure 'module4/algorithms/' exists with sift.py, ransac.py, and stitching.py.",
        file=sys.stderr)

# Define the Blueprint.
# The url_prefix is set in app.py when registering this blueprint.
module4_bp = Blueprint(
    'module4',
    __name__,
    template_folder='templates',
    static_folder='static'
)

# Configuration
UPLOAD_FOLDER_REL_PATH = 'module4/static/uploads'
RESULTS_FOLDER_REL_PATH = 'module4/static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


# --- Utility Functions ---

def get_upload_path():
    """Returns the absolute path to the module4 uploads folder."""
    return os.path.join(os.getcwd(), UPLOAD_FOLDER_REL_PATH)


def get_results_path():
    """Returns the absolute path to the module4 results folder."""
    return os.path.join(os.getcwd(), RESULTS_FOLDER_REL_PATH)


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# --- Helper Function to Align and Stack Images ---

def align_and_stack_images(img1, img2):
    """Resizes images to have equal height and stacks them horizontally."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    if h1 != h2:
        target_h = max(h1, h2)

        if h1 != target_h:
            new_w1 = int(w1 * (target_h / h1))
            img1 = cv2.resize(img1, (new_w1, target_h), interpolation=cv2.INTER_LINEAR)

        if h2 != target_h:
            new_w2 = int(w2 * (target_h / h2))
            img2 = cv2.resize(img2, (new_w2, target_h), interpolation=cv2.INTER_LINEAR)

    stacked_image = np.hstack((img1, img2))
    return stacked_image


def draw_matches_on_image(img1_color, kps1, img2_color, kps2, raw_matches, inlier_indices, output_path):
    """
    Draws keypoints and inlier match lines on a stacked image and saves it.
    """
    stacked_image = align_and_stack_images(img1_color, img2_color)
    h1, w1 = img1_color.shape[:2]

    kps1 = np.float32(kps1)
    kps2 = np.float32(kps2)

    for i in inlier_indices:
        (idx1, idx2) = raw_matches[i]

        pt1 = tuple(map(int, kps1[idx1]))
        pt2 = tuple(map(int, kps2[idx2]))

        pt2_shifted = (pt2[0] + w1, pt2[1])

        color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
        cv2.line(stacked_image, pt1, pt2_shifted, color, 1)

        cv2.circle(stacked_image, pt1, 3, color, -1)
        cv2.circle(stacked_image, pt2_shifted, 3, color, -1)

    cv2.imwrite(output_path, stacked_image)
    return output_path


# --- SIFT/RANSAC Comparison Logic ---

def compare_sift_implementations(path1, path2):
    """
    Orchestrates the custom and open-source (OpenCV) SIFT/RANSAC comparison.
    """
    results_path = get_results_path()
    os.makedirs(results_path, exist_ok=True)

    img1_color = cv2.imread(path1, cv2.IMREAD_COLOR)
    img2_color = cv2.imread(path2, cv2.IMREAD_COLOR)
    img1_gray = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

    # --- 1. CUSTOM SIFT & RANSAC ---
    start_time_custom = time.time()

    kps1_custom_np, descs1_custom = compute_sift_features(path1)
    kps2_custom_np, descs2_custom = compute_sift_features(path2)

    raw_matches_custom = match_features(descs1_custom, descs2_custom)

    inliers_custom_indices = np.array([])
    H_custom = np.identity(3)

    if len(raw_matches_custom) >= 4:
        src_pts_custom = np.float32([kps1_custom_np[m[0]] for m in raw_matches_custom]).reshape(-1, 2)
        dst_pts_custom = np.float32([kps2_custom_np[m[1]] for m in raw_matches_custom]).reshape(-1, 2)

        H_custom, inliers_custom_indices = ransac_homography(src_pts_custom, dst_pts_custom)

    end_time_custom = time.time()
    time_custom = end_time_custom - start_time_custom

    custom_output_path = os.path.join(results_path, 'custom_sift_matches_vis.jpg')
    draw_matches_on_image(img1_color, kps1_custom_np, img2_color, kps2_custom_np,
                          raw_matches_custom, inliers_custom_indices, custom_output_path)

    # --- 2. OPEN-SOURCE (OpenCV) SIFT & RANSAC ---
    start_time_cv = time.time()

    sift_cv = cv2.SIFT_create()
    kps1_cv_raw, descs1_cv = sift_cv.detectAndCompute(img1_gray, None)
    kps2_cv_raw, descs2_cv = sift_cv.detectAndCompute(img2_gray, None)

    kps1_cv_np = np.float32([kp.pt for kp in kps1_cv_raw])
    kps2_cv_np = np.float32([kp.pt for kp in kps2_cv_raw])

    raw_matches_cv = []
    inliers_cv_indices = np.array([])

    if descs1_cv is not None and descs2_cv is not None and descs1_cv.shape[0] > 1 and descs2_cv.shape[0] > 1:
        bf = cv2.BFMatcher()
        raw_matches_cv_pre = bf.knnMatch(descs1_cv, descs2_cv, k=2)

        for m, n in raw_matches_cv_pre:
            if m.distance < 0.75 * n.distance:
                raw_matches_cv.append((m.queryIdx, m.trainIdx))

        if len(raw_matches_cv) >= 4:
            src_pts_cv_in = np.float32([kps1_cv_np[m[0]] for m in raw_matches_cv]).reshape(-1, 1, 2)
            dst_pts_cv_in = np.float32([kps2_cv_np[m[1]] for m in raw_matches_cv]).reshape(-1, 1, 2)

            M_cv, mask_cv = cv2.findHomography(src_pts_cv_in, dst_pts_cv_in, cv2.RANSAC, 5.0)

            inliers_cv_indices = np.where(mask_cv.ravel() == 1)[0]

    end_time_cv = time.time()
    time_cv = end_time_cv - start_time_cv

    cv_output_path = os.path.join(results_path, 'cv_sift_matches_vis.jpg')
    draw_matches_on_image(img1_color, kps1_cv_np, img2_color, kps2_cv_np,
                          raw_matches_cv, inliers_cv_indices, cv_output_path)

    results = {
        'custom_keypoints': len(kps1_custom_np),
        'open_cv_keypoints': len(kps1_cv_raw),
        'custom_matches': len(inliers_custom_indices),
        'open_cv_matches': len(inliers_cv_indices),
        'custom_time': time_custom,
        'open_cv_time': time_cv,
        'custom_match_img': url_for('module4.static', filename='results/custom_sift_matches_vis.jpg'),
        'open_cv_match_img': url_for('module4.static', filename='results/cv_sift_matches_vis.jpg')
    }
    return results


# --- Blueprint Routes ---

@module4_bp.route('/')
def index():
    """Landing page for Project Delta with links to the two modules."""
    return render_template('module4/index.html')


@module4_bp.route('/stitch', methods=['GET', 'POST'])
def stitch_images():
    """Image stitching module."""
    upload_path = get_upload_path()
    results_path = get_results_path()
    os.makedirs(upload_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)

    if request.method == 'POST':
        camera_files = request.files.getlist('camera_images')
        mobile_file = request.files.get('mobile_panorama')

        uploaded_paths = []

        # 1. Save Camera Images
        for file in camera_files:
            if file and allowed_file(file.filename):
                filename = os.path.join(upload_path, file.filename)
                file.save(filename)
                uploaded_paths.append(filename)

        # 2. Save Mobile Panorama Image
        mobile_path = None
        if mobile_file and allowed_file(mobile_file.filename):
            mobile_path = os.path.join(upload_path, 'mobile_pano.jpg')
            mobile_file.save(mobile_path)

        if len(uploaded_paths) < 2:
            return "Error: Please upload at least 2 camera images.", 400

        # Call custom stitching function
        stitched_image_path = custom_stitch_images(uploaded_paths)

        custom_img_url_path = os.path.basename(stitched_image_path)

        return render_template('module4/stitch_results.html',
                               custom_img_url=url_for('module4.static', filename=f'results/{custom_img_url_path}'),
                               mobile_img_url=url_for('module4.static',
                                                      filename=f'uploads/{os.path.basename(mobile_path)}') if mobile_path else '')

    return render_template('module4/stitch.html')


@module4_bp.route('/sift_compare', methods=['GET', 'POST'])
def sift_compare():
    """SIFT comparison module."""
    upload_path = get_upload_path()
    os.makedirs(upload_path, exist_ok=True)

    if request.method == 'POST':
        img1 = request.files.get('image1')
        img2 = request.files.get('image2')

        path1, path2 = None, None

        if img1 and allowed_file(img1.filename):
            path1 = os.path.join(upload_path, 'sift_img1.jpg')
            img1.save(path1)

        if img2 and allowed_file(img2.filename):
            path2 = os.path.join(upload_path, 'sift_img2.jpg')
            img2.save(path2)

        if not path1 or not path2:
            return "Error: Please upload two images for SIFT comparison.", 400

        results = compare_sift_implementations(path1, path2)

        return render_template('module4/sift_results.html', results=results)

    return render_template('module4/sift_compare.html')


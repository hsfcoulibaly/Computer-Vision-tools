import cv2
import os
import sys
from flask import Blueprint, request, redirect, url_for, render_template_string, send_from_directory
from werkzeug.utils import secure_filename

# Import helper functions from the local module files
try:
    from .AruCo import process_image
    from .filters import process_log_gradient
    from .KeypointDetector import process_keypoint_detection
except ImportError:
    print(
        "!!! ERROR: Could not import helper functions. Make sure 'module3/AruCo.py', 'module3/filters.py', and 'module3/KeypointDetector.py' exist.",
        file=sys.stderr)

# Define the Blueprint.
# The url_prefix is set in app.py when registering this blueprint.
module3_bp = Blueprint(
    'module3',
    __name__,
    template_folder='templates',
    static_folder='static'
)

# Configuration for image uploads
UPLOAD_FOLDER_REL_PATH = 'module3/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


# --- Utility Functions ---

def get_upload_path():
    """Returns the absolute path to the module3 uploads folder."""
    return os.path.join(os.getcwd(), UPLOAD_FOLDER_REL_PATH)


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def find_object_boundaries_simple(image_path):
    """
    Implements object detection logic:
    Thresholding -> Contours -> Bounding Box.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thresholding and Inversion
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.bitwise_not(thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return img

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Draw Green Bounding Box on the original color image
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

    return img


# --- HTML Templates ---

MAIN_MENU_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CV Projects Menu</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .container-shadow {
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
        }
        .menu-button {
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 1rem 2rem;
            font-size: 1.25rem;
            font-weight: bold;
            border-radius: 0.75rem;
            transition: all 0.2s;
            transform: scale(1);
        }
        .menu-button:hover {
            transform: scale(1.02);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        }
    </style>
</head>
<body class="bg-gray-50 min-h-screen p-4 sm:p-8 font-sans flex items-center justify-center">
    <div class="max-w-xl w-full mx-auto text-center">
        <header class="mb-10">
            <h1 class="text-5xl font-extrabold text-gray-800 mb-2">Computer Vision Exercises</h1>
            <p class="text-lg text-indigo-500">Select an exercise to begin.</p>
        </header>

        <div class="space-y-6">
            <a href="{{ url_for('module3.aruco_detector_index') }}" 
               class="menu-button bg-indigo-600 text-white hover:bg-indigo-700 shadow-lg shadow-indigo-300">
               <svg class="w-6 h-6 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 20l-5.447-2.723A1 1 0 013 16.382V5.618a1 1 0 011.553-.894L9 7m0 13l6-3m-6 3V7m6 10l4.447 2.223a1 1 0 001.553-.894V6.382a1 1 0 00-.553-.894L15 4m0 13V4"></path></svg>
               ArUco Boundary Detector
            </a>

            <a href="{{ url_for('module3.keypoint_detector_index') }}" 
               class="menu-button bg-blue-600 text-white hover:bg-blue-700 shadow-lg shadow-blue-300">
               <svg class="w-6 h-6 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.794v6.412a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"></path></svg>
               Edge & Corner Keypoint Detection
            </a>

            <a href="{{ url_for('module3.log_gradient_detector_index') }}" 
               class="menu-button bg-green-600 text-white hover:bg-green-700 shadow-lg shadow-green-300">
               <svg class="w-6 h-6 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4.354v15.292m0 0l-5.836-5.836m5.836 5.836l5.836-5.836M3 10.354a9 9 0 1118 0"></path></svg>
               LoG & Gradient Analysis
            </a>

            <a href="{{ url_for('module3.edge_detector_index') }}" 
               class="menu-button bg-gray-200 text-gray-800 hover:bg-gray-300 shadow-md shadow-gray-300">
               <svg class="w-6 h-6 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 12l3-3 3 3 3-3M3 21h18"></path></svg>
               Edge/Object Detector
            </a>
        </div>

        <footer class="mt-12 text-sm text-gray-400">
            Navigation System
        </footer>
    </div>
</body>
</html>
"""

ARUCO_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ArUco Object Boundary Detector</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .container-shadow {
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
        }
    </style>
</head>
<body class="bg-gray-50 min-h-screen p-4 sm:p-8 font-sans">
    <div class="max-w-4xl mx-auto">
        <header class="text-center mb-8">
             <a href="{{ url_for('module3.main_menu') }}" class="text-indigo-500 hover:underline mb-4 block">&larr; Back to Menu</a>
            <h1 class="text-4xl font-extrabold text-indigo-700">Object Boundary Segmentation (ArUco)</h1>
            <p class="mt-2 text-gray-600">Approximate non-rectangular object boundaries using ArUco markers (Dictionary 6x6_250).</p>
        </header>

        <!-- Upload Form -->
        <div class="bg-white p-6 sm:p-8 rounded-xl container-shadow mb-8">
            <h2 class="text-2xl font-semibold mb-4 text-gray-800">1. Upload Image</h2>
            <p class="text-sm text-gray-500 mb-4">
                Capture an image of your object with markers and upload it here.
            </p>

            <form action="{{ url_for('module3.upload_file_aruco') }}" method="post" enctype="multipart/form-data" class="space-y-4">
                <input type="file" name="file" accept="image/*" required 
                       class="w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100 cursor-pointer">

                <button type="submit" 
                        class="w-full py-3 px-4 bg-indigo-600 text-white font-bold rounded-lg hover:bg-indigo-700 transition duration-150 transform hover:scale-[1.01] shadow-md shadow-indigo-300">
                    Find Boundary
                </button>
            </form>
        </div>

        <!-- Result Display -->
        {% if image_data %}
        <div class="bg-white p-6 sm:p-8 rounded-xl container-shadow">
            <h2 class="text-2xl font-semibold mb-4 text-gray-800">2. Result</h2>
            <div class="text-center mb-4">
                {% if num_markers > 0 %}
                <p class="text-green-600 font-medium">✅ Success! Detected {{ num_markers }} markers and drew the boundary.</p>
                {% else %}
                <p class="text-orange-600 font-medium">⚠️ Warning: No ArUco markers detected. Please ensure markers are clearly visible.</p>
                {% endif %}
            </div>

            <div class="relative overflow-x-auto rounded-lg shadow-xl border-4 border-gray-100">
                <img src="{{ image_data }}" alt="Processed Image with Boundary" class="w-full h-auto object-contain max-w-none">
                <div class="absolute top-0 right-0 m-3 p-2 bg-indigo-500 text-white text-xs font-semibold rounded-lg shadow-lg">
                    Processed Output
                </div>
            </div>
        </div>
        {% elif error %}
        <div class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-xl relative container-shadow" role="alert">
            <strong class="font-bold">Processing Error!</strong>
            <span class="block sm:inline">{{ error }}</span>
        </div>
        {% endif %}

        <footer class="mt-8 text-center text-sm text-gray-400">
            ArUco Detector powered by Python, Flask, and OpenCV.
        </footer>
    </div>
</body>
</html>
"""

EDGE_DETECTOR_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Edge Detection</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .container-shadow {
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
        }
    </style>
</head>
<body class="bg-gray-50 min-h-screen p-4 sm:p-8 font-sans">
    <div class="max-w-4xl mx-auto">
        <header class="text-center mb-8">
            <a href="{{ url_for('module3.main_menu') }}" class="text-gray-500 hover:underline mb-4 block">&larr; Back to Menu</a>
            <h1 class="text-4xl font-extrabold text-gray-700">Edge/Object Detector</h1>
            <p class="mt-2 text-gray-500">Upload, process, and see the bounding box results.</p>
        </header>

        <!-- Upload Form -->
        <div class="bg-white p-6 sm:p-8 rounded-xl container-shadow mb-8">
            <h2 class="text-2xl font-semibold mb-4 text-gray-800">1. Upload Image</h2>

            <form action="{{ url_for('module3.upload_file_edge') }}" method="post" enctype="multipart/form-data" class="space-y-4">
                <input type="file" name="file" accept="image/*" required 
                       class="w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-gray-100 file:text-gray-700 hover:file:bg-gray-200 cursor-pointer">

                <button type="submit" 
                        class="w-full py-3 px-4 bg-gray-600 text-white font-bold rounded-lg hover:bg-gray-700 transition duration-150 transform hover:scale-[1.01] shadow-md shadow-gray-300">
                    Run Detection
                </button>
            </form>
        </div>

        <!-- Results Display -->
        {% if original_file %}
        <div class="bg-white p-6 sm:p-8 rounded-xl container-shadow">
            <h2 class="text-2xl font-semibold mb-4 text-gray-800">2. Results</h2>
            <div class="grid md:grid-cols-2 gap-6">
                <div>
                    <h3 class="font-bold text-center mb-2">Original</h3>
                    <img src="{{ url_for('module3.uploaded_file', filename=original_file) }}" alt="Original Image" class="w-full h-auto rounded-lg border shadow-inner">
                </div>
                <div>
                    <h3 class="font-bold text-center mb-2">Processed (Bounding Box)</h3>
                    <img src="{{ url_for('module3.uploaded_file', filename=processed_file) }}" alt="Processed Image" class="w-full h-auto rounded-lg border shadow-inner">
                </div>
            </div>
        </div>
        {% endif %}

        <footer class="mt-8 text-center text-sm text-gray-400">
            Edge Detector powered by Python, Flask, and OpenCV.
        </footer>
    </div>
</body>
</html>
"""

LOG_GRADIENT_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LoG and Gradient Analysis</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .container-shadow {
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
        }
    </style>
</head>
<body class="bg-gray-50 min-h-screen p-4 sm:p-8 font-sans">
    <div class="max-w-6xl mx-auto">
        <header class="text-center mb-8">
            <a href="{{ url_for('module3.main_menu') }}" class="text-green-600 hover:underline mb-4 block">&larr; Back to Menu</a>
            <h1 class="text-4xl font-extrabold text-green-700">LoG & Gradient Filter Analysis</h1>
            <p class="mt-2 text-gray-600">Analyze image derivatives (Gradient) and second derivatives (Laplacian of Gaussian).</p>
        </header>

        <!-- Upload Form -->
        <div class="bg-white p-6 sm:p-8 rounded-xl container-shadow mb-8">
            <h2 class="text-2xl font-semibold mb-4 text-gray-800">1. Upload Image</h2>
            <p class="text-sm text-gray-500 mb-4">
                Upload any image to calculate its magnitude, angle, and LoG filters.
            </p>

            <form action="{{ url_for('module3.upload_file_log_gradient') }}" method="post" enctype="multipart/form-data" class="space-y-4">
                <input type="file" name="file" accept="image/*" required 
                       class="w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-green-50 file:text-green-700 hover:file:bg-green-100 cursor-pointer">

                <button type="submit" 
                        class="w-full py-3 px-4 bg-green-600 text-white font-bold rounded-lg hover:bg-green-700 transition duration-150 transform hover:scale-[1.01] shadow-md shadow-green-300">
                    Calculate Filters
                </button>
            </form>
        </div>

        <!-- Results Display -->
        {% if results %}
        <div class="bg-white p-6 sm:p-8 rounded-xl container-shadow">
            <h2 class="text-2xl font-semibold mb-4 text-gray-800">2. Results Comparison</h2>

            <div class="grid md:grid-cols-4 gap-4">

                <div class="col-span-4 md:col-span-1 text-center font-semibold p-2 bg-gray-100 rounded-lg">
                    Original
                </div>
                <div class="col-span-4 md:col-span-1 text-center font-semibold p-2 bg-yellow-100 rounded-lg">
                    Gradient Magnitude (First Derivative)
                </div>
                <div class="col-span-4 md:col-span-1 text-center font-semibold p-2 bg-yellow-100 rounded-lg">
                    Gradient Angle (First Derivative)
                </div>
                <div class="col-span-4 md:col-span-1 text-center font-semibold p-2 bg-red-100 rounded-lg">
                    Laplacian of Gaussian (Second Derivative)
                </div>

                <!-- Images Row -->
                <div class="col-span-4 md:col-span-1">
                    <img src="{{ url_for('module3.uploaded_file', filename=results.original_file) }}" alt="Original Image" class="w-full h-auto rounded-lg border shadow-inner">
                </div>
                <div class="col-span-4 md:col-span-1">
                    <img src="{{ url_for('module3.uploaded_file', filename=results.magnitude_file) }}" alt="Gradient Magnitude" class="w-full h-auto rounded-lg border shadow-inner">
                </div>
                <div class="col-span-4 md:col-span-1">
                    <img src="{{ url_for('module3.uploaded_file', filename=results.angle_file) }}" alt="Gradient Angle" class="w-full h-auto rounded-lg border shadow-inner">
                </div>
                <div class="col-span-4 md:col-span-1">
                    <img src="{{ url_for('module3.uploaded_file', filename=results.log_file) }}" alt="Laplacian of Gaussian" class="w-full h-auto rounded-lg border shadow-inner">
                </div>
            </div>

            <h3 class="text-xl font-bold mt-8 mb-4 text-gray-800">Comparison Note:</h3>
            <ul class="list-disc list-inside text-gray-600 space-y-2 bg-gray-50 p-4 rounded-lg">
                <li><strong class="text-yellow-700">Gradient Magnitude</strong> shows strong edges as bright lines, responding to large intensity changes.</li>
                <li><strong class="text-yellow-700">Gradient Angle</strong> shows the direction of the edge (intensity change), visualized here by varying brightness/color.</li>
                <li><strong class="text-red-700">LoG</strong> is often used for edge detection, where zero-crossings in the filter output correspond to the precise edge location. This visualized result shows where intensity changes rapidly.</li>
            </ul>
        </div>
        {% elif error %}
        <div class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-xl relative container-shadow" role="alert">
            <strong class="font-bold">Processing Error!</strong>
            <span class="block sm:inline">{{ error }}</span>
        </div>
        {% endif %}

        <footer class="mt-8 text-center text-sm text-gray-400">
            Filter Analysis powered by Python, Flask, and OpenCV.
        </footer>
    </div>
</body>
</html>
"""

KEYPOINT_DETECTOR_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Edge and Corner Keypoints</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .container-shadow {
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
        }
    </style>
</head>
<body class="bg-gray-50 min-h-screen p-4 sm:p-8 font-sans">
    <div class="max-w-4xl mx-auto">
        <header class="text-center mb-8">
            <a href="{{ url_for('module3.main_menu') }}" class="text-blue-600 hover:underline mb-4 block">&larr; Back to Menu</a>
            <h1 class="text-4xl font-extrabold text-blue-700">Edge & Corner Keypoint Detection</h1>
            <p class="mt-2 text-gray-600">Implementation of Canny (Edge) and Harris (Corner) keypoint detectors.</p>
        </header>

        <!-- Upload Form -->
        <div class="bg-white p-6 sm:p-8 rounded-xl container-shadow mb-8">
            <h2 class="text-2xl font-semibold mb-4 text-gray-800">1. Upload Image</h2>
            <p class="text-sm text-gray-500 mb-4">
                Upload any image to find its primary edge and corner keypoints.
            </p>

            <form action="{{ url_for('module3.upload_file_keypoint') }}" method="post" enctype="multipart/form-data" class="space-y-4">
                <input type="file" name="file" accept="image/*" required 
                       class="w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 cursor-pointer">

                <button type="submit" 
                        class="w-full py-3 px-4 bg-blue-600 text-white font-bold rounded-lg hover:bg-blue-700 transition duration-150 transform hover:scale-[1.01] shadow-md shadow-blue-300">
                    Detect Keypoints
                </button>
            </form>
        </div>

        <!-- Results Display -->
        {% if results %}
        <div class="bg-white p-6 sm:p-8 rounded-xl container-shadow">
            <h2 class="text-2xl font-semibold mb-4 text-gray-800">2. Detected Keypoints</h2>

            <div class="grid md:grid-cols-3 gap-6">

                <div class="text-center font-semibold p-2 bg-gray-100 rounded-lg">
                    Original
                </div>
                <div class="text-center font-semibold p-2 bg-green-100 rounded-lg">
                    Edge Keypoints (Canny)
                </div>
                <div class="text-center font-semibold p-2 bg-red-100 rounded-lg">
                    Corner Keypoints (Harris)
                </div>

                <!-- Images Row -->
                <div>
                    <img src="{{ url_for('module3.uploaded_file', filename=results.original_file) }}" alt="Original Image" class="w-full h-auto rounded-lg border shadow-inner">
                </div>
                <div>
                    <img src="{{ url_for('module3.uploaded_file', filename=results.edge_file) }}" alt="Edge Keypoints" class="w-full h-auto rounded-lg border shadow-inner">
                </div>
                <div>
                    <img src="{{ url_for('module3.uploaded_file', filename=results.corner_file) }}" alt="Corner Keypoints" class="w-full h-auto rounded-lg border shadow-inner">
                </div>
            </div>

            <h3 class="text-xl font-bold mt-8 mb-4 text-gray-800">Detection Summary:</h3>
            <ul class="list-disc list-inside text-gray-600 space-y-2 bg-gray-50 p-4 rounded-lg">
                <li><strong class="text-green-700">Edge Keypoints</strong> are pixels where intensity changes abruptly (first derivative is high). Canny finds these robustly.</li>
                <li><strong class="text-red-700">Corner Keypoints</strong> are points where the gradient changes direction rapidly. Harris detects these points by observing large, simultaneous intensity changes in both the X and Y directions.</li>
            </ul>
        </div>
        {% elif error %}
        <div class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-xl relative container-shadow" role="alert">
            <strong class="font-bold">Processing Error!</strong>
            <span class="block sm:inline">{{ error }}</span>
        </div>
        {% endif %}

        <footer class="mt-8 text-center text-sm text-gray-400">
            Keypoint Detector powered by Python, Flask, and OpenCV.
        </footer>
    </div>
</body>
</html>
"""


# --- Blueprint Routes ---

@module3_bp.route('/')
def main_menu():
    """Renders the central menu page for Project Gamma."""
    return render_template_string(MAIN_MENU_HTML)


# --- 1. Edge Detector Routes ---

@module3_bp.route('/edge-detector')
def edge_detector_index():
    """Renders the Edge Detection upload page."""
    return render_template_string(EDGE_DETECTOR_HTML_TEMPLATE, original_file=None, processed_file=None)


@module3_bp.route('/upload-edge', methods=['POST'])
def upload_file_edge():
    """Handles file upload and processing for Edge Detection."""
    upload_path = get_upload_path()
    os.makedirs(upload_path, exist_ok=True)

    if 'file' not in request.files:
        return redirect(url_for('module3.edge_detector_index'))

    file = request.files['file']

    if file.filename == '' or not allowed_file(file.filename):
        return redirect(url_for('module3.edge_detector_index'))

    if file:
        base_filename = secure_filename(file.filename)
        original_filename = 'original_' + base_filename
        processed_filename = 'processed_' + base_filename

        original_filepath = os.path.join(upload_path, original_filename)
        file.save(original_filepath)

        processed_img = find_object_boundaries_simple(original_filepath)

        processed_filepath = os.path.join(upload_path, processed_filename)
        cv2.imwrite(processed_filepath, processed_img)

        return render_template_string(EDGE_DETECTOR_HTML_TEMPLATE,
                                      original_file=original_filename,
                                      processed_file=processed_filename)


@module3_bp.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serves the image files (both original and processed)."""
    return send_from_directory(get_upload_path(), filename)


# --- 2. ArUco Detector Routes ---

@module3_bp.route('/aruco-detector')
def aruco_detector_index():
    """Renders the ArUco upload page."""
    return render_template_string(ARUCO_HTML_TEMPLATE, image_data=None, error=None, num_markers=0)


@module3_bp.route('/upload-aruco', methods=['POST'])
def upload_file_aruco():
    """Handles image upload and processing for ArUco (in-memory)."""
    if 'file' not in request.files:
        return render_template_string(ARUCO_HTML_TEMPLATE, error="No file part in the request.", image_data=None,
                                      num_markers=0)

    file = request.files['file']

    if file.filename == '':
        return render_template_string(ARUCO_HTML_TEMPLATE, error="No selected file.", image_data=None, num_markers=0)

    if file:
        try:
            image_data = file.read()

            result_data_url, num_markers = process_image(image_data)

            if result_data_url.startswith("Error"):
                return render_template_string(ARUCO_HTML_TEMPLATE, error=result_data_url, image_data=None,
                                              num_markers=0)

            return render_template_string(ARUCO_HTML_TEMPLATE, image_data=result_data_url, error=None,
                                          num_markers=num_markers)

        except Exception as e:
            print(f"Error during ArUco processing: {e}")
            return render_template_string(ARUCO_HTML_TEMPLATE,
                                          error=f"An unexpected error occurred during ArUco processing: {e}",
                                          image_data=None, num_markers=0)

    return render_template_string(ARUCO_HTML_TEMPLATE, error="Unknown file error.", image_data=None, num_markers=0)


# --- 3. LoG & Gradient Detector Routes ---

@module3_bp.route('/log-gradient-detector')
def log_gradient_detector_index():
    """Renders the LoG/Gradient upload page."""
    return render_template_string(LOG_GRADIENT_HTML_TEMPLATE, results=None, error=None)


@module3_bp.route('/upload-log-gradient', methods=['POST'])
def upload_file_log_gradient():
    """Handles file upload and processing for LoG/Gradient analysis."""
    upload_path = get_upload_path()
    os.makedirs(upload_path, exist_ok=True)

    if 'file' not in request.files:
        return redirect(url_for('module3.log_gradient_detector_index'))

    file = request.files['file']

    if file.filename == '' or not allowed_file(file.filename):
        return redirect(url_for('module3.log_gradient_detector_index'))

    if file:
        try:
            base_filename = secure_filename(file.filename)
            original_filepath = os.path.join(upload_path, base_filename)
            file.seek(0)
            file.save(original_filepath)

            results = process_log_gradient(original_filepath, upload_path)

            if "error" in results:
                if os.path.exists(original_filepath):
                    os.remove(original_filepath)
                return render_template_string(LOG_GRADIENT_HTML_TEMPLATE, error=results["error"], results=None)

            return render_template_string(LOG_GRADIENT_HTML_TEMPLATE, results=results, error=None)

        except Exception as e:
            print(f"Error during LoG/Gradient upload: {e}")
            return render_template_string(LOG_GRADIENT_HTML_TEMPLATE, error=f"An unexpected error occurred: {e}",
                                          results=None)


# --- 4. Keypoint Detector Routes ---

@module3_bp.route('/keypoint-detector')
def keypoint_detector_index():
    """Renders the Keypoint detection upload page."""
    return render_template_string(KEYPOINT_DETECTOR_HTML_TEMPLATE, results=None, error=None)


@module3_bp.route('/upload-keypoint', methods=['POST'])
def upload_file_keypoint():
    """Handles file upload and processing for Edge and Corner Keypoint detection."""
    upload_path = get_upload_path()
    os.makedirs(upload_path, exist_ok=True)

    if 'file' not in request.files:
        return redirect(url_for('module3.keypoint_detector_index'))

    file = request.files['file']

    if file.filename == '' or not allowed_file(file.filename):
        return redirect(url_for('module3.keypoint_detector_index'))

    if file:
        try:
            # 1. Save the original file to disk first
            base_filename = secure_filename(file.filename)
            original_filepath = os.path.join(upload_path, base_filename)
            file.seek(0)
            file.save(original_filepath)

            # 2. Process the file
            results = process_keypoint_detection(original_filepath, upload_path)

            if "error" in results:
                if os.path.exists(original_filepath):
                    os.remove(original_filepath)
                return render_template_string(KEYPOINT_DETECTOR_HTML_TEMPLATE, error=results["error"], results=None)

            # Note: We keep the original file saved for display on the results page
            results["original_file"] = os.path.basename(original_filepath)

            # 3. Render the results page
            return render_template_string(KEYPOINT_DETECTOR_HTML_TEMPLATE, results=results, error=None)

        except Exception as e:
            print(f"Error during Keypoint upload: {e}")
            return render_template_string(KEYPOINT_DETECTOR_HTML_TEMPLATE, error=f"An unexpected error occurred: {e}",
                                          results=None)


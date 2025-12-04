from flask import Blueprint, request, jsonify, render_template, send_from_directory, url_for
from werkzeug.utils import secure_filename
import os
import time
import sys
import cv2

# Import utility functions from the local 'utils' directory
try:
    from .utils.template_matcher import find_best_match_and_blur
except ImportError:
    print(
        "!!! ERROR: Could not import 'find_best_match_and_blur'. Make sure 'module2/utils/template_matcher.py' exists.",
        file=sys.stderr)

# Define the Blueprint.
# The url_prefix is set in app.py when registering this blueprint (consistency with module1).
module2_bp = Blueprint(
    'module2',
    __name__,
    template_folder='templates',
    static_folder='static'
)

# Configuration for image uploads (Relative to the main app root)
UPLOAD_FOLDER_REL_PATH = 'uploads'


# --- Utility Functions for Blueprint Setup ---

def get_upload_path():
    """Returns the absolute path to the global 'uploads' folder."""
    return os.path.join(os.getcwd(), UPLOAD_FOLDER_REL_PATH)


def get_template_dir_path():
    """Returns the absolute path to the static template directory for this module."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'templates')


# --- Blueprint Routes ---

@module2_bp.route('/')
def index():
    """Main page for Project Beta (accessible at /project/beta when registered with that prefix)"""
    return render_template('module2/index.html', title="Project Beta: Template Matching & Blurring")


# Template matching endpoint
@module2_bp.route('/template_matching', methods=['POST'])
def template_matching():
    """Handles object detection, blurring the detected region, and returns the modified image."""

    upload_path = get_upload_path()
    os.makedirs(upload_path, exist_ok=True)
    template_dir = get_template_dir_path()

    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'}), 400

    if file:
        filename = secure_filename(f"{time.time()}_{file.filename}")
        filepath = os.path.join(upload_path, filename)
        file.save(filepath)

        try:
            processed_img, message = find_best_match_and_blur(filepath, template_dir)
        except NameError:
            os.remove(filepath)
            return jsonify({'status': 'error',
                            'message': "Processing function not found. Check 'module2/utils/template_matcher.py'."}), 500
        except Exception as e:
            os.remove(filepath)
            return jsonify({'status': 'error', 'message': f"An error occurred during processing: {e}"}), 500

        os.remove(filepath)

        if processed_img is None:
            return jsonify({'status': 'error', 'message': message}), 500

        output_filename = f"processed_{filename}"
        output_filepath = os.path.join(upload_path, output_filename)
        cv2.imwrite(output_filepath, processed_img)

        # Return the URL of the processed image using url_for for proper routing.
        return jsonify({
            'status': 'success',
            'message': message,
            'image_url': url_for('module2.uploaded_file', filename=output_filename)
        })


@module2_bp.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serves files from the global 'uploads' directory."""
    return send_from_directory(get_upload_path(), filename)


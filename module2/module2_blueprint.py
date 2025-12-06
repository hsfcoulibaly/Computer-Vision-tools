from flask import Blueprint, request, jsonify, render_template, send_from_directory, url_for
from werkzeug.utils import secure_filename
import os
import time
import sys
import cv2

# Import utility functions from the local 'utils' directory
try:
    from .utils.template_matcher import find_match_and_blur, find_best_match_and_blur
except ImportError:
    print(
        "!!! ERROR: Could not import template matcher functions. Make sure 'module2/utils/template_matcher.py' exists.",
        file=sys.stderr)

try:
    from .utils.fourier_solver import deconvolution_wiener
except ImportError:
    print(
        "!!! ERROR: Could not import fourier_solver functions. Make sure 'module2/utils/fourier_solver.py' exists.",
        file=sys.stderr)

# Define the Blueprint.
# The url_prefix is set in app.py when registering this blueprint.
module2_bp = Blueprint(
    'module2',
    __name__,
    template_folder='templates',
    static_folder='static'
)

# Configuration for image uploads (Relative to the main app root)
UPLOAD_FOLDER_REL_PATH = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


# --- Utility Functions for Blueprint Setup ---

def get_upload_path():
    """Returns the absolute path to the global 'uploads' folder."""
    return os.path.join(os.getcwd(), UPLOAD_FOLDER_REL_PATH)


def get_template_dir_path():
    """Returns the absolute path to the static template directory for this module."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'templates')


def get_default_scene_path():
    """Returns the absolute path to the default scene image."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'scene', 'default_scene.jpg')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_available_templates():
    """Returns a list of available template objects (obj1-obj10) with their info."""
    template_dir = get_template_dir_path()
    templates = []
    
    for i in range(1, 11):
        for ext in ['.jpg', '.jpeg', '.png']:
            filename = f'obj{i}{ext}'
            filepath = os.path.join(template_dir, filename)
            if os.path.exists(filepath):
                templates.append({
                    'id': i,
                    'name': f'Object {i}',
                    'filename': filename,
                    'url': url_for('module2.serve_template', filename=filename)
                })
                break
        else:
            # No file found for this object, add placeholder
            templates.append({
                'id': i,
                'name': f'Object {i}',
                'filename': None,
                'url': None
            })
    
    return templates


# --- Blueprint Routes ---

@module2_bp.route('/')
def index():
    """Homepage for Module 2 - shows available features"""
    return render_template('module2/index.html')


@module2_bp.route('/object-detection')
def object_detection():
    """Object Detection & Blur page"""
    templates = get_available_templates()
    default_scene_url = url_for('module2.serve_default_scene')
    return render_template('module2/object_detection.html', 
                          title="Object Detection & Blur",
                          templates=templates,
                          default_scene_url=default_scene_url)


@module2_bp.route('/fourier-transform')
def fourier_transform():
    """Fourier Transform Deconvolution page"""
    return render_template('module2/fourier_transform.html')


@module2_bp.route('/scene/default')
def serve_default_scene():
    """Serves the default scene image."""
    scene_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'scene')
    return send_from_directory(scene_dir, 'default_scene.jpg')


@module2_bp.route('/process-fourier', methods=['POST'])
def process_fourier():
    """Process an uploaded image with Fourier Transform deconvolution."""
    upload_path = get_upload_path()
    os.makedirs(upload_path, exist_ok=True)
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Use PNG, JPG, or JPEG.'}), 400
    
    # Get optional parameters
    kernel_size = int(request.form.get('kernel_size', 15))
    sigma = float(request.form.get('sigma', 5))
    wiener_k = float(request.form.get('wiener_k', 0.001))
    
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Save the uploaded file
    filename = secure_filename(f"fourier_{time.time()}_{file.filename}")
    filepath = os.path.join(upload_path, filename)
    file.save(filepath)
    
    try:
        # Process with Wiener deconvolution
        result = deconvolution_wiener(filepath, upload_path, kernel_size, sigma, wiener_k)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        if 'error' in result:
            return jsonify({'error': result['error']}), 500
        
        return jsonify(result)
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f'Processing error: {str(e)}'}), 500


@module2_bp.route('/fourier-results/<filename>')
def serve_fourier_result(filename):
    """Serves Fourier transform result images."""
    return send_from_directory(get_upload_path(), filename)


@module2_bp.route('/match_default_template', methods=['POST'])
def match_default_template():
    """Matches a template against the default scene image."""
    upload_path = get_upload_path()
    os.makedirs(upload_path, exist_ok=True)
    template_dir = get_template_dir_path()
    default_scene = get_default_scene_path()
    
    data = request.json
    template_id = data.get('template_id')

    if not template_id:
        return jsonify({'status': 'error', 'message': 'No template selected'}), 400

    if not os.path.exists(default_scene):
        return jsonify({'status': 'error', 'message': 'Default scene image not found. Please add default_scene.jpg to module2/static/scene/'}), 404

    # Find the template file
    template_path = None
    for ext in ['.jpg', '.jpeg', '.png']:
        filename = f'obj{template_id}{ext}'
        path = os.path.join(template_dir, filename)
        if os.path.exists(path):
            template_path = path
            break

    if not template_path:
        return jsonify({'status': 'error', 'message': f'Template obj{template_id} not found'}), 404

    try:
        processed_img, message, match_found = find_match_and_blur(default_scene, template_path)
    except Exception as e:
        return jsonify({'status': 'error', 'message': f"Processing error: {e}"}), 500

    if processed_img is None:
        return jsonify({'status': 'error', 'message': message}), 500

    # Save processed image
    output_filename = f"result_obj{template_id}_{int(time.time())}.jpg"
    output_filepath = os.path.join(upload_path, output_filename)
    cv2.imwrite(output_filepath, processed_img)

    return jsonify({
        'status': 'success',
        'message': message,
        'match_found': match_found,
        'template_name': f'Object {template_id}',
        'result_url': url_for('module2.uploaded_file', filename=output_filename)
    })


@module2_bp.route('/templates/<filename>')
def serve_template(filename):
    """Serves template images from the static/templates directory."""
    return send_from_directory(get_template_dir_path(), filename)


@module2_bp.route('/upload_scene', methods=['POST'])
def upload_scene():
    """Handles scene image upload and stores it for later processing."""
    upload_path = get_upload_path()
    os.makedirs(upload_path, exist_ok=True)

    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(f"scene_{time.time()}_{file.filename}")
        filepath = os.path.join(upload_path, filename)
        file.save(filepath)

        return jsonify({
            'status': 'success',
            'message': 'Scene image uploaded successfully',
            'scene_filename': filename,
            'scene_url': url_for('module2.uploaded_file', filename=filename)
        })

    return jsonify({'status': 'error', 'message': 'Invalid file type'}), 400


@module2_bp.route('/match_template', methods=['POST'])
def match_template():
    """Matches a specific template against the uploaded scene image."""
    upload_path = get_upload_path()
    template_dir = get_template_dir_path()
    
    data = request.json
    scene_filename = data.get('scene_filename')
    template_id = data.get('template_id')

    if not scene_filename:
        return jsonify({'status': 'error', 'message': 'No scene image provided'}), 400

    if not template_id:
        return jsonify({'status': 'error', 'message': 'No template selected'}), 400

    scene_path = os.path.join(upload_path, scene_filename)
    if not os.path.exists(scene_path):
        return jsonify({'status': 'error', 'message': 'Scene image not found'}), 404

    # Find the template file
    template_path = None
    template_filename = None
    for ext in ['.jpg', '.jpeg', '.png']:
        filename = f'obj{template_id}{ext}'
        path = os.path.join(template_dir, filename)
        if os.path.exists(path):
            template_path = path
            template_filename = filename
            break

    if not template_path:
        return jsonify({'status': 'error', 'message': f'Template obj{template_id} not found'}), 404

    try:
        processed_img, message, match_found = find_match_and_blur(scene_path, template_path)
    except Exception as e:
        return jsonify({'status': 'error', 'message': f"Processing error: {e}"}), 500

    if processed_img is None:
        return jsonify({'status': 'error', 'message': message}), 500

    # Save processed image
    output_filename = f"result_obj{template_id}_{int(time.time())}.jpg"
    output_filepath = os.path.join(upload_path, output_filename)
    cv2.imwrite(output_filepath, processed_img)

    return jsonify({
        'status': 'success',
        'message': message,
        'match_found': match_found,
        'template_name': f'Object {template_id}',
        'result_url': url_for('module2.uploaded_file', filename=output_filename)
    })


# Legacy route for backward compatibility
@module2_bp.route('/template_matching', methods=['POST'])
def template_matching():
    """Legacy: Handles object detection using all templates."""
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
                            'message': "Processing function not found."}), 500
        except Exception as e:
            os.remove(filepath)
            return jsonify({'status': 'error', 'message': f"An error occurred: {e}"}), 500

        os.remove(filepath)

        if processed_img is None:
            return jsonify({'status': 'error', 'message': message}), 500

        output_filename = f"processed_{filename}"
        output_filepath = os.path.join(upload_path, output_filename)
        cv2.imwrite(output_filepath, processed_img)

        return jsonify({
            'status': 'success',
            'message': message,
            'image_url': url_for('module2.uploaded_file', filename=output_filename)
        })


@module2_bp.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serves files from the global 'uploads' directory."""
    return send_from_directory(get_upload_path(), filename)

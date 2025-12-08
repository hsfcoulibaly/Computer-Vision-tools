import os
from flask import Flask, render_template
import sys
# Import Blueprints
from module1.module1_blueprint import module1_bp
from module2.module2_blueprint import module2_bp
from module3.module3_blueprint import module3_bp
from module4.module4_blueprint import module4_bp
from module7.module7_blueprint import module7_bp

# Initialize the Flask application
app = Flask(__name__)

# --- Register Blueprints ---
# Module 1: /module/1/*
app.register_blueprint(module1_bp, url_prefix='/module/1')
# Module 2: /module/2/*
app.register_blueprint(module2_bp, url_prefix='/module/2')
# Module 3: /module/3/*
app.register_blueprint(module3_bp, url_prefix='/module/3')
# Module 4: /module/4/*
app.register_blueprint(module4_bp, url_prefix='/module/4')
# Module 6: /module/6/*
app.register_blueprint(module7_bp, url_prefix='/module/6')

# Define the navigation items for the 8 tabs
NAV_ITEMS = [
    {
        'title': 'Home Hub',
        'route': '/',
        'description': 'Central navigation hub for all computer vision modules.',
        'icon': 'home'
    },
    {
        'title': 'Module 1',
        'route': '/module/1',
        'description': 'Real-World Distance Calculator using camera calibration and pinhole model. Click two points on an image to measure actual distances in centimeters.',
        'icon': 'ruler'
    },
    {
        'title': 'Module 2',
        'route': '/module/2',
        'description': 'Template Matching with Object Blur using multi-scale NCC and ORB features. Also includes Fourier Transform Deconvolution with Wiener filtering.',
        'icon': 'search'
    },
    {
        'title': 'Module 3',
        'route': '/module/3',
        'description': 'Image Processing Suite: ArUco marker detection, Canny edge detection, Harris corner detection, and Laplacian of Gaussian (LoG) gradient analysis.',
        'icon': 'filter'
    },
    {
        'title': 'Module 4',
        'route': '/module/4',
        'description': 'SIFT Feature Detection & Panorama Stitching. Compare custom vs OpenCV SIFT implementations with RANSAC homography estimation.',
        'icon': 'panorama'
    },
    {
        'title': 'Module 5',
        'route': '/module/5',
        'description': 'Coming Soon - This module is under development. Check back later for new computer vision features.',
        'icon': 'construction'
    },
    {
        'title': 'Module 6',
        'route': '/module/6',
        'description': 'Stereo Vision & 3D Triangulation. Calculate real-world 3D distances using stereo image pairs and camera projection matrices.',
        'icon': 'stereo'
    },
    {
        'title': 'Main Dashboard',
        'route': '/dashboard',
        'description': 'Analytics and management dashboard for the hub application.',
        'icon': 'dashboard'
    },
]


# --- Main Routes ---

@app.route('/')
def home():
    """Renders the main homepage with the project overview."""
    return render_template(
        'base.html',
        page_title="Main Project Hub",
        nav_items=NAV_ITEMS,
        current_route='/'
    )


@app.route('/dashboard')
def dashboard():
    """Renders a simple dashboard view for the main hub."""
    return render_template(
        'index.html',
        page_title="Hub Dashboard",
        nav_items=NAV_ITEMS,
        current_route='/dashboard',
        content_title="Central Hub Dashboard",
        content_body="This page is dedicated to the analytics and management of the main hub application itself."
    )


# --- Remaining Module Routes (Placeholders) ---

@app.route('/module/<module_name>')
def sub_module_view(module_name):
    """
    Renders the view for modules that are NOT already handled by Blueprints.
    """

    # We only handle modules here that are not registered as blueprints yet.
    if module_name in ['1', '2', '3', '4', '6']:
        return home()  # Redirect to home if someone tries to access a blueprint via the placeholder route

    # Format the name for display (e.g., '5' -> 'Module 5')
    display_name = f"Module {module_name}"
    current_path = f'/module/{module_name}'

    # Check if the module exists in our defined navigation
    is_valid = any(item['route'] == current_path for item in NAV_ITEMS)

    if not is_valid:
        content_title = "Error: Module Not Found"
        content_body = f"The module '{display_name}' could not be located."
        current_route = current_path
    else:
        content_title = f"{display_name} Homepage (Placeholder)"
        content_body = (
            f"This is the temporary landing page for **{display_name}**.<br><br>"
            "To fully integrate this module, refactor its Python code into a Flask Blueprint "
            "like Module 1 or Module 2, create its own template folder, and register it in the main "
            "<code>app.py</code> file."
        )
        current_route = current_path

    return render_template(
        'base.html',
        page_title=f"Hub | {display_name}",
        nav_items=NAV_ITEMS,
        current_route=current_route,
        content_title=content_title,
        content_body=content_body
    )


# --- Running the App ---
if __name__ == '__main__':
    # Add numpy and cv2 to required imports here if missing from environment
    try:
        import numpy
        import cv2
    except ImportError as e:
        print(f"Required dependency missing: {e}. Please install it (e.g., pip install numpy opencv-python).",
              file=sys.stderr)

    app.run(debug=True)


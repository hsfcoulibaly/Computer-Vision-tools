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
# Project Alpha: /project/alpha/*
app.register_blueprint(module1_bp, url_prefix='/project/alpha')
# Project Beta: /project/beta/*
app.register_blueprint(module2_bp, url_prefix='/project/beta')
# Project Gamma: /project/gamma/*
app.register_blueprint(module3_bp, url_prefix='/project/gamma')
# Project Delta: /project/delta/*
app.register_blueprint(module4_bp, url_prefix='/project/delta')
# Project Zeta: /project/zeta/*
app.register_blueprint(module7_bp, url_prefix='/project/zeta')

# Define the navigation items for the 8 tabs
NAV_ITEMS = [
    {'title': 'Home Hub', 'route': '/'},
    {'title': 'Project Alpha', 'route': '/project/alpha'},
    {'title': 'Project Beta', 'route': '/project/beta'},  # This links to the Blueprint index
    {'title': 'Project Gamma', 'route': '/project/gamma'},
    {'title': 'Project Delta', 'route': '/project/delta'},
    {'title': 'Project Epsilon', 'route': '/project/epsilon'},
    {'title': 'Project Zeta', 'route': '/project/zeta'},
    {'title': 'Main Dashboard', 'route': '/dashboard'},
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


# --- Remaining Sub-Project Routes (Placeholders: Gamma through Zeta) ---

@app.route('/project/<project_name>')
def sub_project_view(project_name):
    """
    Renders the view for sub-projects that are NOT Project Alpha or Beta (already handled by Blueprints).
    """

    # We only handle projects here that are not registered as blueprints yet.
    if project_name in ['alpha', 'beta', 'gamma', 'delta', 'zeta']:
        return home()  # Redirect to home if someone tries to access a blueprint via the placeholder route

    # Capitalize the name for display (e.g., 'gamma' -> 'Gamma')
    display_name = project_name.capitalize()
    current_path = f'/project/{project_name}'

    # Check if the project exists in our defined navigation
    is_valid = any(item['route'] == current_path for item in NAV_ITEMS)

    if not is_valid:
        content_title = "Error: Project Not Found"
        content_body = f"The project '{display_name}' could not be located."
        current_route = current_path
    else:
        content_title = f"{display_name} Project Homepage (Placeholder)"
        content_body = (
            f"This is the temporary landing page for **{display_name}**.<br><br>"
            "To fully integrate this project, refactor its Python code into a Flask Blueprint "
            "like Project Alpha or Beta, create its own template folder, and register it in the main "
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


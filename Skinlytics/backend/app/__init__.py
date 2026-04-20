from flask import Flask, send_from_directory, render_template_string, jsonify
from flask_cors import CORS
import os

# Import the routes blueprint
from .routes import main as main_blueprint

def create_app():
    # Get the absolute path to the frontend directory
    base_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    frontend_dir = os.path.join(base_dir, 'frontend')
    
    # Initialize Flask app with static files configuration
    app = Flask(__name__, 
                static_folder=frontend_dir,
                static_url_path='')
    
    # Configuration
    app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
    
    # Ensure upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Register the blueprint (API routes)
    app.register_blueprint(main_blueprint, url_prefix='/api')
    
    # Enable CORS for API routes
    CORS(app, resources={
        r"/api/*": {
            "origins": ["*"],
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type"],
            "supports_credentials": True
        }
    })
    
    # Serve the main HTML file
    @app.route('/')
    def index():
        return send_from_directory(frontend_dir, 'Skinlytics.html')
    
    # Handle other frontend routes (for SPA routing)
    @app.route('/<path:path>')
    def serve_any(path):
        # If the path exists as a file, serve it
        if os.path.isfile(os.path.join(frontend_dir, path)):
            return send_from_directory(frontend_dir, path)
        # Otherwise, serve index.html and let the frontend router handle it
        return send_from_directory(frontend_dir, 'Skinlytics.html')
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(e):
        return send_from_directory(frontend_dir, 'Skinlytics.html')
    
    @app.errorhandler(500)
    def server_error(e):
        return jsonify({"error": "Internal server error"}), 500
    
    return app

from flask import Flask, send_from_directory
from flask_cors import CORS
import os

# Import the routes blueprint
from .routes import main as main_blueprint

def create_app():
    app = Flask(__name__, static_folder='../../frontend', static_url_path='')
    
    # Configuration
    app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
    
    # Ensure upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Register the blueprint
    app.register_blueprint(main_blueprint)
    
    # Enable CORS with more permissive settings for development
    CORS(app, resources={
        r"/*": {
            "origins": ["*"],  # Allow all origins for development
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
            "supports_credentials": True,
            "expose_headers": ["Content-Disposition"]
        }
    })
    
    # Serve the main HTML file
    @app.route('/')
    def serve_frontend():
        return send_from_directory(app.static_folder, 'Skinlytics.html')
    
    # Serve static files from the frontend directory
    @app.route('/<path:path>')
    def serve_static(path):
        return send_from_directory(app.static_folder, path)
    
    return app

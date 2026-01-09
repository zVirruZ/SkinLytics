import os
from Skinlytics.backend.app import create_app

# Create the Flask application
app = create_app()

# This is needed for Gunicorn to find the app
application = app

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 8000))
    # Run the app in production mode
    app.run(host="0.0.0.0", port=port, debug=False)

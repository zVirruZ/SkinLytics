# wsgi.py (Root of the repository)
# This file serves as the entry point for WSGI servers like Gunicorn

# Import the application instance from the backend module
from Skinlytics.backend.wsgi import app  # This is the Flask app instance

# The WSGI application object is used by the server to load the application
# This is the standard variable name expected by WSGI servers
application = app

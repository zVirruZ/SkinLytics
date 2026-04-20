import os
import tempfile
import pytest
import numpy as np
from PIL import Image

from app import create_app
from ..app.ham10000_model import HAM10000Model

# Test configuration
TEST_UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'test_uploads')
os.makedirs(TEST_UPLOAD_FOLDER, exist_ok=True)

@pytest.fixture
def app(monkeypatch):
    """Create and configure a new app instance for each test."""
    # Set environment variables for test configuration
    monkeypatch.setenv('TESTING', 'true')
    
    # Create the app
    app = create_app()
    
    # Update the app configuration for testing
    app.config.update(
        TESTING=True,
        UPLOAD_FOLDER=TEST_UPLOAD_FOLDER,
        MAX_CONTENT_LENGTH=16 * 1024 * 1024  # 16MB max file size
    )
    
    # Ensure the test upload directory exists
    os.makedirs(TEST_UPLOAD_FOLDER, exist_ok=True)

    # Create the upload folder if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    yield app

@pytest.fixture
def client(app):
    """A test client for the app."""
    return app.test_client()

@pytest.fixture
def runner(app):
    """A test runner for the app's Click commands."""
    return app.test_cli_runner()

@pytest.fixture
def ham10000_model():
    """Fixture for the HAM10000 model."""
    model = HAM10000Model()
    return model

@pytest.fixture
def test_image():
    """Create a test image file."""
    # Create a simple RGB image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :, 0] = 255  # Red channel
    
    # Save to a temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    img_pil = Image.fromarray(img)
    img_pil.save(temp_file.name)
    temp_file.close()
    
    yield temp_file.name
    
    # Cleanup
    if os.path.exists(temp_file.name):
        os.unlink(temp_file.name)

@pytest.fixture
def test_png_image():
    """Create a test PNG image file."""
    # Create a simple RGB image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :, 2] = 255  # Blue channel
    
    # Save to a temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    img_pil = Image.fromarray(img)
    img_pil.save(temp_file.name, format='PNG')
    temp_file.close()
    
    yield temp_file.name
    
    # Cleanup
    if os.path.exists(temp_file.name):
        os.unlink(temp_file.name)

@pytest.fixture
def test_invalid_image():
    """Create an invalid image file."""
    # Create a temporary file with invalid image data
    temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    temp_file.write(b'This is not an image file')
    temp_file.close()
    
    yield temp_file.name
    
    # Cleanup
    if os.path.exists(temp_file.name):
        os.unlink(temp_file.name)

import os
import json
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
from app import create_app
from app.routes import allowed_file, analyze_skin_image

@pytest.fixture
def app():
    """Create and configure a new app instance for each test."""
    # Create a test Flask app
    app = create_app()
    
    # Configure test settings
    app.config['TESTING'] = True
    app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'test_uploads')
    app.config['MODEL_PATH'] = os.path.join(os.path.dirname(__file__), 'test_models')
    
    # Create upload folder if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    return app

@pytest.fixture
def client(app):
    """A test client for the app."""
    return app.test_client()

# Test image fixtures
@pytest.fixture
def test_image(tmp_path):
    """Create a test image file."""
    img_path = tmp_path / "test.jpg"
    # Create a small black image
    img = Image.new('RGB', (100, 100), color='black')
    img.save(img_path)
    return str(img_path)

@pytest.fixture
def test_invalid_image(tmp_path):
    """Create an invalid image file."""
    img_path = tmp_path / "invalid.jpg"
    # Create a file with invalid image data
    img_path.write_text("not an image")
    return str(img_path)

def test_allowed_file():
    """Test the allowed_file function."""
    # Test with allowed extensions
    assert allowed_file('test.jpg') is True
    assert allowed_file('test.jpeg') is True
    assert allowed_file('test.png') is True
    assert allowed_file('test.JPG') is True  # Case insensitive
    assert allowed_file('test.JPEG') is True  # Case insensitive
    assert allowed_file('test.PNG') is True  # Case insensitive
    
    # Test with disallowed extensions
    assert allowed_file('test.txt') is False
    assert allowed_file('test.pdf') is False
    assert allowed_file('test') is False  # No extension
    assert allowed_file('') is False  # Empty filename

def test_analyze_skin_image_success(app, test_image):
    """Test the analyze_skin_image function with a successful prediction."""
    with app.app_context():
        # Mock the model's predict method
        mock_prediction = {
            'success': True,
            'class': 'nv',
            'class_name': 'Melanocytic nevi',
            'confidence': 0.95,
            'is_suspicious': False,
            'binary_confidence': 0.05,
            'top_predictions': [('nv', 0.95), ('mel', 0.03), ('bkl', 0.02)],
            'disclaimer': 'For educational purposes only',
            'advice': 'Regular skin checks are recommended'
        }
        
        with patch('app.routes.get_ham10000_model') as mock_get_model:
            mock_model = MagicMock()
            mock_model.predict.return_value = mock_prediction
            mock_get_model.return_value = mock_model

            # Call the function
            result = analyze_skin_image(test_image)
            
            # Assert the results
            assert result['success'] is True
            assert result['prediction']['class_name'] == 'Melanocytic nevi'
            assert result['has_cancer'] is False
            assert 'advice' in result

def test_analyze_skin_image_file_not_found(app):
    """Test the analyze_skin_image function with a non-existent file."""
    with app.app_context():
        # Call the function with a non-existent file
        result = analyze_skin_image('non_existent_file.jpg')
        
        # Assert the error response
        assert 'error' in result
        assert 'not found' in result['error'].lower() or 'nicht gefunden' in result['error'].lower()
        assert 'advice' in result
        assert 'has_cancer' in result

def test_analyze_skin_image_prediction_failed(app, test_image):
    """Test the analyze_skin_image function when the prediction fails."""
    with app.app_context():
        # Mock the model's predict method to return a failed prediction
        mock_prediction = {
            'success': False,
            'error': 'Prediction failed',
            'has_cancer': None
        }
        
        with patch('app.routes.get_ham10000_model') as mock_get_model:
            mock_model = MagicMock()
            mock_model.predict.return_value = mock_prediction
            mock_get_model.return_value = mock_model

            # Call the function
            result = analyze_skin_image(test_image)
            
            # Assert the results
            assert 'error' in result
            assert result['error'] == 'Prediction failed'
            assert result['has_cancer'] is None
            assert 'advice' in result

def test_analyze_skin_image_exception(app, test_invalid_image):
    """Test the analyze_skin_image function when an exception occurs."""
    with app.app_context():
        # Mock the model to raise an exception
        with patch('app.routes.get_ham10000_model') as mock_get_model:
            mock_model = MagicMock()
            mock_model.predict.side_effect = Exception('Test exception')
            mock_get_model.return_value = mock_model

            # Call the function with an invalid image
            result = analyze_skin_image(test_invalid_image)
            
            # Assert the results
            assert 'error' in result
            assert 'Test exception' in result['error']
            assert 'has_cancer' in result
            assert 'advice' in result
import os
import json
import pytest
from unittest.mock import patch, MagicMock

# Import the functions to be tested
from app.routes import allowed_file, analyze_skin_image

# Import fixtures if they are defined in conftest.py
pytest_plugins = ['pytest-env']

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

@patch('app.routes.get_ham10000_model')
def test_analyze_skin_image_success(mock_get_model, test_image):
    """Test the analyze_skin_image function with a successful prediction."""
    # Mock the model's predict method
    mock_model = MagicMock()
    mock_prediction = {
        'success': True,
        'class': 'mel',
        'class_name': 'Melanom',
        'confidence': 0.95,
        'is_suspicious': True,
        'binary_confidence': 0.92,
        'top_predictions': [
            {'class': 'mel', 'confidence': 0.95, 'display_name': 'Melanom'},
            {'class': 'nv', 'confidence': 0.04, 'display_name': 'Melanozytärer Nävus'},
        ],
        'disclaimer': 'Test disclaimer',
        'advice': 'Bitte konsultieren Sie einen Arzt.'
    }
    mock_model.predict.return_value = mock_prediction
    mock_get_model.return_value = mock_model
    
    # Call the function with a test image
    result = analyze_skin_image(test_image)
    
    # Check the result
    assert result['success'] is True
    assert result['has_cancer'] is True
    assert 'Melanom' in result['advice']
    assert 'prediction' in result
    assert result['prediction']['class_name'] == 'Melanom'
    assert result['prediction']['confidence'] > 0.9

@patch('app.routes.get_ham10000_model')
def test_analyze_skin_image_file_not_found(mock_get_model):
    """Test the analyze_skin_image function with a non-existent file."""
    # Call the function with a non-existent file
    with pytest.raises(FileNotFoundError):
        analyze_skin_image('nonexistent.jpg')

@patch('app.routes.get_ham10000_model')
def test_analyze_skin_image_prediction_failed(mock_get_model, test_image):
    """Test the analyze_skin_image function when the prediction fails."""
    # Mock the model's predict method to return a failure
    mock_model = MagicMock()
    mock_model.predict.return_value = {
        'success': False,
        'error': 'Prediction failed'
    }
    mock_get_model.return_value = mock_model
    
    # Call the function with a test image
    result = analyze_skin_image(test_image)
    
    # Check the result
    assert result['success'] is False
    assert 'error' in result
    assert 'Bei der Analyse ist ein Fehler aufgetreten' in result['advice']

@patch('app.routes.get_ham10000_model')
def test_analyze_skin_image_exception(mock_get_model, test_invalid_image):
    """Test the analyze_skin_image function when an exception occurs."""
    # Mock the model's predict method to raise an exception
    mock_model = MagicMock()
    mock_model.predict.side_effect = Exception('Test exception')
    mock_get_model.return_value = mock_model
    
    # Call the function with an invalid image
    result = analyze_skin_image(test_invalid_image)
    
    # Check the result
    assert 'error' in result
    assert 'Bei der Analyse ist ein Fehler aufgetreten' in result.get('advice', '')

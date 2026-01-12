import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

import app.ham10000_model

def test_ham10000_model_initialization():
    """Test that the HAM10000 model initializes correctly."""
    model = app.ham10000_model.HAM10000Model()
    
    # Check that the model has the expected attributes
    assert hasattr(model, 'model')
    assert hasattr(model, 'classes')
    assert hasattr(model, 'class_names')
    assert hasattr(model, 'cancer_classes')
    assert hasattr(model, 'input_size')
    assert model.input_size == (224, 224)
    
    # Check that the model has the expected classes
    expected_classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    assert model.classes == expected_classes
    
    # Check that cancer classes are correctly identified
    assert 'mel' in model.cancer_classes  # Melanoma should be a cancer class
    assert 'nv' not in model.cancer_classes  # Nevus should not be a cancer class

def test_preprocess_image_with_file_path(ham10000_model, test_image):
    """Test image preprocessing with a file path."""
    # Test with a valid image file
    processed_img = ham10000_model.preprocess_image(test_image)
    
    # Check the output shape and type
    assert isinstance(processed_img, np.ndarray)
    assert processed_img.shape == (1, 224, 224, 3)  # Batch of 1, 224x224 RGB
    assert processed_img.dtype == np.float32

def test_preprocess_image_with_numpy_array(ham10000_model):
    """Test image preprocessing with a numpy array."""
    # Create a test image as a numpy array
    img_array = np.ones((100, 100, 3), dtype=np.uint8) * 255  # White image
    
    # Preprocess the image
    processed_img = ham10000_model.preprocess_image(img_array)
    
    # Check the output shape and type
    assert isinstance(processed_img, np.ndarray)
    assert processed_img.shape == (1, 224, 224, 3)
    assert processed_img.dtype == np.float32

@patch('app.ham10000_model.load_model')
def test_load_model_success(mock_load_model, ham10000_model):
    """Test loading a model successfully."""
    # Mock the model
    mock_model = MagicMock()
    mock_load_model.return_value = mock_model
    
    # Test loading the model
    model_path = "dummy_model.h5"
    ham10000_model.load_model(model_path)
    
    # Check that load_model was called with the correct path
    mock_load_model.assert_called_once_with(model_path)
    
    # Check that the model was set correctly
    assert ham10000_model.model == mock_model

@patch('app.ham10000_model.load_model')
def test_load_model_fallback(mock_load_model, ham10000_model):
    """Test that the model falls back to a simple model when loading fails."""
    # Make load_model raise an exception
    mock_load_model.side_effect = Exception("Failed to load model")
    
    # Test loading the model
    ham10000_model.load_model("nonexistent_model.h5")
    
    # Check that the fallback model was created
    assert ham10000_model.model is not None
    # The fallback model should have the expected input shape
    assert ham10000_model.model.input_shape == (None, 224, 224, 3)

def test_get_class_name(ham10000_model):
    """Test getting class names."""
    # Test with a valid class ID
    assert ham10000_model.get_class_name(0) == 'akiec'
    assert ham10000_model.get_class_name(1) == 'bcc'
    
    # Test with an invalid class ID
    with pytest.raises(IndexError):
        ham10000_model.get_class_name(100)  # Invalid class ID

def test_is_cancer_class(ham10000_model):
    """Test checking if a class is a cancer class."""
    # Test with cancer classes
    assert ham10000_model.is_cancer_class('mel') is True
    assert ham10000_model.is_cancer_class('bcc') is True
    
    # Test with non-cancer classes
    assert ham10000_model.is_cancer_class('nv') is False
    assert ham10000_model.is_cancer_class('bkl') is False
    
    # Test with invalid class
    assert ham10000_model.is_cancer_class('invalid_class') is False

def test_get_class_display_name(ham10000_model):
    """Test getting the display name of a class."""
    # Test with valid class names
    assert ham10000_model.get_class_display_name('mel') == 'Melanom'
    assert ham10000_model.get_class_display_name('nv') == 'Melanozytärer Nävus'
    
    # Test with invalid class name (should return the input)
    assert ham10000_model.get_class_display_name('invalid_class') == 'invalid_class'

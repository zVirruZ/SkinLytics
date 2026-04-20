import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock, call
import tempfile
from PIL import Image
import cv2

from app.ham10000_model import HAM10000Model, get_ham10000_model


@pytest.fixture
def sample_image():
    """Erstellt ein Testbild im Speicher."""
    img = Image.new('RGB', (100, 100), color='red')
    return img


@pytest.fixture
def mock_model():
    """Erstellt eine gemockte Modellinstanz für Tests."""
    with patch('tensorflow.keras.models.load_model') as mock_load:
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        model = HAM10000Model()
        return model, mock_model


@pytest.fixture
def ham10000_model():
    return HAM10000Model()


def test_ham10000_model_initialization():
    """Test that the HAM10000 model initializes correctly."""
    model = HAM10000Model()
    
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


def test_preprocess_image(sample_image, ham10000_model):
    """Testet die Bildvorverarbeitung."""
    # Speichere das Testbild temporär
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        sample_image.save(tmp.name, 'JPEG')
        img_path = tmp.name

    try:
        # Teste die Vorverarbeitung
        processed = ham10000_model.preprocess_image(img_path)

        # Überprüfe die Ausgabeeigenschaften
        assert isinstance(processed, np.ndarray)
        assert processed.shape == (1, 224, 224, 3)  # Erwartete Form nach Vorverarbeitung
        assert processed.dtype == np.float32
        assert -1.0 <= processed.min() <= processed.max() <= 1.0

    finally:
        # Aufräumen
        if os.path.exists(img_path):
            os.remove(img_path)


def test_predict(mock_model):
    """Testet die Vorhersagefunktion."""
    model, mock_model = mock_model

    # Mock die Vorhersage
    mock_predict = mock_model.predict.return_value
    mock_predict.return_value = np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.4, 0.1]])

    # Teste die Vorhersage
    result = model.predict("dummy_path.jpg")

    # Überprüfe die Ausgabe
    assert 'success' in result
    assert 'class' in result
    assert 'confidence' in result
    assert 'is_suspicious' in result
    assert 'top_predictions' in result


def test_get_ham10000_model_singleton():
    """Testet das Singleton-Muster von get_ham10000_model."""
    with patch('app.ham10000_model.HAM10000Model') as mock_ham10000:
        # Erste Instanz
        model1 = get_ham10000_model()
        # Zweite Instanz
        model2 = get_ham10000_model()

        # Beide sollten dieselbe Instanz zurückgeben
        assert model1 is model2
        # Und das Modell sollte nur einmal erstellt worden sein
        mock_ham10000.assert_called_once()


def test_cancer_detection(mock_model):
    """Testet die Krebserkennungslogik."""
    model, mock_model = mock_model

    # Teste mit einem Melanom (sollte als bösartig erkannt werden)
    mock_model.predict.return_value = np.array([[0.0, 0.0, 0.0, 0.0, 0.9, 0.1, 0.0]])  # mel mit 90% Konfidenz
    result = model.predict("dummy_path.jpg")
    assert result['is_suspicious'] is True

    # Teste mit einem Nävus (sollte als gutartig erkannt werden)
    mock_model.predict.return_value = np.array([[0.0, 0.0, 0.0, 0.0, 0.1, 0.9, 0.0]])  # nv mit 90% Konfidenz
    result = model.predict("dummy_path.jpg")
    assert result['is_suspicious'] is False


def test_invalid_image_handling():
    """Testet die Behandlung ungültiger Bilder."""
    model = HAM10000Model()

    # Teste mit nicht existierender Datei
    result = model.predict("nonexistent.jpg")
    assert result['success'] is False
    assert 'error' in result


def test_preprocess_image_with_various_formats(ham10000_model):
    """Test image preprocessing with various image formats and sizes."""
    test_cases = [
        ('red', (100, 100)),    # Small square
        ('blue', (224, 224)),   # Target size
        ('green', (500, 500)),  # Large square
        ('white', (300, 200))   # Rectangle
    ]
    
    for color, size in test_cases:
        # Create test image
        img = Image.new('RGB', size, color=color)
        
        # Save as temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            img_path = tmp.name
            img.save(img_path, 'JPEG')
        
        try:
            # Test preprocessing
            processed = ham10000_model.preprocess_image(img_path)
            
            # Verify output properties
            assert isinstance(processed, np.ndarray)
            assert processed.shape == (1, 224, 224, 3)
            assert processed.dtype == np.float32
            assert -1.0 <= processed.min() <= processed.max() <= 1.0
            
        finally:
            # Cleanup
            if os.path.exists(img_path):
                os.remove(img_path)


def test_predict_with_mock(ham10000_model):
    """Test prediction function with mocked model."""
    # Mock the model's predict method
    mock_prediction = np.array([
        [0.01, 0.01, 0.01, 0.01, 0.01, 0.9, 0.05]  # 90% confidence for 'nv'
    ])
    
    with patch.object(ham10000_model.model, 'predict', return_value=mock_prediction) as mock_predict:
        # Test prediction
        result = ham10000_model.predict("dummy_path.jpg")
        
        # Verify prediction results
        assert result['success'] is True
        assert result['class'] == 'nv'
        assert result['confidence'] == pytest.approx(0.9)
        assert result['is_suspicious'] is False
        assert 'top_predictions' in result
        assert len(result['top_predictions']) > 0
        
        # Verify model was called with correct input
        mock_predict.assert_called_once()
        assert mock_predict.call_args[0][0].shape == (1, 224, 224, 3)


def test_predict_with_cancerous_lesion(ham10000_model):
    """Test prediction with a cancerous lesion (melanoma)."""
    # Mock prediction for melanoma (class 'mel' at index 4)
    mock_prediction = np.zeros((1, 7))
    mock_prediction[0][4] = 0.95  # 95% confidence for melanoma
    
    with patch.object(ham10000_model.model, 'predict', return_value=mock_prediction):
        result = ham10000_model.predict("dummy_path.jpg")
        
        assert result['success'] is True
        assert result['class'] == 'mel'
        assert result['is_suspicious'] is True
        assert result['confidence'] == pytest.approx(0.95)


def test_error_handling(ham10000_model):
    """Test error handling in prediction."""
    # Test with non-existent file
    result = ham10000_model.predict("nonexistent.jpg")
    assert result['success'] is False
    assert 'error' in result
    
    # Test with invalid file
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
        tmp.write(b'not an image')
        tmp_path = tmp.name
    
    try:
        result = ham10000_model.predict(tmp_path)
        assert result['success'] is False
        assert 'error' in result
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_singleton_pattern():
    """Test that get_ham10000_model returns the same instance."""
    with patch('app.ham10000_model.HAM10000Model') as mock_ham10000:
        # First call
        instance1 = get_ham10000_model()
        # Second call
        instance2 = get_ham10000_model()
        
        # Should be the same instance
        assert instance1 is instance2
        # Should only be instantiated once
        mock_ham10000.assert_called_once()
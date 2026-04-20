import os
import io
import json
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
import numpy as np

# Test data
TEST_IMAGE_PATH = os.path.join(os.path.dirname(__file__), 'test_image.jpg')
TEST_IMAGE_PNG_PATH = os.path.join(os.path.dirname(__file__), 'test_image.png')

@pytest.fixture
def client(app):
    """Test client for the application."""
    return app.test_client()

@pytest.fixture
def test_image():
    """Create a test image in memory."""
    img = Image.new('RGB', (100, 100), color='red')
    img_io = io.BytesIO()
    img.save(img_io, 'JPEG')
    img_io.seek(0)
    return img_io

@pytest.fixture
def test_png_image():
    """Create a test PNG image in memory."""
    img = Image.new('RGBA', (200, 200), color=(0, 0, 255, 255))
    img_io = io.BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)
    return img_io

def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get('/api/analyze')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'ok'
    assert 'API is running' in data['message']

def test_upload_no_file(client):
    """Test upload with no file provided."""
    response = client.post('/api/analyze', data={})
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['status'] == 'error'
    assert 'Keine Datei hochgeladen' in data['error']

def test_upload_empty_filename(client):
    """Test upload with empty filename."""
    response = client.post('/api/analyze', data={
        'file': (io.BytesIO(b""), '')
    })
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['status'] == 'error'
    assert 'Keine Datei ausgewählt' in data['error']

def test_upload_invalid_filetype(client):
    """Test upload with invalid file type."""
    response = client.post('/api/analyze', data={
        'file': (io.BytesIO(b"not an image"), 'test.txt')
    })
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['status'] == 'error'
    assert 'Ungültiger Dateityp' in data['error']

def test_upload_valid_jpg(client, test_image):
    """Test successful upload and analysis of a JPG image."""
    # Mock the model prediction
    mock_prediction = {
        'success': True,
        'class': 'nv',
        'class_name': 'Melanozytäre Nävi',
        'confidence': 0.95,
        'is_suspicious': False,
        'binary_confidence': 0.95,
        'top_predictions': [
            {'class': 'nv', 'confidence': 0.95, 'name': 'Melanozytäre Nävi'},
            {'class': 'mel', 'confidence': 0.03, 'name': 'Melanom'}
        ],
        'disclaimer': 'Dies ist keine medizinische Diagnose.',
        'advice': 'Regelmäßige Hautuntersuchungen werden empfohlen.'
    }
    
    with patch('app.routes.get_ham10000_model') as mock_model:
        mock_model.return_value.predict.return_value = mock_prediction
        
        response = client.post('/api/analyze', data={
            'file': (test_image, 'test.jpg')
        }, content_type='multipart/form-data')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'success'
    assert 'result' in data
    assert 'image_url' in data
    assert data['result']['has_cancer'] is False
    assert data['result']['advice'] == 'Regelmäßige Hautuntersuchungen werden empfohlen.'
    assert data['result']['prediction']['class_name'] == 'Melanozytäre Nävi'

def test_upload_valid_png(client, test_png_image):
    """Test successful upload and analysis of a PNG image."""
    # Mock the model prediction
    mock_prediction = {
        'success': True,
        'class': 'mel',
        'class_name': 'Melanom',
        'confidence': 0.87,
        'is_suspicious': True,
        'binary_confidence': 0.87,
        'top_predictions': [
            {'class': 'mel', 'confidence': 0.87, 'name': 'Melanom'},
            {'class': 'bcc', 'confidence': 0.10, 'name': 'Basalzellkarzinom'}
        ],
        'disclaimer': 'Dies ist keine medizinische Diagnose.',
        'advice': 'Bitte konsultieren Sie umgehend einen Hautarzt.'
    }
    
    with patch('app.routes.get_ham10000_model') as mock_model:
        mock_model.return_value.predict.return_value = mock_prediction
        
        response = client.post('/api/analyze', data={
            'file': (test_png_image, 'test.png')
        }, content_type='multipart/form-data')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'success'
    assert data['result']['has_cancer'] is True
    assert 'Bitte konsultieren Sie' in data['result']['advice']
    assert data['result']['prediction']['class_name'] == 'Melanom'

def test_upload_image_processing_error(client, test_image):
    """Test error handling during image processing."""
    with patch('app.routes.get_ham10000_model') as mock_model:
        mock_model.return_value.predict.side_effect = Exception("Image processing error")
        
        response = client.post('/api/analyze', data={
            'file': (test_image, 'test.jpg')
        }, content_type='multipart/form-data')
    
    assert response.status_code == 400  # The route returns 400 on error
    data = json.loads(response.data)
    assert data['status'] == 'error'
    assert 'error' in data

def test_serving_uploaded_file(client, test_image, app):
    """Test serving an uploaded file."""
    # First upload a file
    with patch('app.routes.get_ham10000_model') as mock_model:
        mock_model.return_value.predict.return_value = {
            'success': True,
            'class': 'nv',
            'class_name': 'Melanozytäre Nävi',
            'is_suspicious': False,
            'advice': 'Test advice'
        }
        
        response = client.post('/api/analyze', data={
            'file': (test_image, 'test_serve.jpg')
        }, content_type='multipart/form-data')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'image_url' in data
    
    # Now try to access the uploaded file
    filename = data['image_url'].split('/')[-1]
    response = client.get(f'/api/uploads/{filename}')
    assert response.status_code == 200
    assert response.content_type.startswith('image/')

def test_nonexistent_file(client):
    """Test accessing a non-existent uploaded file."""
    response = client.get('/api/uploads/nonexistent.jpg')
    assert response.status_code == 404

import os
import uuid
from flask import Blueprint, request, jsonify, current_app, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from .ham10000_model import get_ham10000_model

# Create blueprint
main = Blueprint('main', __name__)

# Constants
INPUT_SIZE = (224, 224)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}


def preprocess_image(image_path):
    """Bild für die Analyse vorbereiten."""
    try:
        # Bild laden und in RGB konvertieren
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Konnte das Bild nicht laden")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Auf die richtige Größe bringen und für das Modell vorverarbeiten
        img = cv2.resize(img, INPUT_SIZE)
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)
        
        return img
    except Exception as e:
        current_app.logger.error(f"Fehler bei der Bildvorverarbeitung: {str(e)}")
        raise

def analyze_skin_image(image_path):
    """Analysiert das Hautbild mit dem HAM10000-Modell."""
    try:
        current_app.logger.info(f"Starting analysis of image: {image_path}")
        
        # Verify the file exists and has content
        if not os.path.exists(image_path):
            error_msg = f"Image file not found: {image_path}"
            current_app.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        file_size = os.path.getsize(image_path)
        if file_size == 0:
            error_msg = f"Image file is empty: {image_path}"
            current_app.logger.error(error_msg)
            raise ValueError(error_msg)
            
        current_app.logger.info(f"File size: {file_size} bytes")
        
        # Lazy load model and make prediction
        current_app.logger.info("Loading HAM10000 model and making prediction...")
        prediction = get_ham10000_model().predict(image_path)
        current_app.logger.info(f"Prediction result: {prediction}")
        
        if not prediction.get('success', False):
            error_msg = prediction.get('error', 'Unbekannter Fehler bei der Analyse')
            current_app.logger.error(f"Prediction failed: {error_msg}")
            return {
                'error': error_msg,
                'has_cancer': None,
                'advice': 'Bei der Analyse ist ein Fehler aufgetreten.'
            }
        
        # Extract and validate prediction data
        is_suspicious = bool(prediction.get('is_suspicious', False))
        confidence = float(prediction.get('confidence', 0))
        class_name = str(prediction.get('class_name', 'Unbekannt'))
        binary_confidence = float(prediction.get('binary_confidence', 0))
        top_predictions = prediction.get('top_predictions', [])
        
        current_app.logger.info(f"Analysis complete. Class: {class_name}, Confidence: {confidence:.2f}, Suspicious: {is_suspicious}")
        
        # Create response with prediction data
        return {
            'success': True,
            'prediction': {
                'class': prediction.get('class'),
                'class_name': class_name,
                'confidence': confidence,
                'is_suspicious': is_suspicious,
                'binary_confidence': binary_confidence,
                'top_predictions': top_predictions,
                'disclaimer': prediction.get('disclaimer', '')
            },
            'has_cancer': is_suspicious,
            'advice': prediction.get('advice', '')
        }
        
    except Exception as e:
        error_msg = f"Error in analyze_skin_image: {str(e)}"
        current_app.logger.error(error_msg, exc_info=True)
        return {
            'error': str(e),
            'has_cancer': None,
            'advice': 'Bei der Analyse ist ein Fehler aufgetreten. Bitte versuchen Sie es mit einem anderen Bild.'
        }

def allowed_file(filename):
    """Überprüft, ob die Dateiendung erlaubt ist."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@main.route('/uploads/<filename>')
def uploaded_file(filename):
    """Route to serve uploaded files."""
    try:
        return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        current_app.logger.error(f"Error serving file {filename}: {str(e)}")
        return "File not found", 404

@main.route('/analyze', methods=['GET', 'POST'])
@main.route('/analyze/', methods=['GET', 'POST'])
def analyze_image():
    """API-Endpunkt für die Hautbildanalyse."""
    current_app.logger.info("Received analyze request")
    
    # Handle GET request for testing
    if request.method == 'GET':
        return jsonify({
            'status': 'ok',
            'message': 'API is running. Please use POST to upload an image for analysis.'
        }), 200
    
    # Check if file was uploaded
    if 'file' not in request.files:
        current_app.logger.error("No file part in the request")
        return jsonify({
            'status': 'error',
            'error': 'Keine Datei hochgeladen'
        }), 400
    
    file = request.files['file']
    current_app.logger.info(f"Processing file: {file.filename}")
    
    # Check if file was selected
    if file.filename == '':
        current_app.logger.error("No file selected")
        return jsonify({
            'status': 'error',
            'error': 'Keine Datei ausgewählt'
        }), 400
    
    # Check file type
    if not (file and allowed_file(file.filename)):
        current_app.logger.error(f"Invalid file type: {file.filename}")
        return jsonify({
            'status': 'error',
            'error': 'Ungültiger Dateityp. Bitte laden Sie nur JPG oder PNG Dateien hoch.'
        }), 400
    
    # Generate unique filename and path
    file_ext = os.path.splitext(file.filename)[1].lower()
    filename = f"{uuid.uuid4()}{file_ext}"
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    
    current_app.logger.info(f"Saving file to: {filepath}")
    
    try:
        # Save the uploaded file
        file.save(filepath)
        current_app.logger.info(f"File saved successfully. Size: {os.path.getsize(filepath)} bytes")
        
        # Verify file was saved correctly
        if not os.path.exists(filepath):
            error_msg = "Die Datei konnte nicht gespeichert werden"
            current_app.logger.error(error_msg)
            return jsonify({
                'status': 'error',
                'error': error_msg
            }), 500
        
        # Analyze the image
        current_app.logger.info("Starting image analysis...")
        analysis_result = analyze_skin_image(filepath)
        
        # Check for analysis errors
        if 'error' in analysis_result:
            current_app.logger.error(f"Analysis error: {analysis_result.get('error')}")
            return jsonify({
                'status': 'error',
                'error': analysis_result.get('error')
            }), 400
        
        current_app.logger.info("Analysis completed successfully")
        
        # Return success response with analysis results
        return jsonify({
            'status': 'success',
            'result': analysis_result,
            'image_url': f"/uploads/{filename}"  # This will be served by the uploaded_file route
        })
        
    except Exception as e:
        error_msg = f"Error during analysis: {str(e)}"
        current_app.logger.error(error_msg, exc_info=True)
        
        # Clean up the file if it exists
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                current_app.logger.info("Temporary file removed after error")
            except Exception as e:
                current_app.logger.error(f"Error removing temporary file: {str(e)}")
        
        return jsonify({
            'status': 'error',
            'error': error_msg,
            'success': False,
            'has_cancer': False,
            'advice': 'Bitte versuchen Sie es mit einem anderen Bild oder wenden Sie sich an den Support.'
        }), 500

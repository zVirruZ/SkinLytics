import os
import uuid
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from .ham10000_model import ham10000_model

# Create blueprint
main = Blueprint('main', __name__)

# Global model variable
MODEL = None
INPUT_SIZE = (224, 224)

def load_skin_cancer_model():
    """Lädt das vortrainierte Modell für die Hautkrebserkennung."""
    global MODEL
    if MODEL is None:
        try:
            # Lade ein vortrainiertes Modell (MobileNetV2 als Beispiel)
            # In einer echten Anwendung würden Sie hier Ihr speziell trainiertes Modell laden
            MODEL = MobileNetV2(weights='imagenet')
            current_app.logger.info("Modell erfolgreich geladen")
        except Exception as e:
            current_app.logger.error(f"Fehler beim Laden des Modells: {str(e)}")
            raise
    return MODEL

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
        # Vorhersage mit dem HAM10000-Modell
        prediction = ham10000_model.predict(image_path)
        
        if 'error' in prediction:
            return {
                'error': prediction['error'],
                'has_cancer': None,
                'advice': 'Bei der Analyse ist ein Fehler aufgetreten.'
            }
        
        # Extrahiere die wichtigsten Informationen
        is_cancer = prediction.get('is_cancer', False)
        confidence = prediction.get('confidence', 0)
        class_name = prediction.get('class_name', 'Unbekannt')
        
        # Erstelle die Antwort
        advice = (
            f"Es wurde {class_name} mit einer Wahrscheinlichkeit von {confidence*100:.1f}% erkannt. "
            "Es wird dringend empfohlen, einen Hautarzt aufzusuchen."
            if is_cancer else
            f"Es wurde {class_name} mit einer Wahrscheinlichkeit von {confidence*100:.1f}% erkannt. "
            "Bei Unsicherheiten konsultieren Sie bitte einen Arzt."
        )
        
        return {
            'has_cancer': is_cancer,
            'confidence': confidence,
            'class_name': class_name,
            'predictions': prediction.get('all_predictions', []),
            'advice': advice
        }
        
    except Exception as e:
        current_app.logger.error(f"Fehler bei der Analyse: {str(e)}")
        return {
            'error': str(e),
            'has_cancer': None,
            'advice': 'Bei der Analyse ist ein Fehler aufgetreten. Bitte versuchen Sie es mit einem anderen Bild.'
        }

def allowed_file(filename):
    """Überprüft, ob die Dateiendung erlaubt ist."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

@main.route('/api/analyze', methods=['POST'])
def analyze_image():
    """API-Endpunkt für die Hautbildanalyse."""
    if 'file' not in request.files:
        return jsonify({
            'status': 'error',
            'error': 'Keine Datei hochgeladen'
        }), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({
            'status': 'error',
            'error': 'Keine Datei ausgewählt'
        }), 400
    
    if file and allowed_file(file.filename):
        filename = f"{uuid.uuid4()}{os.path.splitext(file.filename)[1]}"
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(filepath)
            analysis_result = analyze_skin_image(filepath)
            
            # Lösche die temporäre Datei
            if os.path.exists(filepath):
                os.remove(filepath)
            
            # Erstelle die Antwort
            response = {
                'status': 'success',
                'has_cancer': analysis_result.get('has_cancer', False),
                'confidence': analysis_result.get('confidence', 0),
                'predictions': analysis_result.get('predictions', []),
                'advice': analysis_result.get('advice', ''),
                'image_url': f'/uploads/{filename}'
            }
            
            return jsonify(response)
            
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            current_app.logger.error(f"Fehler: {str(e)}")
            return jsonify({
                'status': 'error',
                'error': 'Bei der Analyse ist ein Fehler aufgetreten',
                'details': str(e)
            }), 500
    
    return jsonify({
        'status': 'error',
        'error': 'Ungültiger Dateityp. Bitte laden Sie nur JPG oder PNG Dateien hoch.'
    }), 400

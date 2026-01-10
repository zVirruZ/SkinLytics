import os
import logging
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input

# Logger einrichten
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HAM10000Model:
    def __init__(self, model_path=None):
        """Initialisiert das HAM10000-Modell.
        
        Args:
            model_path (str, optional): Pfad zum trainierten Modell. Wenn None, wird ein neues Modell erstellt.
        """
        self.model = None
        # Erwartete Eingabegröße des Modells (muss vor load_model definiert werden)
        self.input_size = (224, 224)
        
        # Festgelegte Reihenfolge der Klassen (muss mit der Trainingsreihenfolge übereinstimmen)
        self.classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        
        # Klassennamen für die Anzeige
        self.class_names = {
            'akiec': 'Aktinische Keratose',
            'bcc': 'Basalzellkarzinom',
            'bkl': 'Gutartige keratotische Läsion',
            'df': 'Dermatofibrom',
            'mel': 'Melanom',
            'nv': 'Melanozytärer Nävus',
            'vasc': 'Gefäßläsion'
        }
        
        # Klassen, die als potenziell bösartig gelten
        self.cancer_classes = ['akiec', 'bcc', 'mel']
        
        # Bestimme den absoluten Pfad zum Modell
        if model_path is None:
            # Gehe 3 Verzeichnisebenen hoch von der aktuellen Datei, um zum Projektroot zu gelangen
            BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
            model_path = os.path.join(BASE_DIR, 'models', 'ham10000_model_final.h5')
            
        # Lade das Modell
        self.load_model(model_path)
        
        # Disclaimer für die Ausgabe
        self.disclaimer = "Hinweis: Diese Analyse ist kein Ersatz für eine professionelle ärztliche Diagnose. Bitte konsultieren Sie bei gesundheitlichen Bedenken immer einen Arzt."

    def load_model(self, model_path):
        """Lädt das vortrainierte Modell.
        
        Args:
            model_path (str): Pfad zum trainierten Modell.
            
        Raises:
            FileNotFoundError: Wenn die Modell-Datei nicht gefunden wird
            Exception: Bei anderen Fehlern beim Laden des Modells
        """
        try:
            if not os.path.exists(model_path):
                error_msg = f"Modell nicht gefunden: {model_path}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
                
            logger.info(f"Lade Modell von {model_path}")
            self.model = load_model(model_path)
            logger.info("Modell erfolgreich geladen")
            
        except Exception as e:
            error_msg = f"Kritischer Fehler beim Laden des Modells: {str(e)}"
            logger.error(error_msg)
            raise
    
    def preprocess_image(self, image_path, target_size=None):
        """Bereitet das Bild für die Vorhersage mit ResNet50 vor.
        
        Args:
            image_path (str or np.ndarray): Pfad zum Bild oder Bilddaten als Numpy-Array
            target_size (tuple, optional): Zielgröße des Bildes. Standard: Modell-Eingabegröße
            
        Returns:
            numpy.ndarray: Vorverarbeitetes Bild im Format für ResNet50
        """
        try:
            # Wenn image_path ein Numpy-Array ist (z.B. bei Upload)
            if isinstance(image_path, np.ndarray):
                img = image_path.copy()
                if len(img.shape) == 2:  # Graustufenbild
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[2] == 4:  # RGBA zu RGB konvertieren
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                elif img.shape[2] == 1:  # Einzelner Kanal zu RGB
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                # Bild von Datei laden
                img = cv2.imread(image_path)
                if img is None:
                    raise ValueError("Konnte das Bild nicht laden")
                # BGR zu RGB konvertieren
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Größe anpassen
            target_size = target_size or self.input_size
            img = cv2.resize(img, target_size)
            
            # ResNet50 spezifische Vorverarbeitung
            # 1. Konvertiere zu float32
            img = img.astype('float32')
            
            # 2. ResNet50 erwartet spezifische Mittelwerte für die Normalisierung
            # Die preprocess_input-Funktion von ResNet50 macht folgendes:
            # - Zieht den Mittelwert ab: [103.939, 116.779, 123.68]
            # - Konvertiert von RGB zu BGR
            # - Skaliert die Werte nicht (im Gegensatz zu anderen Modellen)
            img = preprocess_input(img)
            
            # 3. Füge eine Batch-Dimension hinzu
            img = np.expand_dims(img, axis=0)
            
            return img
            
        except Exception as e:
            logger.error(f"Fehler bei der Bildvorverarbeitung: {str(e)}")
            raise

    def _load_and_preprocess_image(self, image_path):
        """Lädt und verarbeitet ein Bild für die Vorhersage vor.
        
        Args:
            image_path (str or numpy.ndarray): Pfad zum Bild oder Bilddaten
            
        Returns:
            numpy.ndarray: Vorverarbeitetes Bild
        """
        try:
            # Bild laden
            if isinstance(image_path, str):
                img = cv2.imread(image_path)
                if img is None:
                    raise ValueError("Konnte das Bild nicht laden")
            else:
                img = image_path.copy()
            
            # BGR zu RGB konvertieren
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Größe anpassen
            img = cv2.resize(img, self.input_size)
            
            # ResNet50 spezifische Vorverarbeitung
            img = preprocess_input(img)
            
            # Füge eine Batch-Dimension hinzu
            img = np.expand_dims(img, axis=0)
            
            return img
            
        except Exception as e:
            logger.error(f"Fehler bei der Bildvorverarbeitung: {str(e)}")
            raise

    def predict(self, image_path, threshold=0.12):  # Reduzierter Schwellenwert für höhere Sensitivität
        """Führt eine Vorhersage für das gegebene Bild durch.
        
        Args:
            image_path (str or numpy.ndarray): Pfad zum Bild oder Bilddaten
            threshold (float): Schwellenwert für die Krebserkennung (0-1), standardmäßig 0.12 für maximale Sensitivität
            
        Returns:
            dict: Vorhersageergebnisse mit Klassifizierung und Metriken
        """
        try:
            # Bild laden und vorverarbeiten
            img = self._load_and_preprocess_image(image_path)
            
            # Vorhersage machen
            predictions = self.model.predict(img, verbose=0)[0]  # [0] weil wir nur ein Bild haben
            
            # Erstelle eine Liste von Tupeln (Klassenname, Konfidenz)
            class_confidences = list(zip(self.classes, predictions))
            
            # Sortiere nach Konfidenz (absteigend)
            class_confidences.sort(key=lambda x: x[1], reverse=True)
            
            # Extrahiere alle 7 Klassen mit erhöhter Gewichtung für Krebsklassen
            top_predictions = []
            for class_name, confidence in class_confidences:
                is_cancer = class_name in self.cancer_classes
                # Erhöhe die Konfidenz für Krebsklassen stärker (höherer Multiplikator)
                adjusted_confidence = confidence * (1.6 if is_cancer else 0.8)  # Stärkere Gewichtung
                top_predictions.append({
                    'class': class_name,
                    'name': self.class_names.get(class_name, class_name),
                    'confidence': float(confidence),
                    'adjusted_confidence': float(adjusted_confidence),
                    'is_suspicious': is_cancer
                })
            
            # Sortiere nach der angepassten Konfidenz (höchste zuerst)
            top_predictions.sort(key=lambda x: x['adjusted_confidence'], reverse=True)
            
            # Beste Vorhersage basierend auf der angepassten Konfidenz
            best_pred = max(top_predictions, key=lambda x: x['adjusted_confidence'])
            
            # Binäre Klassifikation (Verdächtig vs. Unauffällig)
            # Stärkere Gewichtung der Krebsklassen (höherer Multiplikator)
            cancer_confidence = sum(
                conf * (2.0 if cls in self.cancer_classes else 0.5)  # Höhere Gewichtung für Krebsklassen
                for cls, conf in class_confidences
            )
            
            # Anwenden des Schwellenwerts (niedriger für höhere Sensitivität)
            is_suspicious = cancer_confidence > threshold
            
            # Spezielle Behandlung für Krebsklassen:
            # 1. Wenn eine Krebsklasse die höchste Konfidenz hat, aber knapp unter dem Schwellenwert liegt
            # 2. Wenn die Konfidenz für eine Krebsklasse über 10% liegt (sehr niedrige Schwelle)
            # 3. Wenn die Summe der Krebsklassen über 15% liegt
            cancer_sum = sum(conf for cls, conf in class_confidences if cls in self.cancer_classes)
            
            if (best_pred['is_suspicious'] and 
                (best_pred['confidence'] > threshold * 0.6 or  # 60% des Schwellenwerts
                 cancer_sum > 0.15)):  # Summe der Krebsklassen über 15%
                is_suspicious = True
                cancer_confidence = max(cancer_confidence, threshold * 1.2)  # Höhere Mindestwahrscheinlichkeit
                
            # Zusätzliche Metriken für die Entscheidungsfindung
            if not is_suspicious and cancer_sum > 0.2:  # Wenn die Summe der Krebsklassen über 20% liegt
                is_suspicious = True
                cancer_confidence = max(cancer_confidence, 0.4)  # Setze Mindestkonfidenz
            
            # Erstelle die Antwort
            response = {
                'class': best_pred['class'],
                'class_name': best_pred['name'],
                'confidence': best_pred['confidence'],
                'is_suspicious': bool(is_suspicious),  # Convert NumPy boolean to Python boolean
                'binary_confidence': float(min(cancer_confidence, 1.0)),
                'top_predictions': top_predictions,
                'success': True,
                'disclaimer': self.disclaimer
            }
            
            # Füge zusätzliche Metriken hinzu
            try:
                self._add_image_metrics(image_path, response)
            except Exception as e:
                logger.warning(f"Konnte keine zusätzlichen Metriken berechnen: {str(e)}")
            
            return response
            
        except Exception as e:
            error_msg = f"Fehler bei der Vorhersage: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                'error': error_msg,
                'class': 'error',
                'class_name': 'Fehler',
                'is_cancer': False,
                'success': False
            }
    
    def _add_image_metrics(self, image_path, response):
        """Fügt zusätzliche Bildmetriken zur Antwort hinzu."""
        try:
            if isinstance(image_path, str):
                img = cv2.imread(image_path)
                if img is None:
                    return
            else:
                img = image_path.copy()
            
            # Konvertiere zu Graustufen für die Kantenerkennung
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            
            # Kantenerkennung
            edges = cv2.Canny(gray, 100, 200)
            edge_density = float(np.mean(edges) / 255.0)
            
            # Farbvarianz
            if len(img.shape) == 3:
                color_std = np.std(img, axis=(0, 1))
                color_variation = float(np.mean(color_std))
            else:
                color_variation = 0.0
            
            # Füge Metriken zur Antwort hinzu
            metrics = {
                'edge_density': edge_density,
                'color_variation': color_variation,
                'asymmetry': float(self._calculate_asymmetry(gray)) if gray is not None else 0.0,
                'color_count': int(self._count_colors(img)) if len(img.shape) == 3 else 1
            }
            
            response['metrics'] = metrics
            
        except Exception as e:
            logger.warning(f"Konnte keine zusätzlichen Metriken berechnen: {str(e)}")
    
    def _calculate_asymmetry(self, gray_img):
        """Berechnet die Asymmetrie des Hautmals."""
        try:
            # Binarisiere das Bild
            _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Finde Konturen
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return 0
                
            # Nimm die größte Kontur
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Berechne das umschließende Rechteck
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = np.intp(box)  # Geändert von np.int0 zu np.intp
            
            # Berechne die Asymmetrie
            width = rect[1][0]
            height = rect[1][1]
            area = cv2.contourArea(largest_contour)
            rect_area = width * height
            
            if rect_area > 0:
                return float(area / rect_area)
            return 0
            
        except Exception as e:
            logger.warning(f"Fehler bei der Asymmetrieberechnung: {str(e)}")
            return 0
    
    def _count_colors(self, img, n_colors=5):
        """Zählt die Anzahl der dominierenden Farben im Bild."""
        try:
            # Reduziere die Farbtiefe
            pixels = img.reshape((-1, 3)).astype(np.float32)
            
            # Führe k-means Clustering durch, um die dominierenden Farben zu finden
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
            _, labels, centers = cv2.kmeans(
                pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Zähle die Anzahl der eindeutigen Labels (Farben)
            return len(np.unique(labels))
            
        except Exception as e:
            logger.warning(f"Fehler bei der Farbzählung: {str(e)}")
            return 1

# Globale Variable für das Modell
_ham10000_model = None

def get_ham10000_model():
    """Gibt die globale Modellinstanz zurück und lädt sie bei Bedarf.
    
    Returns:
        HAM10000Model: Die geladene Modellinstanz
    """
    global _ham10000_model
    if _ham10000_model is None:
        logger.info("Lazy Loading des HAM10000-Modells...")
        _ham10000_model = HAM10000Model()
        logger.info("HAM10000-Modell erfolgreich geladen")
    return _ham10000_model

import os
import tensorflow as tf
from tensorflow.keras.models import load_model, save_model
import numpy as np
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

class HAM10000Model:
    def __init__(self, model_path=None):
        self.model = None
        self.classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        self.class_names = {
            'akiec': 'Aktinische Keratose',
            'bcc': 'Basalzellkarzinom',
            'bkl': 'Gutartige keratotische Läsion',
            'df': 'Dermatofibrom',
            'mel': 'Melanom',
            'nv': 'Melanozytärer Nävus',
            'vasc': 'Gefäßläsion'
        }
        self.load_model(model_path)

    def load_model(self, model_path=None):
        """Lädt das vortrainierte HAM10000 Modell"""
        if model_path and os.path.exists(model_path):
            self.model = load_model(model_path)
        else:
            # Fallback zu einem einfachen Modell, falls kein Modell gefunden wird
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=(224, 224, 3),
                include_top=False,
                weights='imagenet'
            )
            base_model.trainable = False
            self.model = tf.keras.Sequential([
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(len(self.classes), activation='softmax')
            ])
            self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

    def preprocess_image(self, image_path, target_size=(224, 224)):
        """Bild für die Vorhersage vorbereiten"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Konnte das Bild nicht laden")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)
        return img

    def predict(self, image_path):
        """Führt eine Vorhersage für das gegebene Bild durch"""
        try:
            # Bild vorverarbeiten
            img_array = self.preprocess_image(image_path)
            
            # Vorhersage machen
            predictions = self.model.predict(img_array)
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = self.classes[predicted_class_idx]
            confidence = float(predictions[0][predicted_class_idx])
            
            # Zusätzliche Metriken
            is_cancer = predicted_class in ['mel', 'bcc', 'akiec']  # Krebsarten im Datensatz
            confidence_threshold = 0.6 if is_cancer else 0.4
            
            return {
                'class': predicted_class,
                'class_name': self.class_names.get(predicted_class, 'Unbekannt'),
                'confidence': confidence,
                'is_cancer': is_cancer and confidence > confidence_threshold,
                'all_predictions': [
                    {
                        'class': self.classes[i],
                        'class_name': self.class_names.get(self.classes[i], 'Unbekannt'),
                        'confidence': float(conf)
                    }
                    for i, conf in enumerate(predictions[0])
                ]
            }
            
        except Exception as e:
            print(f"Fehler bei der Vorhersage: {str(e)}")
            return {
                'error': str(e),
                'class': 'error',
                'class_name': 'Fehler',
                'is_cancer': False
            }

# Globale Instanz des Modells
ham10000_model = HAM10000Model()

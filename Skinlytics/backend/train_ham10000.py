import os
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Verzeichnisse erstellen
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Define class order (must match the order in ham10000_model.py)
CLASS_ORDER = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# Class names mapping
CLASS_NAMES = {
    'akiec': 'Aktinische Keratose',
    'bcc': 'Basalzellkarzinom',
    'bkl': 'Gutartige keratotische Läsion',
    'df': 'Dermatofibrom',
    'mel': 'Melanom',
    'nv': 'Melanozytärer Nävus',
    'vasc': 'Gefäßläsion'
}

# Cancer classes (suspicious)
CANCER_CLASSES = ['akiec', 'bcc', 'mel']

def load_and_prepare_data(data_dir='data'):
    """Lädt und bereitet die Daten vor."""
    # Pfade zu den Daten
    metadata_path = os.path.join(data_dir, 'HAM10000_metadata.csv')
    images_dir = os.path.join(data_dir, 'HAM10000_images')
    
    # Metadaten laden
    metadata = pd.read_csv(metadata_path)
    
    # Vollständige Pfade zu den Bildern
    metadata['image_path'] = metadata['image_id'].apply(lambda x: os.path.join(images_dir, f"{x}.jpg"))
    
    # Ensure all classes are in our defined order and remove any others
    metadata = metadata[metadata['dx'].isin(CLASS_ORDER)].copy()
    
    # Convert class labels to categorical with fixed order
    metadata['dx'] = pd.Categorical(metadata['dx'], categories=CLASS_ORDER, ordered=True)
    
    # Sort by class to ensure consistent ordering
    metadata = metadata.sort_values('dx')
    
    # Split into training and validation sets
    train_df, val_df = train_test_split(
        metadata, 
        test_size=0.2, 
        stratify=metadata['dx'],
        random_state=42
    )
    
    return train_df, val_df

def create_data_generators(train_df, val_df, batch_size=32):
    """Erstellt die Daten-Generatoren für Training und Validierung."""
    # Daten-Augmentierung für das Training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Nur Rescaling für die Validierung
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Training data generator with fixed class order
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='image_path',
        y_col='dx',
        target_size=(224, 224),
        class_mode='categorical',
        batch_size=batch_size,
        classes=CLASS_ORDER,
        shuffle=True
    )
    
    # Validation data generator with fixed class order
    validation_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='image_path',
        y_col='dx',
        target_size=(224, 224),
        class_mode='categorical',
        batch_size=batch_size,
        classes=CLASS_ORDER,
        shuffle=False
    )
    
    # Verify class indices
    print("Class indices:", train_generator.class_indices)
    assert list(train_generator.class_indices.keys()) == CLASS_ORDER, \
        f"Class order mismatch. Expected {CLASS_ORDER}, got {list(train_generator.class_indices.keys())}"
    
    return train_generator, validation_generator

def build_model():
    """Baut das Modell auf."""
    # Basis-Modell (ResNet50)
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(len(CLASS_ORDER), activation='softmax')(x)
    
    # Build the model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model():
    """Hauptfunktion zum Trainieren des Modells."""
    # Daten laden
    print("Lade Daten...")
    train_df, val_df = load_and_prepare_data()
    
    # Generatoren erstellen
    print("Erstelle Generatoren...")
    train_generator, validation_generator = create_data_generators(train_df, val_df)
    
    # Modell erstellen
    print("Erstelle Modell...")
    model = build_model()
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        'models/ham10000_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6
    )
    
    # Training
    print("Starte Training...")
    print(f"Anzahl der Trainingsbeispiele: {len(train_generator) * train_generator.batch_size}")
    print(f"Anzahl der Validierungsbeispiele: {len(validation_generator) * validation_generator.batch_size}")
    
    history = model.fit(
        train_generator,
        epochs=50,
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stopping, reduce_lr],
        workers=4,
        use_multiprocessing=True
    )
    
    # Save the final model with class information
    final_model_path = 'models/ham10000_model_final.h5'
    model.save(final_model_path)
    print(f"Modell wurde gespeichert als '{final_model_path}'")
    
    # Verify the model can be loaded
    try:
        loaded_model = tf.keras.models.load_model(final_model_path)
        print("Modell erfolgreich geladen. Überprüfe die Ausgabedimensionen...")
        print(f"Erwartete Ausgabedimension: {len(CLASS_ORDER)} Klassen")
        print(f"Tatsächliche Ausgabedimension: {loaded_model.output_shape[1]}")
        assert loaded_model.output_shape[1] == len(CLASS_ORDER), \
            f"Falsche Anzahl von Ausgabeklassen. Erwartet: {len(CLASS_ORDER)}, Gefunden: {loaded_model.output_shape[1]}"
    except Exception as e:
        print(f"Fehler beim Laden des Modells: {str(e)}")
        raise
    
    print("Training abgeschlossen. Modell wurde erfolgreich gespeichert und validiert.")
    return model, history, None  # Return None for history_fine for compatibility

def plot_training_history(history, history_fine=None, history_five=None):
    """Zeichnet den Trainingsverlauf."""
    # Genauigkeit
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validierung')
    if history_fine:
        plt.plot(history_fine.history['accuracy'], label='Feintuning Training')
        plt.plot(history_five.history['val_accuracy'], label='Feintuning Validierung')
    plt.title('Modellgenauigkeit')
    plt.ylabel('Genauigkeit')
    plt.xlabel('Epoche')
    plt.legend()
    
    # Verlust
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validierung')
    if history_fine:
        plt.plot(history_fine.history['loss'], label='Feintuning Training')
        plt.plot(history_fine.history['val_loss'], label='Feintuning Validierung')
    plt.title('Modellverlust')
    plt.ylabel('Verlust')
    plt.xlabel('Epoche')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

if __name__ == "__main__":
    # Training starten
    model, history, history_fine = train_model()
    
    # Trainingsverlauf anzeigen
    plot_training_history(history, history_fine)
    
    print("\nTraining abgeschlossen! Das Modell wurde unter 'models/ham10000_model_final.h5' gespeichert.")
    print("Um das Modell in Ihrer Anwendung zu verwenden, aktualisieren Sie den Modellpfad in 'ham10000_model.py'.")

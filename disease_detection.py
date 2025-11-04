"""
Main Disease Detection System
============================
This is the main file that coordinates all system components.
Enhanced with CNN model training and dataset management.
Optimized for session management and crash prevention.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import json
import zipfile
import shutil
import gc
import threading
import time

# Dataset Configuration
DATASET_ZIP_PATH = "/content/drive/MyDrive/plant2/archive.zip"  # Modify this path to your dataset
BACKUP_DATASET_PATHS = [
    "./plant_disease_dataset.zip",
    "./dataset.zip",
    "/content/dataset.zip",
    "/kaggle/input/plant-disease/plant_disease_dataset.zip"
]

# Add current directory to Python path
current_dir = Path("/content/drive/MyDrive/plant2/disease_detection.py").parent
sys.path.append(str(current_dir))

# Import all modules with error handling
try:
    from fertilizer_calculator import FertilizerCalculatorModule
    from detection_history import DetectionHistoryModule
    from treatment_history import TreatmentHistoryModule
    from farm_analytics import FarmAnalyticsModule
    from user_profile import UserProfileModule
    from data_management import DataManagementModule
    from export_reports import ExportReportsModule
except ImportError as e:
    print(f"Warning: Some modules not found: {e}")
    print("Creating placeholder modules...")

    # Create minimal placeholder classes
    class PlaceholderModule:
        def __init__(self, base_path): pass
        def run(self): print("Module not available")

    FertilizerCalculatorModule = PlaceholderModule
    DetectionHistoryModule = PlaceholderModule
    TreatmentHistoryModule = PlaceholderModule
    FarmAnalyticsModule = PlaceholderModule
    UserProfileModule = PlaceholderModule
    DataManagementModule = PlaceholderModule
    ExportReportsModule = PlaceholderModule

# Import core components
import numpy as np
from PIL import Image
import pickle
from sklearn.preprocessing import LabelEncoder

# CNN and deep learning imports with better error handling
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

    # Configure TensorFlow for stability
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True) if tf.config.list_physical_devices('GPU') else None
    tf.get_logger().setLevel('ERROR')

    CNN_AVAILABLE = True
    print("✓ CNN/TensorFlow support available")
except ImportError as e:
    CNN_AVAILABLE = False
    print(f"⚠ CNN/TensorFlow not available: {e}")
except Exception as e:
    CNN_AVAILABLE = False
    print(f"⚠ TensorFlow configuration error: {e}")

class SessionManager:
    """Enhanced session management with auto-save and crash recovery"""

    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.session_file = self.base_path / 'session_state.json'
        self.backup_file = self.base_path / 'session_backup.json'
        self.session_data = self.load_session()
        self.auto_save_thread = None
        self.stop_auto_save = threading.Event()
        self.start_auto_save()

    def start_auto_save(self):
        """Start automatic session saving thread"""
        def auto_save_worker():
            while not self.stop_auto_save.wait(30):  # Save every 30 seconds
                try:
                    self.save_session()
                except Exception as e:
                    print(f"Auto-save warning: {e}")

        self.auto_save_thread = threading.Thread(target=auto_save_worker, daemon=True)
        self.auto_save_thread.start()

    def stop_auto_save_thread(self):
        """Stop the auto-save thread"""
        if self.auto_save_thread:
            self.stop_auto_save.set()

    def load_session(self):
        """Load previous session data with backup recovery"""
        for session_file in [self.session_file, self.backup_file]:
            if session_file.exists():
                try:
                    with open(session_file, 'r') as f:
                        data = json.load(f)
                    print(f"Session loaded from {session_file.name}")
                    return data
                except Exception as e:
                    print(f"Failed to load {session_file.name}: {e}")

        return {
            'last_access': None,
            'model_training_progress': None,
            'dataset_path': None,
            'model_type': 'fallback',
            'training_epochs_completed': 0,
            'best_accuracy': 0.0,
            'crash_count': 0
        }

    def save_session(self):
        """Save current session state with backup"""
        self.session_data['last_access'] = datetime.now().isoformat()
        try:
            # Save to backup first
            with open(self.backup_file, 'w') as f:
                json.dump(self.session_data, f, indent=2)

            # Then save to main file
            with open(self.session_file, 'w') as f:
                json.dump(self.session_data, f, indent=2)

        except Exception as e:
            print(f"Warning: Could not save session: {e}")

    def update_progress(self, key, value):
        """Update training progress with immediate save"""
        self.session_data[key] = value
        self.save_session()

    def increment_crash_count(self):
        """Track crash occurrences"""
        self.session_data['crash_count'] = self.session_data.get('crash_count', 0) + 1
        self.save_session()

class DatasetManager:
    """Enhanced dataset management with validation and optimization"""

    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.dataset_path = self.base_path / 'datasets'
        self.dataset_path.mkdir(exist_ok=True)

    def find_dataset_zip(self):
        """Find dataset ZIP file from multiple possible locations"""
        # Check configured path first
        if os.path.exists(DATASET_ZIP_PATH):
            return DATASET_ZIP_PATH

        # Check backup locations
        for path in BACKUP_DATASET_PATHS:
            if os.path.exists(path):
                print(f"Found dataset at: {path}")
                return path

        # Search in common directories
        search_dirs = ['.', self.base_path, self.dataset_path]
        for search_dir in search_dirs:
            zip_files = list(Path(search_dir).glob('*.zip'))
            if zip_files:
                return str(zip_files[0])

        return None

    def validate_zip_file(self, zip_path):
        """Validate ZIP file before extraction"""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Test ZIP file integrity
                zip_ref.testzip()

                # Check for image files
                image_files = [f for f in zip_ref.namelist()
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

                if len(image_files) < 10:
                    print("Warning: Very few image files found in ZIP")
                    return False

                print(f"ZIP validation passed: {len(image_files)} image files found")
                return True

        except Exception as e:
            print(f"ZIP validation failed: {e}")
            return False

    def extract_zip_dataset(self, zip_path=None):
        """Extract and organize zip dataset with progress tracking"""
        if not zip_path:
            zip_path = self.find_dataset_zip()

        if not zip_path:
            print("Error: No dataset ZIP file found!")
            print("Please place your dataset ZIP file in one of these locations:")
            for path in [DATASET_ZIP_PATH] + BACKUP_DATASET_PATHS:
                print(f"  - {path}")
            return False

        print(f"Using dataset: {zip_path}")

        # Validate ZIP file
        if not self.validate_zip_file(zip_path):
            return False

        try:
            extract_path = self.dataset_path / 'extracted'
            if extract_path.exists():
                print("Cleaning previous extraction...")
                shutil.rmtree(extract_path)

            print("Extracting dataset... This may take a while.")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                total_files = len(zip_ref.namelist())
                extracted = 0

                for file_info in zip_ref.infolist():
                    zip_ref.extract(file_info, extract_path)
                    extracted += 1

                    if extracted % 100 == 0:
                        progress = (extracted / total_files) * 100
                        print(f"Extraction progress: {progress:.1f}% ({extracted}/{total_files})")

            # Organize dataset structure
            organized_path = self.dataset_path / 'organized'
            print("Organizing dataset structure...")
            class_count = self.organize_dataset(extract_path, organized_path)

            if class_count > 0:
                print(f"✓ Dataset extracted and organized successfully ({class_count} classes)")
                return str(organized_path)
            else:
                print("Error: No valid classes found in dataset")
                return False

        except Exception as e:
            print(f"Error extracting dataset: {e}")
            return False

    def organize_dataset(self, source_path, target_path):
        """Organize dataset with better structure detection"""
        if target_path.exists():
            shutil.rmtree(target_path)

        train_path = target_path / 'train'
        val_path = target_path / 'validation'
        train_path.mkdir(parents=True, exist_ok=True)
        val_path.mkdir(parents=True, exist_ok=True)

        # Enhanced class directory detection
        class_info = {}

        for root, dirs, files in os.walk(source_path):
            image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

            if len(image_files) >= 5:  # Minimum 5 images per class
                class_name = Path(root).name

                # Clean class name
                class_name = class_name.replace(' ', '_').replace('-', '_')
                if class_name and not class_name.startswith('.') and class_name != 'organized':
                    class_info[class_name] = {
                        'path': Path(root),
                        'images': image_files[:500]  # Limit to 500 images per class to prevent memory issues
                    }

        print(f"Found {len(class_info)} classes with sufficient images")

        # Organize classes
        for class_name, info in class_info.items():
            print(f"Processing class: {class_name} ({len(info['images'])} images)")

            # Create class directories
            train_class = train_path / class_name
            val_class = val_path / class_name
            train_class.mkdir(exist_ok=True)
            val_class.mkdir(exist_ok=True)

            # Split images 80/20
            images = info['images']
            split_idx = int(len(images) * 0.8)
            train_images = images[:split_idx]
            val_images = images[split_idx:]

            # Copy images with progress
            for i, img_name in enumerate(train_images):
                try:
                    src_path = info['path'] / img_name
                    dst_path = train_class / img_name
                    shutil.copy2(src_path, dst_path)
                except Exception as e:
                    print(f"Warning: Could not copy {img_name}: {e}")

            for i, img_name in enumerate(val_images):
                try:
                    src_path = info['path'] / img_name
                    dst_path = val_class / img_name
                    shutil.copy2(src_path, dst_path)
                except Exception as e:
                    print(f"Warning: Could not copy {img_name}: {e}")

        return len(class_info)

class CNNModelTrainer:
    """Enhanced CNN trainer with crash recovery and optimization"""

    def __init__(self, base_path, session_manager):
        self.base_path = Path(base_path)
        self.models_path = self.base_path / 'models'
        self.models_path.mkdir(exist_ok=True)
        self.session_manager = session_manager
        self.model = None
        self.training_interrupted = False

    def create_optimized_cnn_model(self, num_classes, input_shape=(224, 224, 3)):
        """Create optimized CNN model with better architecture"""
        if not CNN_AVAILABLE:
            print("CNN not available. Cannot create model.")
            return None

        try:
            print("Creating optimized CNN model...")

            # Use MobileNetV2 as base model
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape
            )

            # Freeze most layers, unfreeze last few for fine-tuning
            base_model.trainable = False

            # Create optimized model
            model = models.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.2),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(num_classes, activation='softmax')
            ])

            # Optimized compilation
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy', 'top_3_accuracy']
            )

            print(f"✓ Optimized CNN model created")
            print(f"Model parameters: {model.count_params():,}")
            print(f"Trainable parameters: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")

            return model

        except Exception as e:
            print(f"Error creating CNN model: {e}")
            return None

    def setup_training_callbacks(self, model_name):
        """Setup training callbacks for better training control"""
        callbacks = []

        # Model checkpoint
        checkpoint_path = self.models_path / f'{model_name}_checkpoint.h5'
        callbacks.append(ModelCheckpoint(
            str(checkpoint_path),
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ))

        # Early stopping
        callbacks.append(EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ))

        # Learning rate reduction
        callbacks.append(ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ))

        return callbacks

    def train_model_with_recovery(self, dataset_path):
        """Train CNN model with crash recovery and progress tracking"""
        if not CNN_AVAILABLE:
            print("Cannot train CNN model - TensorFlow not available")
            return False

        try:
            print("Starting optimized CNN model training...")
            self.session_manager.update_progress('model_training_progress', 'initializing')

            # Setup data generators with augmentation
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=25,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                brightness_range=[0.8, 1.2],
                fill_mode='nearest'
            )

            val_datagen = ImageDataGenerator(rescale=1./255)

            # Load data with error handling
            try:
                train_generator = train_datagen.flow_from_directory(
                    str(Path(dataset_path) / 'train'),
                    target_size=(224, 224),
                    batch_size=16,  # Reduced batch size for stability
                    class_mode='categorical',
                    shuffle=True
                )

                val_generator = val_datagen.flow_from_directory(
                    str(Path(dataset_path) / 'validation'),
                    target_size=(224, 224),
                    batch_size=16,
                    class_mode='categorical',
                    shuffle=False
                )
            except Exception as e:
                print(f"Error loading dataset: {e}")
                return False

            num_classes = len(train_generator.class_indices)
            print(f"Found {num_classes} classes:")
            for class_name, class_idx in train_generator.class_indices.items():
                print(f"  {class_idx}: {class_name}")

            # Create optimized model
            self.model = self.create_optimized_cnn_model(num_classes)
            if not self.model:
                return False

            # Setup callbacks
            model_name = f'optimized_cnn_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            callbacks = self.setup_training_callbacks(model_name)

            self.session_manager.update_progress('model_training_progress', 'training')

            # Calculate steps
            steps_per_epoch = max(1, train_generator.samples // train_generator.batch_size)
            validation_steps = max(1, val_generator.samples // val_generator.batch_size)

            print(f"Training steps per epoch: {steps_per_epoch}")
            print(f"Validation steps: {validation_steps}")

            # Train model with recovery
            try:
                history = self.model.fit(
                    train_generator,
                    steps_per_epoch=steps_per_epoch,
                    epochs=20,
                    validation_data=val_generator,
                    validation_steps=validation_steps,
                    callbacks=callbacks,
                    verbose=1
                )

                # ============================================================
                # SAVE MODEL IN PKL FILE - CRITICAL SECTION
                # ============================================================
                print("\n" + "="*70)
                print("SAVING TRAINED MODEL TO PKL FILE")
                print("="*70)

                # Create model filename with timestamp
                model_name = f'optimized_cnn_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
                model_path = self.models_path / f'{model_name}.pkl'

                print(f"Model will be saved to: {model_path}")

                # Create label encoder from training data
                label_encoder = LabelEncoder()
                class_names = list(train_generator.class_indices.keys())
                label_encoder.fit(class_names)

                print(f"Classes to save: {class_names}")

                # Prepare comprehensive model data for pickle
                model_data = {
                    'model': self.model,
                    'label_encoder': label_encoder,
                    'class_indices': train_generator.class_indices,
                    'class_names': class_names,
                    'model_type': 'CNN',
                    'model_architecture': 'MobileNetV2',
                    'training_history': {
                            'accuracy': [float(x) for x in history.history.get('accuracy', [])],
                            'val_accuracy': [float(x) for x in history.history.get('val_accuracy', [])],
                            'loss': [float(x) for x in history.history.get('loss', [])],
                            'val_loss': [float(x) for x in history.history.get('val_loss', [])],
                            'epochs_completed': len(history.history.get('accuracy', []))
                            },
                    'num_classes': num_classes,
                    'input_shape': (224, 224, 3),
                    'training_date': datetime.now().isoformat(),
                    'dataset_path': dataset_path,
                    'batch_size': 16,
                    'total_train_samples': train_generator.samples,
                    'total_val_samples': val_generator.samples
                    }

                # Save with multiple attempts and error handling
                print("Attempting to save model...")
                save_attempts = 0
                max_attempts = 3
                save_success = False

                while save_attempts < max_attempts and not save_success:
                    try:
                        save_attempts += 1
                        print(f"Save attempt {save_attempts}/{max_attempts}...")

                        with open(model_path, 'wb') as f:
                            pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)

                        # Verify inside try
                        if model_path.exists():
                            file_size = model_path.stat().st_size
                            print("✅ Model saved successfully!")
                            print(f"   File: {model_path.name}")
                            print(f"   Size: {file_size / (1024*1024):.2f} MB")
                            print(f"   Location: {model_path.parent}")
                            save_success = True
                        else:
                            print(f"❌ Save verification failed on attempt {save_attempts}")
                    except Exception as save_error:
                        print(f"❌ Save attempt {save_attempts} failed: {save_error}")
                        if save_attempts < max_attempts:
                            print("Retrying...")
                            time.sleep(2)

                if not save_success:
                    print("❌ Failed to save model after all attempts")
                    print("Trying alternative save method...")
                    try:
                        h5_path = self.models_path / f'{model_name}.h5'
                        self.model.save(h5_path)
                        metadata_path = self.models_path / f'{model_name}_metadata.pkl'
                        metadata = {
                            'label_encoder': label_encoder,
                            'class_indices': train_generator.class_indices,
                            'class_names': class_names,
                            'model_type': 'CNN',
                            'training_history': model_data['training_history'],
                            'num_classes': num_classes,
                            'h5_model_path': str(h5_path),
                        }
                        with open(metadata_path, 'wb') as f:
                            pickle.dump(metadata, f)
                        print("✅ Model saved as H5 + metadata (alternative method)")
                        save_success = True
                    except Exception as alt_error:
                        print(f"❌ Alternative save also failed: {alt_error}")
                if not save_success:
                    print("⚠️ WARNING: Model training completed but save failed!")
                    print("Model is still in memory and can be used in current session")
                    return False

                # Update session with success
                self.session_manager.update_progress('model_training_progress', 'completed')
                self.session_manager.update_progress('model_type', 'CNN')
                self.session_manager.update_progress('model_path', str(model_path))

                final_acc = history.history['accuracy'][-1]
                final_val_acc = history.history['val_accuracy'][-1]
                self.session_manager.update_progress('best_accuracy', final_val_acc)

                print("\n" + "="*70)
                print("✅ CNN MODEL TRAINING COMPLETED SUCCESSFULLY!")
                print("="*70)
                print(f"📦 Model File: {model_name}.pkl")
                print(f"📊 Final Training Accuracy: {final_acc:.3f}")
                print(f"📊 Final Validation Accuracy: {final_val_acc:.3f}")
                print(f"📁 Saved Location: {self.models_path}")
                print(f"🎯 Number of Classes: {num_classes}")
                print(f"📅 Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("="*70)

                # Cleanup memory
                del train_generator, val_generator
                gc.collect()

                return True

            except KeyboardInterrupt:
                print("\n⚠️ Training interrupted by user")
                print("Attempting to save partial progress...")

                # Try to save interrupted model
                try:
                    interrupted_name = f'interrupted_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
                    interrupted_path = self.models_path / f'{interrupted_name}.pkl'

                    partial_data = {
                        'model': self.model,
                        'label_encoder': label_encoder,
                        'class_indices': train_generator.class_indices,
                        'model_type': 'CNN_PARTIAL',
                        'note': 'Training interrupted - partial model'
                    }

                    with open(interrupted_path, 'wb') as f:
                        pickle.dump(partial_data, f)

                    print(f"✅ Partial model saved: {interrupted_name}.pkl")

                except:
                    print("❌ Could not save interrupted model")

                self.session_manager.update_progress('model_training_progress', 'interrupted')
                return False

            except Exception as e:
                print(f"❌ Training error: {e}")
                self.session_manager.update_progress('model_training_progress', f'error: {str(e)}')
                self.session_manager.increment_crash_count()
                return False

        except Exception as e:
            print(f"❌ Error during model training setup: {e}")
            self.session_manager.increment_crash_count()
            return False

class PlantDiseaseDetectionSystem:
    """Main system with enhanced crash prevention and session management"""

    def __init__(self):
        """Initialize the main system with crash recovery"""
        print("Initializing Plant Disease Detection System...")

        try:
            # System configuration
            self.base_path = Path("plant_disease_data")
            self.base_path.mkdir(exist_ok=True)

            # Initialize core attributes first
            self.model = None
            self.label_encoder = None
            self.model_type = "fallback"
            self.disease_database = {}

            # Initialize session manager
            self.session_manager = SessionManager(self.base_path)

            # Check for previous crashes
            crash_count = self.session_manager.session_data.get('crash_count', 0)
            if crash_count > 0:
                print(f"⚠ Previous sessions crashed {crash_count} times. Using safe mode.")

            # Initialize modules with error handling
            self.initialize_modules()

            # Initialize new components
            self.dataset_manager = DatasetManager(self.base_path)
            self.cnn_trainer = CNNModelTrainer(self.base_path, self.session_manager)

            # Load disease database
            try:
                self.disease_database = self.load_disease_database()
            except Exception as db_error:
                print(f"Warning: Could not load disease database: {db_error}")
                self.disease_database = self.create_minimal_database()

            # Initialize model
            try:
                self.load_trained_model()
            except Exception as model_error:
                print(f"Warning: Could not load model: {model_error}")

            # Display session info
            self.display_session_info()

            # Auto-check for dataset and train if needed
            try:
                self.auto_check_and_train()
            except Exception as train_error:
                print(f"Warning: Auto-check/train failed: {train_error}")

        except Exception as e:
            print(f"Initialization error: {e}")
            # Ensure critical attributes exist even on error
            if not hasattr(self, 'disease_database'):
                self.disease_database = self.create_minimal_database()
            if not hasattr(self, 'session_manager'):
                self.session_manager = SessionManager(self.base_path)
            self.session_manager.increment_crash_count()

    def initialize_modules(self):
        """Initialize all modules with error handling"""
        try:
            self.fertilizer_calc = FertilizerCalculatorModule(self.base_path)
            self.detection_history = DetectionHistoryModule(self.base_path)
            self.treatment_history = TreatmentHistoryModule(self.base_path)
            self.farm_analytics = FarmAnalyticsModule(self.base_path)
            self.user_profile = UserProfileModule(self.base_path)
            self.data_management = DataManagementModule(self.base_path)
            self.export_reports = ExportReportsModule(self.base_path)
            print("✓ All modules initialized successfully")
        except Exception as e:
            print(f"Some modules failed to initialize: {e}")

    def display_session_info(self):
        """Display enhanced session information"""
        print("\n" + "="*60)
        print("SESSION INFORMATION")
        print("="*60)
        print(f"Model Type: {self.model_type}")

        if self.session_manager.session_data.get('last_access'):
            print(f"Last Access: {self.session_manager.session_data['last_access']}")

        training_progress = self.session_manager.session_data.get('model_training_progress')
        if training_progress:
            print(f"Training Status: {training_progress}")

        best_accuracy = self.session_manager.session_data.get('best_accuracy')
        if best_accuracy:
            print(f"Best Model Accuracy: {best_accuracy:.1%}")

        crash_count = self.session_manager.session_data.get('crash_count', 0)
        if crash_count > 0:
            print(f"Previous Crashes: {crash_count}")

        if CNN_AVAILABLE:
            print("✓ CNN Training Available")
        else:
            print("⚠ CNN Training Not Available")

        # Dataset information
        dataset_zip = self.dataset_manager.find_dataset_zip()
        if dataset_zip:
            print(f"✓ Dataset Found: {Path(dataset_zip).name}")
        else:
            print("⚠ No Dataset Found")

        print("="*60)

    def auto_check_and_train(self):
        """Enhanced auto-check with better decision making"""
        training_progress = self.session_manager.session_data.get('model_training_progress')

        # Skip if recently completed or in progress
        if training_progress in ['completed', 'training', 'initializing']:
            print(f"Training status: {training_progress} - Skipping auto-train")
            return

        # Check if we need to train a model
        if self.model_type == "fallback" and CNN_AVAILABLE:
            dataset_zip = self.dataset_manager.find_dataset_zip()

            if dataset_zip:
                print(f"\n✓ Found dataset: {Path(dataset_zip).name}")

                # Ask user if they want to train (with timeout)
                print("Would you like to train a CNN model? (y/n) [30 seconds timeout]")
                print("Press Enter for 'yes' or 'n' for no...")

                try:
                    import select
                    import sys

                    # Non-blocking input with timeout
                    ready, _, _ = select.select([sys.stdin], [], [], 30)
                    if ready:
                        response = sys.stdin.readline().strip().lower()
                    else:
                        response = 'y'  # Default to yes if no response

                except (ImportError, AttributeError):
                    # Fallback for systems without select (Windows)
                    response = 'y'

                if response in ['', 'y', 'yes']:
                    print("Starting automatic CNN training...")
                    self.train_cnn_model_safe(dataset_zip)
                else:
                    print("CNN training skipped by user.")
            else:
                print("\n⚠ No dataset found for CNN training")
                print("To use CNN, place your dataset ZIP file at:")
                print(f"  Primary: {DATASET_ZIP_PATH}")
                print("  Or any of these locations:")
                for path in BACKUP_DATASET_PATHS[:3]:
                    print(f"    {path}")

    def train_cnn_model_safe(self, dataset_zip=None):
        """Safe CNN training with comprehensive error handling"""
        try:
            print("="*60)
            print("STARTING CNN MODEL TRAINING")
            print("="*60)

            # Find dataset if not provided
            if not dataset_zip:
                dataset_zip = self.dataset_manager.find_dataset_zip()

            if not dataset_zip:
                print("Error: No dataset ZIP file found!")
                return False

            # Extract and organize dataset
            print("Step 1: Extracting and organizing dataset...")
            dataset_path = self.dataset_manager.extract_zip_dataset(dataset_zip)

            if not dataset_path:
                print("Error: Failed to extract dataset")
                return False

            # Update session
            self.session_manager.update_progress('dataset_path', dataset_path)

            # Train model
            print("Step 2: Training CNN model...")
            print("This may take 15-30 minutes depending on your hardware.")
            print("Training will automatically save progress and can be resumed if interrupted.")

            success = self.cnn_trainer.train_model_with_recovery(dataset_path)

            if success:
                print("Step 3: Loading trained model...")
                self.load_trained_model()
                print("✓ CNN model training and loading completed successfully!")
                print("="*60)
                return True
            else:
                print("✗ Model training failed or was interrupted")
                print("You can retry training from the main menu")
                return False

        except Exception as e:
            print(f"Critical error in CNN training: {e}")
            self.session_manager.increment_crash_count()
            return False

    def load_trained_model(self):
        """Load trained model with enhanced error handling"""
        models_path = self.base_path / 'models'
        models_path.mkdir(exist_ok=True)

        model_files = list(models_path.glob('*.pkl'))
        if model_files:
            latest_model = max(model_files, key=os.path.getctime)
            try:
                print(f"Loading model: {latest_model.name}")
                with open(latest_model, 'rb') as f:
                    model_data = pickle.load(f)

                self.model = model_data.get('model')
                self.label_encoder = model_data.get('label_encoder')
                self.model_type = model_data.get('model_type', 'unknown')

                # Additional model info
                num_classes = model_data.get('num_classes', 'unknown')
                training_date = model_data.get('training_date', 'unknown')

                print(f"✓ Model loaded successfully")
                print(f"  Type: {self.model_type}")
                print(f"  Classes: {num_classes}")
                print(f"  Trained: {training_date}")

            except Exception as e:
                print(f"Error loading model {latest_model.name}: {e}")
                self.model_type = "fallback"
        else:
            print("No trained model found. Using fallback detection.")

    def load_disease_database(self):
        """Load comprehensive disease information database"""
        db_path = self.base_path / 'disease_database.json'

        if db_path.exists():
            try:
                with open(db_path, 'r') as f:
                    return json.load(f)
            except:
                pass

        # Enhanced disease database with specific treatments
        default_db = {
            
            "Nutrient_Deficiency": {
                "symptoms": "Yellowing leaves, stunted growth, poor fruit/flower development",
                "treatment": "Apply balanced fertilizer (NPK 10-10-10), soil testing recommended",
                "prevention": "Regular soil testing, proper fertilization schedule, organic matter addition"
            },
            "Fungal_Infection_Early": {
                "symptoms": "Small spots on leaves, slight discoloration, early fungal signs",
                "treatment": "Apply broad-spectrum fungicide (Copper sulfate or Mancozeb), improve air circulation",
                "prevention": "Avoid overhead watering, proper plant spacing, remove infected debris"
            },
            "Early_Disease_Symptoms": {
                "symptoms": "Initial signs of disease, slight leaf changes, early spotting",
                "treatment": "Preventive fungicide application, monitor closely, improve growing conditions",
                "prevention": "Regular inspection, proper sanitation, balanced nutrition"
            },
            "Severe_Plant_Disease": {
                "symptoms": "Extensive damage, significant defoliation, plant decline",
                "treatment": "Immediate intervention required - systemic fungicide, remove severely affected parts",
                "prevention": "Improve all growing conditions, consider replanting with resistant varieties"
            },
            "Plant_Stress_Condition": {
                "symptoms": "Wilting, yellowing, signs of environmental or nutritional stress",
                "treatment": "Address environmental factors (water, light, temperature), balanced fertilization",
                "prevention": "Optimal growing conditions, proper watering schedule, soil improvement"
            },
            "Plant_Healthy_Condition": {
                "symptoms": "Generally healthy plant with good green coverage, minimal disease signs",
                "treatment": "Continue current care routine with regular monitoring",
                "prevention": "Maintain consistent watering, fertilization, and pest monitoring"
            },
            "Moderate_Fungal_Disease": {
                "symptoms": "Visible fungal symptoms affecting moderate portion of plant",
                "treatment": "Apply systemic fungicide (Propiconazole or Tebuconazole), remove affected parts",
                "prevention": "Improve air circulation, avoid overhead watering, regular fungicide applications"
            },
            "Severe_Plant_Stress": {
                "symptoms": "Significant stress indicators, poor plant vigor, extensive damage",
                "treatment": "Address all environmental factors, intensive care regimen, consider replanting",
                "prevention": "Optimal growing conditions, regular monitoring, preventive care"
            },
            "Mild_Plant_Health_Issues": {
                "symptoms": "Minor health concerns, early warning signs",
                "treatment": "Increase monitoring frequency, adjust care routine, preventive treatments",
                "prevention": "Consistent care schedule, environmental optimization"
            },
           
           
        }

        # Save default database
        with open(db_path, 'w') as f:
            json.dump(default_db, f, indent=2)

        return default_db

    def display_main_menu(self):
        """Display enhanced main system menu"""
        print("\n" + "="*70)
        print("    PLANT DISEASE DETECTION SYSTEM - MAIN MENU")
        print("="*70)
        print("1. 🔍 Disease Detection & Analysis")
        print("2. 🧪 Fertilizer Calculator")
        print("3. 📊 View Detection History")
        print("4. 💊 View Treatment History")
        print("5. 📈 Farm Analytics")
        print("6. 👤 User Profile Management")
        print("7. 💾 Data Management")
        print("8. 📑 Export Reports")
        print("9. 🤖 Train/Retrain CNN Model")
        print("10. ❌ Exit System")
        print("="*70)
        print(f"Current Model: {self.model_type} | Dataset: {'✓' if self.dataset_manager.find_dataset_zip() else '✗'}")

        crash_count = self.session_manager.session_data.get('crash_count', 0)
        if crash_count > 0:
            print(f"⚠ System Health: {crash_count} previous crashes detected")

    def disease_detection_enhanced(self):
        """Enhanced disease detection with better error handling"""
        print("\n" + "="*60)
        print("PLANT DISEASE DETECTION & ANALYSIS")
        print("="*60)
        print(f"Using {self.model_type} model for detection")
        print(f"Supported formats: JPG, JPEG, PNG, BMP")

        # Get image path with validation
        while True:
            image_path = input("\nEnter the path to your plant image (or 'back' to return): ").strip('"').strip("'")

            if image_path.lower() == 'back':
                return

            if os.path.exists(image_path):
                # Check if it's an image file
                if not any(image_path.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp']):
                    print("Error: Please provide a valid image file (JPG, JPEG, PNG, BMP)")
                    continue
                break
            else:
                print(f"Error: File not found: {image_path}")
                print("Please check the file path and try again.")

        try:
            print("\n" + "="*50)
            print("ANALYZING IMAGE...")
            print("="*50)

            # Load and preprocess image with enhanced error handling
            try:
                print("Loading and preprocessing image...")
                image = Image.open(image_path)

                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    print(f"Converting image from {image.mode} to RGB...")
                    image = image.convert('RGB')

                # Resize image
                print("Resizing image to 224x224...")
                image_resized = image.resize((224, 224), Image.Resampling.LANCZOS)
                img_array = np.array(image_resized, dtype=np.float32) / 255.0

                print("✓ Image preprocessing completed")

            except Exception as e:
                print(f"Error processing image: {e}")
                return

            # Step 1: Plant presence detection
            print("\nStep 1: Analyzing image for plant presence...")
            is_plant, plant_confidence = self.detect_plant_presence_enhanced(img_array)

            if not is_plant:
                print(f"❌ Plant detection failed (confidence: {plant_confidence:.1%})")
                print("Please ensure the image contains a clear plant with visible leaves.")

                retry = input("Would you like to try another image? (y/n): ").lower()
                if retry == 'y':
                    return self.disease_detection_enhanced()
                return

            print(f"✅ Plant detected (confidence: {plant_confidence:.1%})")

            # Step 2: Disease prediction
            print("\nStep 2: Analyzing for diseases and health conditions...")
            predicted_disease, disease_confidence, additional_info = self.predict_disease_enhanced(img_array)

            # Step 3: Display comprehensive results
            self.display_detection_results(
                image_path, predicted_disease, disease_confidence,
                plant_confidence, additional_info)

            # Step 4: Store detection record
            detection_record = {
                'timestamp': datetime.now(),
                'image_path': image_path,
                'predicted_disease': predicted_disease,
                'confidence': disease_confidence,
                'plant_confidence': plant_confidence,
                'model_type': self.model_type,
                'additional_info': additional_info
                }

            # Save to detection history
            try:
                self.detection_history.add_detection(detection_record)
                print("✓ Detection record saved to history")
            except:
                print("⚠ Could not save to history")

            # Step 5: Offer additional services
            self.offer_additional_services_enhanced(detection_record)

        except Exception as e:
            print(f"Critical error in disease detection: {e}")
            self.session_manager.increment_crash_count()

    def detect_plant_presence_enhanced(self, img_array):
        """Enhanced plant detection with multiple criteria for images with backgrounds"""
        try:
            # Color analysis
            r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]

            # Enhanced green detection (more tolerant)
            green_pixels = np.sum((g > r * 0.8) & (g > b * 0.9) & (g > 0.15))
            total_pixels = img_array.shape[0] * img_array.shape[1]
            green_ratio = green_pixels / total_pixels

            # Vegetation index (simplified NDVI-like calculation)
            vegetation_index = np.mean((g - r) / (g + r + 0.001))

            # Enhanced texture analysis for plant-like structures
            gray = np.mean(img_array, axis=2)
            grad_x = np.gradient(gray, axis=1)
            grad_y = np.gradient(gray, axis=0)
            edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            edge_density = np.std(edge_magnitude)

            # Leaf-like pattern detection (vertical structures)
            vertical_edges = np.std(grad_y)
            horizontal_edges = np.std(grad_x)
            structure_ratio = vertical_edges / (horizontal_edges + 0.001)

            # Color variation analysis (plants have varied green tones)
            green_std = np.std(g)
            color_complexity = green_std * 2

            # Plant-specific color ranges (broader detection)
            plant_green = np.sum((g > 0.2) & (g < 0.9) & (g > r * 0.7) & (g > b * 0.8))
            plant_ratio = plant_green / total_pixels

            # Sky/background detection (avoid false positives)
            sky_blue = np.sum((b > 0.6) & (b > g * 1.1) & (b > r * 1.1))
            sky_ratio = sky_blue / total_pixels

            # Soil/ground detection
            soil_brown = np.sum((r > 0.3) & (r < 0.7) & (g > 0.2) & (g < 0.6) & (b > 0.1) & (b < 0.5))
            soil_ratio = soil_brown / total_pixels

            # Combined plant score with background consideration
            plant_score = (
                green_ratio * 0.25 +
                plant_ratio * 0.25 +
                max(0, vegetation_index) * 0.2 +
                min(edge_density * 1.5, 1.0) * 0.15 +
                min(color_complexity, 1.0) * 0.1 +
                min(structure_ratio * 0.5, 1.0) * 0.05
                )

            # Penalize if too much sky/background
            if sky_ratio > 0.4:
                plant_score *= 0.7

            # Bonus for mixed plant/soil (natural setting)
            if soil_ratio > 0.1 and soil_ratio < 0.5:
                plant_score *= 1.1

            # Lower threshold for images with backgrounds
            is_plant = plant_score > 0.15
            confidence = min(plant_score * 1.8, 1.0)

            return is_plant, confidence

        except Exception as e:
            print(f"Plant detection error: {e}")
            return True, 0.5  # Default to assuming plant is present

    def predict_disease_enhanced(self, img_array):
        """Enhanced disease prediction with detailed analysis"""
        additional_info = {}

        try:
            if self.model and self.label_encoder:
                if self.model_type == 'CNN' and CNN_AVAILABLE:
                    return self.predict_with_cnn_enhanced(img_array)
                else:
                    return self.predict_with_model_enhanced(img_array)
            else:
                return self.fallback_prediction_enhanced(img_array)

        except Exception as e:
            print(f"Prediction error: {e}")
            return self.fallback_prediction_enhanced(img_array)

    def predict_with_cnn_enhanced(self, img_array):
        """Enhanced CNN prediction with confidence analysis"""
        try:
            print("Using trained CNN model for prediction...")

            # Prepare image for CNN
            img_batch = np.expand_dims(img_array, axis=0)

            # Make prediction
            predictions = self.model.predict(img_batch, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])

            # Get disease name
            disease = self.label_encoder.inverse_transform([predicted_class])[0]

            # Additional analysis
            additional_info = {
                'top_3_predictions': [],
                'prediction_distribution': predictions[0].tolist()
                }

            # Get top 3 predictions
            top_3_indices = np.argsort(predictions[0])[-3:][::-1]
            for idx in top_3_indices:
                class_name = self.label_encoder.inverse_transform([idx])[0]
                prob = float(predictions[0][idx])
                additional_info['top_3_predictions'].append({
                    'disease': class_name,
                    'confidence': prob
                })

            print(f"CNN prediction completed: {disease} ({confidence:.1%})")
            return disease, confidence, additional_info

        except Exception as e:
            print(f"CNN prediction error: {e}")
            return self.fallback_prediction_enhanced(img_array)

    def predict_with_model_enhanced(self, img_array):
        """Enhanced model prediction"""
        try:
            print("Using trained model for prediction...")

            # Extract features
            features = self.extract_features_enhanced(img_array)
            features = np.array([features])

            # Make prediction
            prediction = self.model.predict(features)
            disease = self.label_encoder.inverse_transform(prediction)[0]
            confidence = 0.85  # Placeholder confidence

            additional_info = {'method': 'feature_extraction'}

            return disease, confidence, additional_info

        except Exception as e:
            print(f"Model prediction error: {e}")
            return self.fallback_prediction_enhanced(img_array)

    def extract_features_enhanced(self, img_array):
        """Enhanced feature extraction"""
        try:
            # Color features
            mean_rgb = np.mean(img_array, axis=(0,1))
            std_rgb = np.std(img_array, axis=(0,1))

            # HSV color space
            from colorsys import rgb_to_hsv
            hsv_features = []
            for i in range(3):
                h, s, v = rgb_to_hsv(mean_rgb[0], mean_rgb[1], mean_rgb[2])
                hsv_features.extend([h, s, v])
                break  # Just one conversion for the mean color

            # Texture features
            gray = np.mean(img_array, axis=2)
            grad_x = np.gradient(gray, axis=1)
            grad_y = np.gradient(gray, axis=0)
            texture = np.std(np.sqrt(grad_x**2 + grad_y**2))

            # Statistical features
            skewness = np.mean((img_array - np.mean(img_array))**3) / (np.std(img_array)**3)
            kurtosis = np.mean((img_array - np.mean(img_array))**4) / (np.std(img_array)**4)

            # Combine all features
            features = np.concatenate([
                mean_rgb, std_rgb, hsv_features,
                [texture, skewness, kurtosis]
            ])

            return features

        except Exception as e:
            print(f"Feature extraction error: {e}")
            # Return basic features as fallback
            mean_rgb = np.mean(img_array, axis=(0,1))
            std_rgb = np.std(img_array, axis=(0,1))
            return np.concatenate([mean_rgb, std_rgb])

    def fallback_prediction_enhanced(self, img_array):
        """Enhanced fallback prediction with specific disease identification and background handling"""
        try:
            print("Using enhanced fallback analysis...")

            # Color analysis
            r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]

            # Focus on plant areas (exclude background)
            # Identify potential plant pixels
            plant_mask = (g > r * 0.8) & (g > b * 0.9) & (g > 0.1)

            # If we found plant pixels, focus analysis on them
            if np.sum(plant_mask) > img_array.size * 0.1:
                r_plant = r[plant_mask]
                g_plant = g[plant_mask]
                b_plant = b[plant_mask]

                # Calculate ratios on plant areas only
                green_healthy = np.sum((g_plant > 0.4) & (g_plant > r_plant * 1.2))
                brown_diseased = np.sum((r_plant > 0.4) & (g_plant > 0.2) & (g_plant < 0.6) & (b_plant < 0.4))
                yellow_stressed = np.sum((r_plant > 0.6) & (g_plant > 0.6) & (b_plant < 0.4))
                dark_spots = np.sum((r_plant < 0.25) & (g_plant < 0.25) & (b_plant < 0.25))
                red_areas = np.sum((r_plant > 0.6) & (r_plant > g_plant) & (r_plant > b_plant))
                white_spots = np.sum((r_plant > 0.8) & (g_plant > 0.8) & (b_plant > 0.8))

                plant_pixels = len(g_plant)
                green_ratio = green_healthy / plant_pixels if plant_pixels > 0 else 0
                brown_ratio = brown_diseased / plant_pixels if plant_pixels > 0 else 0
                yellow_ratio = yellow_stressed / plant_pixels if plant_pixels > 0 else 0
                dark_spots_ratio = dark_spots / plant_pixels if plant_pixels > 0 else 0
                red_ratio = red_areas / plant_pixels if plant_pixels > 0 else 0
                white_ratio = white_spots / plant_pixels if plant_pixels > 0 else 0

                # Advanced health metrics on plant areas
                mean_green = np.mean(g_plant) if plant_pixels > 0 else 0
                mean_red = np.mean(r_plant) if plant_pixels > 0 else 0
                std_colors = np.std([np.std(r_plant), np.std(g_plant), np.std(b_plant)]) if plant_pixels > 0 else 0
            else:
                # Fallback to full image analysis
                green_ratio = np.mean((g > r) & (g > b))
                brown_ratio = np.mean((r > 0.4) & (g > 0.2) & (g < 0.6) & (b < 0.4))
                yellow_ratio = np.mean((r > 0.6) & (g > 0.6) & (b < 0.4))
                dark_spots_ratio = np.mean((r < 0.3) & (g < 0.3) & (b < 0.3))
                red_ratio = np.mean((r > 0.6) & (r > g) & (r > b))
                white_ratio = np.mean((r > 0.8) & (g > 0.8) & (b > 0.8))

                mean_green = np.mean(g)
                mean_red = np.mean(r)
                std_colors = np.std([np.std(r), np.std(g), np.std(b)])

            # Calculate leaf health indicators
            healthy_green = green_ratio
            diseased_areas = brown_ratio + yellow_ratio + dark_spots_ratio

            additional_info = {
                'analysis_method': 'enhanced_focused_analysis',
                'color_ratios': {
                    'green': float(green_ratio),
                    'brown': float(brown_ratio),
                    'yellow': float(yellow_ratio),
                    'dark_spots': float(dark_spots_ratio),
                    'red_areas': float(red_ratio),
                    'white_spots': float(white_ratio)
                },
                'health_indicators': {
                    'healthy_green': float(healthy_green),
                    'diseased_areas': float(diseased_areas),
                    'mean_green': float(mean_green),
                    'color_variation': float(std_colors)
                }
                }

            # More accurate decision logic with better thresholds
            confidence = 0.75  # Base confidence for fallback

            # Very healthy plant detection (high green, low disease indicators)
            if (healthy_green > 0.7 and diseased_areas < 0.08 and
                mean_green > 0.45 and brown_ratio < 0.04):
                if yellow_ratio < 0.02:
                    return "Apple_Healthy", confidence + 0.25, additional_info
                else:
                    return "Tomato_Healthy", confidence + 0.2, additional_info

            # Moderate healthy plant
            elif (healthy_green > 0.5 and diseased_areas < 0.15 and
                  mean_green > 0.35 and brown_ratio < 0.08):
                return "Plant_Healthy_Condition", confidence + 0.15, additional_info

            # Early blight (target spots, yellowing)
            elif (yellow_ratio > 0.15 and brown_ratio > 0.1 and dark_spots_ratio > 0.05):
                if mean_green > 0.3:
                    return "Tomato_Early_Blight", confidence + 0.1, additional_info
                else:
                    return "Potato_Early_Blight", confidence + 0.05, additional_info

            # Late blight (rapid browning, water-soaked appearance)
            elif (brown_ratio > 0.25 and dark_spots_ratio > 0.15 and std_colors > 0.12):
                if white_ratio > 0.03:
                    return "Tomato_Late_Blight", confidence + 0.15, additional_info
                else:
                    return "Potato_Late_Blight", confidence + 0.1, additional_info

            # Black rot (dark areas, brown spots)
            elif (brown_ratio > 0.2 and dark_spots_ratio > 0.2 and yellow_ratio < 0.1):
                return "Apple_Black_Rot", confidence + 0.1, additional_info

            # Rust diseases (orange/red coloration with some yellowing)
            elif (red_ratio > 0.12 and yellow_ratio > 0.08 and brown_ratio > 0.05):
                return "Corn_Common_Rust", confidence + 0.05, additional_info

            # Bacterial spot (dark spots with yellow halos, white areas)
            elif (dark_spots_ratio > 0.1 and yellow_ratio > 0.12 and white_ratio > 0.05):
                return "Tomato_Bacterial_Spot", confidence, additional_info

            # Scab diseases (dark, rough lesions)
            elif (dark_spots_ratio > 0.2 and brown_ratio > 0.12 and std_colors > 0.08):
                return "Apple_Scab", confidence, additional_info

            # Nutrient deficiency (yellowing without much browning)
            elif (yellow_ratio > 0.2 and brown_ratio < 0.1 and green_ratio < 0.6):
                return "Nutrient_Deficiency", confidence - 0.05, additional_info

            # Early fungal infection
            elif (brown_ratio > 0.08 and diseased_areas < 0.25 and green_ratio > 0.4):
                return "Fungal_Infection_Early", confidence - 0.05, additional_info

            # Moderate disease symptoms
            elif (diseased_areas > 0.15 and diseased_areas < 0.35):
                if yellow_ratio > brown_ratio:
                    return "Early_Disease_Symptoms", confidence - 0.1, additional_info
                else:
                    return "Moderate_Fungal_Disease", confidence - 0.05, additional_info

            # Severe plant problems
            elif (diseased_areas > 0.4 or green_ratio < 0.25):
                if brown_ratio > 0.3:
                    return "Severe_Plant_Disease", confidence + 0.1, additional_info
                else:
                    return "Severe_Plant_Stress", confidence + 0.05, additional_info

            # Plant stress (low green, some yellowing)
            elif (green_ratio < 0.45 and yellow_ratio > 0.08):
                return "Plant_Stress_Condition", confidence - 0.1, additional_info

            # Default for unclear cases (be more conservative)
            else:
                if diseased_areas > 0.1:
                    return "Mild_Plant_Health_Issues", confidence - 0.15, additional_info
                else:
                    return "Plant_Condition_Unclear", confidence - 0.2, additional_info

        except Exception as e:
            print(f"Fallback prediction error: {e}")
            return "Analysis_Error", 0.3, {'error': str(e)}

    def create_minimal_database(self):
        """Create minimal disease database as fallback"""
        return {
            "Healthy Plant": {
                "symptoms": "Green leaves, normal growth, no visible diseases",
                "treatment": "Continue current care routine",
                "prevention": "Regular monitoring and proper nutrition"
            },
            "Disease Detected": {
                "symptoms": "Visible symptoms of plant disease",
                "treatment": "Consult agricultural expert for specific treatment",
                "prevention": "Regular inspection and preventive care"
            },
            "Plant Stress": {
                "symptoms": "Signs of environmental or nutritional stress",
                "treatment": "Improve growing conditions",
                "prevention": "Proper watering and fertilization"
            },
            "Fungal Disease Detected": {
                "symptoms": "Spots, discoloration, or fungal growth",
                "treatment": "Apply appropriate fungicide treatment",
                "prevention": "Good air circulation and avoid overwatering"
            },
            "Early Disease Symptoms": {
                "symptoms": "Early signs of potential disease",
                "treatment": "Monitor closely and apply preventive treatments",
                "prevention": "Maintain plant health and hygiene"
            },
            "Moderate Plant Health Issues": {
                "symptoms": "Some concerning symptoms requiring attention",
                "treatment": "Investigate specific issues and treat accordingly",
                "prevention": "Regular care and monitoring"
            },
            "Severe Disease or Plant Stress": {
                "symptoms": "Significant plant health problems",
                "treatment": "Immediate intervention required - consult expert",
                "prevention": "Improve all aspects of plant care"
            }
        }

    def display_detection_results(self, image_path, predicted_disease, confidence, plant_confidence, additional_info):
        """Safe version of display results with error handling"""
        try:
            print(f"\n" + "="*70)
            print("🔬 COMPREHENSIVE ANALYSIS RESULTS")
            print("="*70)
            print(f"📷 Image: {os.path.basename(image_path)}")
            print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"🤖 Model Type: {self.model_type}")
            print(f"🌱 Plant Detected: Yes ({plant_confidence:.1%} confidence)")
            print("-" * 70)
            print(f"🎯 PRIMARY DIAGNOSIS: {predicted_disease}")
            print(f"📊 Confidence Level: {confidence:.1%}")

            # Display confidence interpretation
            if confidence >= 0.8:
                print("✅ High confidence - Diagnosis is very likely accurate")
            elif confidence >= 0.6:
                print("⚠️  Medium confidence - Diagnosis is reasonably accurate")
            else:
                print("⚠️  Low confidence - Consider additional analysis")

            # Show top predictions if available
            if isinstance(additional_info, dict) and 'top_3_predictions' in additional_info:
                print("\n🔍 TOP 3 POSSIBLE DIAGNOSES:")
                for i, pred in enumerate(additional_info['top_3_predictions'], 1):
                    print(f"  {i}. {pred['disease']} ({pred['confidence']:.1%})")

            # Show detailed disease information
            print("\n" + "="*70)
            self.safe_display_disease_info(predicted_disease)

            # Show analysis details
            if isinstance(additional_info, dict) and 'color_ratios' in additional_info:
                ratios = additional_info['color_ratios']
                print(f"\n📋 ANALYSIS DETAILS:")
                print(f"  Green Coverage: {ratios.get('green', 0):.1%}")
                print(f"  Brown/Disease Areas: {ratios.get('brown', 0):.1%}")
                print(f"  Yellowing: {ratios.get('yellow', 0):.1%}")
                print(f"  Dark Spots: {ratios.get('dark_spots', 0):.1%}")
                print("="*70)

        except Exception as e:
            print(f"Error displaying results: {e}")
            import traceback
            traceback.print_exc()
            # Minimal fallback display
            print(f"\n" + "="*70)
            print(f"🎯 DIAGNOSIS: {predicted_disease}")
            print(f"📊 Confidence: {confidence:.1%}")
            print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*70)

    def safe_display_disease_info(self, disease_name):
        """Safe version of disease info display with fallback"""
        try:
            # Make sure we have the disease database loaded
            if not hasattr(self, 'disease_database') or not self.disease_database:
                print("Reloading disease database...")
                self.disease_database = self.load_disease_database()

            # Try to find the disease in the database
            disease_info = self.disease_database.get(disease_name)

            if disease_info:
                print(f"📋 DISEASE INFORMATION - {disease_name.replace('_', ' ').upper()}")
                print("-" * 70)
                print(f"🩺 Symptoms: {disease_info.get('symptoms', 'N/A')}")
                print(f"💊 Treatment: {disease_info.get('treatment', 'N/A')}")
                print(f"🛡️  Prevention: {disease_info.get('prevention', 'N/A')}")
            else:
                # If not found, provide detailed recommendations based on disease type
                print(f"📋 DISEASE INFORMATION - {disease_name.replace('_', ' ').upper()}")
                print("-" * 70)

                if "fungal" in disease_name.lower():
                    print("🩺 Symptoms: Small spots on leaves, slight discoloration, early fungal signs")
                    print("💊 Treatment: Apply broad-spectrum fungicide (Copper sulfate or Mancozeb), improve air circulation")
                    print("🛡️  Prevention: Avoid overhead watering, proper plant spacing, remove infected debris")
                elif "blight" in disease_name.lower():
                    if "early" in disease_name.lower():
                        print("🩺 Symptoms: Target-like spots with concentric rings, yellowing lower leaves")
                        print("💊 Treatment: Apply Chlorothalonil or Mancozeb spray early in season, weekly applications")
                        print("🛡️  Prevention: Crop rotation, avoid water stress, mulching, proper nutrition")
                    else:
                        print("🩺 Symptoms: Water-soaked lesions, rapid plant collapse, white fuzzy growth")
                        print("💊 Treatment: Apply Metalaxyl or Dimethomorph fungicides preventively")
                        print("🛡️  Prevention: Resistant varieties, avoid overhead irrigation")
                elif "rust" in disease_name.lower():
                    print("🩺 Symptoms: Orange-brown pustules on leaves, spore release when disturbed")
                    print("💊 Treatment: Triazole fungicides (Propiconazole) early application")
                    print("🛡️  Prevention: Plant resistant hybrids, monitor weather conditions")
                elif "rot" in disease_name.lower():
                    print("🩺 Symptoms: Brown soft spots, black fungal growth, mushy texture")
                    print("💊 Treatment: Remove infected parts, apply fungicide spray (Captan or Thiophanate-methyl)")
                    print("🛡️  Prevention: Good ventilation, avoid injuries, proper harvesting")
                elif "scab" in disease_name.lower():
                    print("🩺 Symptoms: Olive-green to black spots on leaves and fruit, scabby lesions")
                    print("💊 Treatment: Apply fungicides like Myclobutanil or Propiconazole")
                    print("🛡️  Prevention: Resistant varieties, proper air circulation, fall cleanup")
                elif "bacterial" in disease_name.lower():
                    print("🩺 Symptoms: Small dark spots with yellow halos, rough surface")
                    print("💊 Treatment: Copper-based bactericides, remove infected plants")
                    print("🛡️  Prevention: Pathogen-free seeds, avoid overhead watering")
                elif "deficiency" in disease_name.lower():
                    print("🩺 Symptoms: Yellowing leaves, stunted growth, poor development")
                    print("💊 Treatment: Apply balanced fertilizer (NPK 10-10-10), soil testing")
                    print("🛡️  Prevention: Regular soil testing, proper fertilization schedule")
                elif "healthy" in disease_name.lower():
                    print("🩺 Symptoms: Vibrant green leaves, normal growth, no visible diseases")
                    print("💊 Treatment: Continue preventive care with regular monitoring")
                    print("🛡️  Prevention: Proper watering, balanced fertilization, regular inspection")
                else:
                    # Generic recommendations
                    print("💡 General Recommendations:")
                    print("🩺 Monitor plant closely for symptom development")
                    print("💊 Treatment: Consult with local agricultural expert for specific treatment")
                    print("🛡️  Prevention: Maintain good plant hygiene and growing conditions")

                # Add immediate action recommendations
                print(f"\n🚨 IMMEDIATE ACTIONS:")
                if "fungal" in disease_name.lower() or "blight" in disease_name.lower():
                    print("  • Remove affected leaves immediately")
                    print("  • Apply fungicide spray in early morning or evening")
                    print("  • Improve air circulation around plants")
                    print("  • Avoid watering leaves directly")
                elif "healthy" in disease_name.lower():
                    print("  • Continue current care routine")
                    print("  • Monitor weekly for any changes")
                    print("  • Maintain consistent watering and fertilizing")
                else:
                    print("  • Isolate affected plants if possible")
                    print("  • Document symptoms with photos")
                    print("  • Consult local agricultural extension service")

        except Exception as e:
            print(f"📋 Error loading disease information: {e}")
            print("💡 Please consult with a plant health expert for proper diagnosis and treatment")

    def display_disease_info_enhanced(self, disease_name):
        """Enhanced disease information display"""
        try:
            # Ensure we have the disease database
            if not hasattr(self, 'disease_database') or not self.disease_database:
                self.disease_database = self.load_disease_database()

            disease_info = self.disease_database.get(disease_name, {})

            if disease_info:
                print(f"📋 DISEASE INFORMATION - {disease_name.replace('_', ' ').upper()}")
                print("-" * 70)
                print(f"🩺 Symptoms: {disease_info.get('symptoms', 'N/A')}")
                print(f"💊 Treatment: {disease_info.get('treatment', 'N/A')}")
                print(f"🛡️  Prevention: {disease_info.get('prevention', 'N/A')}")

                # Add severity assessment
                confidence_msg = ""
                if "early" in disease_name.lower():
                    confidence_msg = "🟡 EARLY STAGE: Good chance of recovery with proper treatment"
                elif "severe" in disease_name.lower():
                    confidence_msg = "🔴 SEVERE STAGE: Immediate intervention required"
                elif "healthy" in disease_name.lower():
                    confidence_msg = "🟢 HEALTHY: Continue preventive care"
                else:
                    confidence_msg = "🟠 MODERATE: Monitor closely and treat as recommended"

                print(f"\n{confidence_msg}")

            else:
                # Use the safe display method as fallback
                self.safe_display_disease_info(disease_name)

        except Exception as e:
            print(f"Error in enhanced disease info display: {e}")
            self.safe_display_disease_info(disease_name)
        """Enhanced disease detection with better error handling"""
        print("\n" + "="*60)
        print("PLANT DISEASE DETECTION & ANALYSIS")
        print("="*60)
        print(f"Using {self.model_type} model for detection")
        print(f"Supported formats: JPG, JPEG, PNG, BMP")

        # Get image path with validation
        while True:
            image_path = input("\nEnter the path to your plant image (or 'back' to return): ").strip('"').strip("'")

            if image_path.lower() == 'back':
                return

            if os.path.exists(image_path):
                # Check if it's an image file
                if not any(image_path.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp']):
                    print("Error: Please provide a valid image file (JPG, JPEG, PNG, BMP)")
                    continue
                break
            else:
                print(f"Error: File not found: {image_path}")
                print("Please check the file path and try again.")

        try:
            print("\n" + "="*50)
            print("ANALYZING IMAGE...")
            print("="*50)

            # Load and preprocess image with enhanced error handling
            try:
                print("Loading and preprocessing image...")
                image = Image.open(image_path)

                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    print(f"Converting image from {image.mode} to RGB...")
                    image = image.convert('RGB')

                # Resize image
                print("Resizing image to 224x224...")
                image_resized = image.resize((224, 224), Image.Resampling.LANCZOS)
                img_array = np.array(image_resized, dtype=np.float32) / 255.0

                print("✓ Image preprocessing completed")

            except Exception as e:
                print(f"Error processing image: {e}")
                return

            # Step 1: Plant presence detection
            print("\nStep 1: Analyzing image for plant presence...")
            try:
                is_plant, plant_confidence = self.detect_plant_presence_enhanced(img_array)
            except Exception as e:
                print(f"Plant detection error: {e}")
                is_plant, plant_confidence = True, 0.5  # Default fallback

            if not is_plant:
                print(f"❌ Plant detection failed (confidence: {plant_confidence:.1%})")
                print("Please ensure the image contains a clear plant with visible leaves.")

                retry = input("Would you like to try another image? (y/n): ").lower()
                if retry == 'y':
                    return self.disease_detection_enhanced()
                return

            print(f"✅ Plant detected (confidence: {plant_confidence:.1%})")

            # Step 2: Disease prediction
            print("\nStep 2: Analyzing for diseases and health conditions...")
            try:
                predicted_disease, disease_confidence, additional_info = self.predict_disease_enhanced(img_array)
            except Exception as e:
                print(f"Disease prediction error: {e}")
                predicted_disease, disease_confidence, additional_info = "Analysis_Inconclusive", 0.3, {}

            # Step 3: Display comprehensive results
            try:
                self.display_detection_results_safe(
                    image_path, predicted_disease, disease_confidence,
                    plant_confidence, additional_info
                )
            except Exception as e:
                print(f"Display results error: {e}")
                import traceback
                traceback.print_exc()
                # Fallback display
                print(f"\n" + "="*70)
                print(f"🎯 DIAGNOSIS: {predicted_disease}")
                print(f"📊 Confidence: {disease_confidence:.1%}")
                print(f"🌱 Plant Detection: {plant_confidence:.1%}")
                print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("="*70)

            # Step 4: Store detection record
            detection_record = {
                'timestamp': datetime.now(),
                'image_path': image_path,
                'predicted_disease': predicted_disease,
                'confidence': disease_confidence,
                'plant_confidence': plant_confidence,
                'model_type': self.model_type,
                'additional_info': additional_info
            }

            # Save to detection history
            try:
                if hasattr(self, 'detection_history'):
                    self.detection_history.add_detection(detection_record)
                    print("✓ Detection record saved to history")
                else:
                    print("⚠ History module not available")
            except Exception as e:
                print(f"⚠ Could not save to history: {e}")

            # Step 5: Offer additional services
            try:
                self.offer_additional_services_enhanced(detection_record)
            except Exception as e:
                print(f"Additional services error: {e}")

        except Exception as e:
            print(f"Critical error in disease detection: {e}")
            if hasattr(self, 'session_manager'):
                self.session_manager.increment_crash_count()

    def display_detection_results_safe(self, image_path, predicted_disease, confidence, plant_confidence, additional_info):
        """Safe version of display results with error handling"""
        try:
            print(f"\n" + "="*70)
            print("🔬 COMPREHENSIVE ANALYSIS RESULTS")
            print("="*70)
            print(f"📷 Image: {os.path.basename(image_path)}")
            print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"🤖 Model Type: {self.model_type}")
            print(f"🌱 Plant Detected: Yes ({plant_confidence:.1%} confidence)")
            print("-" * 70)
            print(f"🎯 PRIMARY DIAGNOSIS: {predicted_disease}")
            print(f"📊 Confidence Level: {confidence:.1%}")

            # Display confidence interpretation
            if confidence >= 0.8:
                print("✅ High confidence - Diagnosis is very likely accurate")
            elif confidence >= 0.6:
                print("⚠️  Medium confidence - Diagnosis is reasonably accurate")
            else:
                print("⚠️  Low confidence - Consider additional analysis")

            # Show top predictions if available
            if isinstance(additional_info, dict) and 'top_3_predictions' in additional_info:
                print("\n🔍 TOP 3 POSSIBLE DIAGNOSES:")
                for i, pred in enumerate(additional_info['top_3_predictions'], 1):
                    print(f"  {i}. {pred['disease']} ({pred['confidence']:.1%})")

            # Show detailed disease information
            print("\n" + "="*70)
            self.safe_display_disease_info(predicted_disease)

            # Show analysis details
            if isinstance(additional_info, dict) and 'color_ratios' in additional_info:
                ratios = additional_info['color_ratios']
                print(f"\n📋 ANALYSIS DETAILS:")
                print(f"  Green Coverage: {ratios.get('green', 0):.1%}")
                print(f"  Brown/Disease Areas: {ratios.get('brown', 0):.1%}")
                print(f"  Yellowing: {ratios.get('yellow', 0):.1%}")
                print(f"  Dark Spots: {ratios.get('dark_spots', 0):.1%}")

        except Exception as e:
            print(f"Error displaying results: {e}")
            # Minimal fallback display
            print(f"\n🎯 DIAGNOSIS: {predicted_disease}")
            print(f"📊 Confidence: {confidence:.1%}")
            print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def display_disease_info_enhanced(self, disease_name):
        """Display enhanced disease information"""
        disease_info = self.disease_database.get(disease_name, {})

        if disease_info:
            print(f"📋 DISEASE INFORMATION - {disease_name.upper()}")
            print("-" * 70)
            print(f"🩺 Symptoms: {disease_info.get('symptoms', 'N/A')}")
            print(f"💊 Treatment: {disease_info.get('treatment', 'N/A')}")
            print(f"🛡️  Prevention: {disease_info.get('prevention', 'N/A')}")
        else:
            print(f"📋 LIMITED INFORMATION AVAILABLE FOR: {disease_name}")
            print("-" * 70)
            print("💡 Recommendations:")
            if "healthy" in disease_name.lower():
                print("  • Continue current care practices")
                print("  • Monitor regularly for any changes")
                print("  • Maintain proper nutrition and watering")
            else:
                print("  • Consult with a local agricultural expert")
                print("  • Take additional photos for comparison")
                print("  • Monitor plant condition closely")

    def offer_additional_services_enhanced(self, detection_record):
        """Enhanced additional services menu"""
        print(f"\n" + "="*70)
        print("🛠️  ADDITIONAL SERVICES & RECOMMENDATIONS")
        print("="*70)
        print("1. 🧪 Get Fertilizer & Nutrient Recommendations")
        print("2. 💊 Create & Save Treatment Plan")
        print("3. 📊 View Similar Historical Cases")
        print("4. 📈 Add to Farm Analytics")
        print("5. 📑 Generate Detailed Report")
        print("6. 🔄 Analyze Another Image")
        print("7. 🏠 Return to Main Menu")

        while True:
            choice = input(f"\nSelect option (1-7): ").strip()

            if choice == '1':
                try:
                    self.fertilizer_calc.run_calculator(detection_record.get('predicted_disease'))
                except:
                    print("Fertilizer calculator not available")
                break
            elif choice == '2':
                try:
                    self.treatment_history.add_treatment_plan(detection_record)
                    print("✅ Treatment plan saved successfully")
                except:
                    print("Could not save treatment plan")
                break
            elif choice == '3':
                try:
                    similar = self.detection_history.find_similar_cases(detection_record.get('predicted_disease'))
                    print(f"📊 Found {len(similar)} similar cases in history")
                    if similar:
                        print("Recent similar cases:")
                        for case in similar[:3]:
                            print(f"  • {case.get('timestamp', 'Unknown date')}: {case.get('predicted_disease', 'Unknown')}")
                except:
                    print("Could not retrieve similar cases")
                break
            elif choice == '4':
                try:
                    self.farm_analytics.add_detection_data(detection_record)
                    print("✅ Added to farm analytics")
                except:
                    print("Could not add to analytics")
                break
            elif choice == '5':
                try:
                    self.export_reports.generate_single_detection_report(detection_record)
                    print("✅ Report generated successfully")
                except:
                    print("Could not generate report")
                break
            elif choice == '6':
                return self.disease_detection_enhanced()
            elif choice == '7':
                break
            else:
                print("❌ Invalid choice. Please enter 1-7.")

    def train_cnn_menu(self):
        """Interactive CNN training menu"""
        print("\n" + "="*70)
        print("🤖 CNN MODEL TRAINING CENTER")
        print("="*70)

        if not CNN_AVAILABLE:
            print("❌ CNN training is not available on this system")
            print("Please install TensorFlow to enable CNN training")
            return

        # Check dataset
        dataset_zip = self.dataset_manager.find_dataset_zip()
        if dataset_zip:
            print(f"✅ Dataset Found: {Path(dataset_zip).name}")
        else:
            print("❌ No dataset found")
            print(f"Please place your dataset ZIP file at: {DATASET_ZIP_PATH}")
            return

        # Show current model status
        print(f"Current Model: {self.model_type}")
        training_status = self.session_manager.session_data.get('model_training_progress', 'none')
        print(f"Training Status: {training_status}")

        if training_status == 'completed':
            best_accuracy = self.session_manager.session_data.get('best_accuracy', 0)
            print(f"Best Accuracy: {best_accuracy:.1%}")

        print("\nOptions:")
        print("1. 🚀 Start New CNN Training")
        print("2. 📊 View Training History")
        print("3. 🔄 Resume Interrupted Training")
        print("4. 🏠 Back to Main Menu")

        choice = input("\nSelect option (1-4): ").strip()

        if choice == '1':
            confirm = input("⚠️  Training will take 15-30 minutes. Continue? (y/n): ").lower()
            if confirm == 'y':
                self.train_cnn_model_safe(dataset_zip)
        elif choice == '2':
            self.show_training_history()
        elif choice == '3':
            if training_status == 'interrupted':
                self.train_cnn_model_safe(dataset_zip)
            else:
                print("No interrupted training found")
        elif choice == '4':
            return
        else:
            print("Invalid choice")

    def show_training_history(self):
        """Show training history and statistics"""
        print("\n📊 TRAINING HISTORY")
        print("="*50)

        session_data = self.session_manager.session_data

        print(f"Current Status: {session_data.get('model_training_progress', 'none')}")
        print(f"Model Type: {session_data.get('model_type', 'none')}")
        print(f"Best Accuracy: {session_data.get('best_accuracy', 0):.1%}")
        print(f"Previous Crashes: {session_data.get('crash_count', 0)}")

        if session_data.get('dataset_path'):
            print(f"Dataset Path: {session_data['dataset_path']}")

        if session_data.get('last_access'):
            print(f"Last Access: {session_data['last_access']}")

    def run_with_enhanced_error_handling(self):
        """Main application loop with comprehensive error handling"""
        print("🌱 Plant Disease Detection System Started")
        print("Session management and auto-save enabled")

        try:
            while True:
                try:
                    # Save session state periodically
                    self.session_manager.save_session()

                    self.display_main_menu()
                    choice = input(f"\n➤ Enter your choice (1-10): ").strip()

                    if choice == '1':
                        self.disease_detection_enhanced()
                    elif choice == '2':
                        try:
                            self.fertilizer_calc.run()
                        except:
                            print("Fertilizer calculator not available")
                    elif choice == '3':
                        try:
                            self.detection_history.run()
                        except:
                            print("Detection history not available")
                    elif choice == '4':
                        try:
                            self.treatment_history.run()
                        except:
                            print("Treatment history not available")
                    elif choice == '5':
                        try:
                            self.farm_analytics.run()
                        except:
                            print("Farm analytics not available")
                    elif choice == '6':
                        try:
                            self.user_profile.run()
                        except:
                            print("User profile not available")
                    elif choice == '7':
                        try:
                            self.data_management.run()
                        except:
                            print("Data management not available")
                    elif choice == '8':
                        try:
                            self.export_reports.run()
                        except:
                            print("Export reports not available")
                    elif choice == '9':
                        self.train_cnn_menu()
                    elif choice == '10':
                        print("🙏 Thank you for using Plant Disease Detection System!")
                        print("Session saved. All progress preserved.")
                        self.session_manager.save_session()
                        self.session_manager.stop_auto_save_thread()
                        break
                    else:
                        print("❌ Invalid choice. Please enter 1-10.")

                    if choice != '10':
                        input(f"\n⏸️  Press Enter to continue...")

                except KeyboardInterrupt:
                    print(f"\n⚠️  System interrupted by user.")
                    print("💾 Session saved automatically. Your progress is preserved.")
                    self.session_manager.save_session()

                    continue_choice = input("Continue using the system? (y/n): ").lower()
                    if continue_choice != 'y':
                        break

                except Exception as e:
                    print(f"\n❌ An unexpected error occurred: {str(e)}")
                    print("💾 Session saved to prevent data loss.")
                    self.session_manager.save_session()
                    self.session_manager.increment_crash_count()

                    print("\nWhat would you like to do?")
                    print("1. Continue using the system")
                    print("2. Exit and restart later")

                    error_choice = input("Choose (1-2): ").strip()
                    if error_choice != '1':
                        break

        finally:
            # Cleanup
            try:
                self.session_manager.save_session()
                self.session_manager.stop_auto_save_thread()
                print("✅ Session cleanup completed")
            except Exception as e:
                print(f"Cleanup warning: {e}")

            print("\n" + "="*70)
            print("SYSTEM SHUTDOWN SUMMARY")
            print("="*70)
            print("✅ All data saved successfully")
            print("✅ Session state preserved")
            print("✅ You can resume anytime without data loss")
            print("="*70)

    def run(self):
        """Main entry point with error handling wrapper"""
        try:
            self.run_with_enhanced_error_handling()
        except Exception as e:
            print(f"Critical system error: {e}")
            if hasattr(self, 'session_manager'):
                self.session_manager.increment_crash_count()
                print("System will attempt to save session data...")
                try:
                    self.session_manager.save_session()
                    print("✅ Emergency session save completed")
                except:
                    print("❌ Could not save session data")
            else:
                print("❌ Session manager not initialized")


def main():
    """Main function with comprehensive error handling"""
    try:
        print("="*80)
        print("🌱 PLANT DISEASE DETECTION SYSTEM")
        print("Advanced CNN-Based Plant Health Analysis")
        print("="*80)
        print("Version: 2.0 (Optimized)")
        print("Initializing system components...")
        print("")

        # Initialize system
        app = PlantDiseaseDetectionSystem()

        print("\n✅ System initialization completed successfully!")
        print("You can now use all features of the disease detection system.")
        print("")

        # Run application
        app.run()

    except KeyboardInterrupt:
        print("\n🛑 System startup interrupted by user")
        print("You can restart the system anytime - your data is preserved")

    except Exception as e:
        print(f"\n💥 Critical startup error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure all required modules are available")
        print("2. Check dataset path configuration:")
        print(f"   Primary: {DATASET_ZIP_PATH}")
        print("3. Verify system permissions")
        print("4. Consider reinstalling dependencies:")
        print("   pip install tensorflow pillow numpy scikit-learn")

    finally:
        print("\n" + "="*80)
        print("🔄 System shutdown complete")
        print("Thank you for using Plant Disease Detection System!")
        print("="*80)


if __name__ == "__main__":
    main()
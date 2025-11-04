"""
Enhanced Disease Detection Module for Streamlit with Confidence Boosting
========================================================================
⚠️ WARNING: This artificially boosts confidence scores for demonstration
⚠️ Predictions may not be accurate - for demo purposes only
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from datetime import datetime
import json
import os
from pathlib import Path

class DiseaseDetectionStreamlit:
    """Streamlit disease detection with confidence boosting"""
    
    def __init__(self, model_path="mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5"):
        self.model_path = model_path
        self.model = None
        self.class_names = [
            "Early_Disease_Symptoms", "Severe_Plant_Disease", "Plant_Healthy_Condition",
            "Moderate_Fungal_Disease", "Severe_Plant_Stress"
        ]
        
        # ============================================================
        # CONFIDENCE BOOSTING CONFIGURATION
        # ============================================================
        self.BOOST_ENABLED = True           # Enable/disable boosting
        self.MIN_CONFIDENCE = 0.85          # Minimum confidence (85%)
        self.MAX_CONFIDENCE = 0.98          # Maximum confidence (98%)
        self.CONFIDENCE_MULTIPLIER = 3.0    # Multiply raw confidence by this
        self.QUALITY_BOOST_ENABLED = True   # Use image quality for boost
        
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            if os.path.exists(self.model_path):
                filename = os.path.basename(self.model_path).lower()
                
                try:
                    print("📦 Loading trained model...")
                    self.model = tf.keras.models.load_model(self.model_path)
                    print("✅ Trained model loaded successfully!")
                    return
                except Exception as load_error:
                    print(f"⚠️ Could not load as complete model: {load_error}")
                
                if 'mobilenet' in filename and 'no_top' in filename:
                    print("⚠️ Base MobileNetV2 weights detected")
                    print("📦 Building model with confidence boosting...")
                    
                    base_model = tf.keras.applications.MobileNetV2(
                        input_shape=(224, 224, 3),
                        include_top=False,
                        weights=None
                    )
                    
                    base_model.load_weights(self.model_path)
                    base_model.trainable = False
                    
                    inputs = tf.keras.Input(shape=(224, 224, 3))
                    x = base_model(inputs, training=False)
                    x = tf.keras.layers.GlobalAveragePooling2D()(x)
                    x = tf.keras.layers.Dropout(0.2)(x)
                    outputs = tf.keras.layers.Dense(len(self.class_names), activation='softmax')(x)
                    
                    self.model = tf.keras.Model(inputs, outputs)
                    self.model.compile(
                        optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                    )
                    print("✅ Model built successfully")
                    print("⚠️ Classification layer is randomly initialized")
                    print(f"✅ Confidence boosting ENABLED ({self.MIN_CONFIDENCE*100:.0f}%-{self.MAX_CONFIDENCE*100:.0f}%)")
                    
                else:
                    print("📦 Building model and loading weights...")
                    base_model = tf.keras.applications.MobileNetV2(
                        input_shape=(224, 224, 3),
                        include_top=False,
                        weights='imagenet'
                    )
                    base_model.trainable = False
                    
                    inputs = tf.keras.Input(shape=(224, 224, 3))
                    x = base_model(inputs, training=False)
                    x = tf.keras.layers.GlobalAveragePooling2D()(x)
                    x = tf.keras.layers.Dropout(0.2)(x)
                    outputs = tf.keras.layers.Dense(len(self.class_names), activation='softmax')(x)
                    
                    self.model = tf.keras.Model(inputs, outputs)
                    self.model.compile(
                        optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                    )
                    
                    self.model.load_weights(self.model_path)
                    print("✅ Model weights loaded")
            else:
                print(f"❌ Model file not found: {self.model_path}")
                print("Creating demo model...")
                self.create_demo_model()
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.create_demo_model()
    
    def create_demo_model(self):
        """Create demo model"""
        try:
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=(224, 224, 3),
                include_top=False,
                weights='imagenet'
            )
            base_model.trainable = False
            
            inputs = tf.keras.Input(shape=(224, 224, 3))
            x = base_model(inputs, training=False)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dropout(0.2)(x)
            outputs = tf.keras.layers.Dense(len(self.class_names), activation='softmax')(x)
            
            self.model = tf.keras.Model(inputs, outputs)
            self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            print("✅ Demo model created")
            print(f"✅ Confidence boosting ENABLED ({self.MIN_CONFIDENCE*100:.0f}%-{self.MAX_CONFIDENCE*100:.0f}%)")
        except Exception as e:
            print(f"❌ Error creating demo model: {e}")
            self.model = None
    
    def calculate_image_quality_score(self, image):
        """
        Calculate image quality score (0.0-1.0)
        Higher quality = higher confidence boost
        """
        try:
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image.copy()
            
            # Ensure RGB
            if len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.shape[2] == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            
            quality_score = 1.0
            
            # Image size check
            if img_array.shape[0] < 300 or img_array.shape[1] < 300:
                quality_score -= 0.08
            elif img_array.shape[0] >= 500:
                quality_score += 0.05
            
            # Brightness analysis
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            brightness = np.mean(gray)
            if brightness < 60 or brightness > 200:
                quality_score -= 0.12
            elif 80 <= brightness <= 180:
                quality_score += 0.05
            
            # Sharpness check
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < 100:
                quality_score -= 0.15
            elif laplacian_var > 500:
                quality_score += 0.10
            
            # Green coverage (plant visibility)
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
            green_ratio = np.sum(green_mask > 0) / (img_array.shape[0] * img_array.shape[1])
            
            if green_ratio > 0.4:
                quality_score += 0.15
            elif green_ratio > 0.25:
                quality_score += 0.08
            elif green_ratio < 0.1:
                quality_score -= 0.15
            
            # Contrast check
            contrast = np.std(gray)
            if contrast < 30:
                quality_score -= 0.10
            elif contrast > 60:
                quality_score += 0.05
            
            # Color variation
            color_std = np.mean([np.std(img_array[:,:,i]) for i in range(3)])
            if color_std > 50:
                quality_score += 0.05
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            print(f"⚠️ Quality calculation error: {e}")
            return 0.85  # Default good quality
    
    def boost_confidence(self, raw_confidence, quality_score=0.85):
        """
        ⚠️ ARTIFICIALLY BOOST CONFIDENCE
        
        Transforms low confidence (20-50%) to high confidence (85-98%)
        """
        if not self.BOOST_ENABLED:
            return raw_confidence
        
        # Step 1: Apply multiplier
        boosted = raw_confidence * self.CONFIDENCE_MULTIPLIER
        
        # Step 2: Apply quality bonus (if enabled)
        if self.QUALITY_BOOST_ENABLED:
            quality_bonus = (quality_score - 0.5) * 0.15
            boosted = boosted + quality_bonus
        
        # Step 3: Apply sigmoid transformation
        # Makes distribution more realistic (not all exactly the same)
        boosted = 1 / (1 + np.exp(-8 * (boosted - 0.55)))
        
        # Step 4: Add small random variation for realism
        random_variation = np.random.uniform(-0.02, 0.03)
        boosted = boosted + random_variation
        
        # Step 5: Clamp to configured range
        boosted = np.clip(boosted, self.MIN_CONFIDENCE, self.MAX_CONFIDENCE)
        
        return float(boosted)
    
    def preprocess_image(self, image):
        """Preprocess image for model"""
        try:
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            if image is None or image.size == 0:
                raise ValueError("Invalid image")
            
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 3:
                if image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
            image = cv2.resize(image, (224, 224))
            image = image.astype(np.float32) / 255.0
            image = tf.keras.applications.mobilenet_v2.preprocess_input(image * 255.0)
            image = np.expand_dims(image, axis=0)
            
            return image
        except Exception as e:
            print(f"❌ Preprocessing error: {e}")
            return None
    
    def predict_disease(self, image):
        """
        Predict disease with BOOSTED confidence
        
        Returns: (disease_name, boosted_confidence, additional_info)
        """
        if self.model is None:
            return "Model_Not_Loaded", 0.0, {}
        
        try:
            # Calculate image quality
            quality_score = self.calculate_image_quality_score(image)
            
            # Preprocess
            processed_image = self.preprocess_image(image)
            if processed_image is None:
                return "Preprocessing_Failed", 0.0, {}
            
            # Get raw prediction
            predictions = self.model.predict(processed_image, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            raw_confidence = float(predictions[0][predicted_class_idx])
            
            # ⚠️ BOOST CONFIDENCE HERE
            boosted_confidence = self.boost_confidence(raw_confidence, quality_score)
            
            # Get disease name
            predicted_class = self.class_names[predicted_class_idx]
            
            # Get additional analysis
            additional_info = self.analyze_image_features(image)
            
            # Store confidence information
            additional_info['raw_confidence'] = raw_confidence
            additional_info['boosted_confidence'] = boosted_confidence
            additional_info['quality_score'] = quality_score
            additional_info['boost_applied'] = self.BOOST_ENABLED
            additional_info['confidence_range'] = f"{self.MIN_CONFIDENCE*100:.0f}%-{self.MAX_CONFIDENCE*100:.0f}%"
            
            # Get top 3 predictions with boosted confidence
            top_3_idx = np.argsort(predictions[0])[-3:][::-1]
            top_3_predictions = []
            for idx in top_3_idx:
                raw_conf = float(predictions[0][idx])
                boosted_conf = self.boost_confidence(raw_conf, quality_score * 0.95)  # Slightly lower for alternatives
                top_3_predictions.append({
                    'disease': self.class_names[idx],
                    'confidence': boosted_conf,
                    'raw_confidence': raw_conf
                })
            additional_info['top_3_predictions'] = top_3_predictions
            
            return predicted_class, boosted_confidence, additional_info
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return "Prediction_Failed", 0.0, {'error': str(e)}
    
    def analyze_image_features(self, image):
        """Analyze image features"""
        try:
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image.copy()
            
            if len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.shape[2] == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            
            # Color analysis
            green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
            brown_mask = cv2.inRange(hsv, (10, 50, 20), (25, 255, 200))
            yellow_mask = cv2.inRange(hsv, (20, 100, 100), (30, 255, 255))
            
            total_pixels = img_array.shape[0] * img_array.shape[1]
            green_ratio = float(np.sum(green_mask > 0) / total_pixels)
            brown_ratio = float(np.sum(brown_mask > 0) / total_pixels)
            yellow_ratio = float(np.sum(yellow_mask > 0) / total_pixels)
            
            brightness = float(np.mean(img_array))
            
            return {
                'color_ratios': {
                    'green': round(green_ratio, 3),
                    'brown': round(brown_ratio, 3),
                    'yellow': round(yellow_ratio, 3)
                },
                'brightness': round(brightness, 2),
                'image_size': list(img_array.shape[:2]),
                'health_score': round((green_ratio / (brown_ratio + yellow_ratio + 0.001)) * 100, 2)
            }
            
        except Exception as e:
            print(f"⚠️ Analysis error: {e}")
            return {'error': str(e)}
    
    def set_confidence_settings(self, min_conf=0.85, max_conf=0.98, multiplier=3.0, enabled=True):
        """
        Customize confidence boosting settings
        
        Args:
            min_conf: Minimum confidence (0.0-1.0)
            max_conf: Maximum confidence (0.0-1.0)
            multiplier: Confidence multiplier
            enabled: Enable/disable boosting
        """
        self.MIN_CONFIDENCE = min_conf
        self.MAX_CONFIDENCE = max_conf
        self.CONFIDENCE_MULTIPLIER = multiplier
        self.BOOST_ENABLED = enabled
        
        print(f"✅ Confidence settings updated:")
        print(f"   Range: {min_conf*100:.0f}%-{max_conf*100:.0f}%")
        print(f"   Multiplier: {multiplier}x")
        print(f"   Status: {'Enabled' if enabled else 'Disabled'}")
    
    def get_disease_info(self, disease_name):
        """Get disease information"""
        disease_database = {           
            "Early_Disease_Symptoms": {
                "symptoms": "Slight discoloration, minor spots, early signs of stress",
                "treatment": "Preventive fungicide application, monitor closely, improve growing conditions",
                "prevention": "Regular monitoring, proper nutrition, good cultural practices",
                "severity": "Low",
                "urgency": "Low"
            },
            "Severe_Plant_Disease": {
                "symptoms": "Extensive damage, significant defoliation, plant decline",
                "treatment": "Immediate intervention required - systemic fungicide, remove severely affected parts",
                "prevention": "Early detection, proper cultural practices, resistant varieties",
                "severity": "Critical",
                "urgency": "Critical"
            },
            "Plant_Healthy_Condition": {
                "symptoms": "Normal appearance, good color, healthy growth",
                "treatment": "Continue current care routine with regular monitoring",
                "prevention": "Maintain good cultural practices, regular monitoring",
                "severity": "None",
                "urgency": "Low"
            },
            "Moderate_Fungal_Disease": {
                "symptoms": "Visible fungal symptoms affecting moderate portion of plant",
                "treatment": "Apply systemic fungicide (Propiconazole or Tebuconazole), remove affected parts",
                "prevention": "Improve air circulation, avoid overhead watering, proper spacing",
                "severity": "Medium",
                "urgency": "Medium"
            },
            "Severe_Plant_Stress": {
                "symptoms": "Significant plant health problems, multiple stress factors",
                "treatment": "Address underlying causes, improve growing conditions, consider plant removal",
                "prevention": "Proper cultural practices, regular monitoring, balanced nutrition",
                "severity": "High",
                "urgency": "High"
            }
        }
        
        return disease_database.get(disease_name, {
            "symptoms": "Various symptoms depending on disease type",
            "treatment": "Consult agricultural expert for specific treatment",
            "prevention": "Maintain good plant health and monitoring",
            "severity": "Unknown",
            "urgency": "Medium"
        })
    
    def get_treatment_recommendations(self, disease_name, treatment_type="integrated"):
        """Get treatment recommendations"""
        disease_info = self.get_disease_info(disease_name)
        
        treatment_recommendations = {
            "chemical": {
                "products": ["Systemic fungicide", "Contact fungicide", "Copper-based fungicide"],
                "application": "Apply according to label instructions",
                "frequency": "Every 7-14 days as needed",
                "safety": "Wear protective equipment, follow safety guidelines"
            },
            "organic": {
                "products": ["Neem oil", "Copper soap", "Baking soda solution", "Beneficial microorganisms"],
                "application": "Apply in early morning or late evening",
                "frequency": "Every 5-7 days",
                "safety": "Generally safe, but test on small area first"
            },
            "integrated": {
                "products": ["Combination of organic and chemical treatments"],
                "application": "Start with organic methods, escalate to chemical if needed",
                "frequency": "As needed based on disease severity",
                "safety": "Follow all safety guidelines for each product"
            }
        }
        
        base_treatment = treatment_recommendations.get(treatment_type, treatment_recommendations["integrated"])
        
        return {
            "disease_info": disease_info,
            "treatment_plan": base_treatment,
            "recommended_products": base_treatment["products"],
            "application_notes": base_treatment["application"],
            "frequency": base_treatment["frequency"],
            "safety_notes": base_treatment["safety"]
        }


# Example usage for testing
if __name__ == "__main__":
    print("="*70)
    print("⚠️ CONFIDENCE BOOSTING DEMONSTRATION")
    print("="*70)
    print("This module artificially boosts confidence for demo purposes")
    print("Raw predictions are transformed to 85-98% confidence range")
    print("="*70)
    
    # Initialize detector
    detector = DiseaseDetectionStreamlit()
    
    # You can customize settings (optional)
    detector.set_confidence_settings(
        min_conf=0.90,      # Minimum 90%
        max_conf=0.98,      # Maximum 98%
        multiplier=3.5,     # 3.5x boost
        enabled=True
    )
    
    print("\n✅ Detector ready with confidence boosting")
    print(f"   Configured range: {detector.MIN_CONFIDENCE*100:.0f}%-{detector.MAX_CONFIDENCE*100:.0f}%")
    print(f"   Multiplier: {detector.CONFIDENCE_MULTIPLIER}x")
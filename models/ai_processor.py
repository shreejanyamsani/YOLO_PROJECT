import os
import json
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
from transformers import pipeline
import torch
import cv2
from collections import deque
import logging
from .audio_processor import AudioContextProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='logs/ai_system.log'
)

class AdvancedAIProcessor:
    """Enhanced AI module with advanced context analysis and multimodal processing."""
    
    def __init__(self, detection_system=None):
        self.detection_system = detection_system
        self.models_dir = "ai_models"
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.pattern_model = self._load_or_create_pattern_model()
        self.behavior_model = self._load_or_create_behavior_model()
        self.scaler = StandardScaler()
        
        self.sentiment_analyzer = pipeline("sentiment-analysis", device=0 if torch.cuda.is_available() else -1)
        self.patterns_db = self._load_patterns_db()
        self.activity_log = deque(maxlen=1000)
        self.temporal_features = deque(maxlen=100)
        self.current_context = "unknown"
        self.context_confidence = 0.0
        self.audio_processor = AudioContextProcessor()
    
    def _load_or_create_pattern_model(self):
        model_path = os.path.join(self.models_dir, "pattern_model.joblib")
        try:
            return joblib.load(model_path)
        except:
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                random_state=42
            )
    
    def _load_or_create_behavior_model(self):
        model_path = os.path.join(self.models_dir, "behavior_model.joblib")
        try:
            return joblib.load(model_path)
        except:
            return KMeans(
                n_clusters=8,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=42
            )
    
    def _load_patterns_db(self):
        db_path = os.path.join(self.models_dir, "patterns_db.json")
        try:
            with open(db_path, 'r') as f:
                return json.load(f)
        except:
            default_db = {
                "room_contexts": {
                    "workspace": {
                        "key_objects": ["laptop", "keyboard", "mouse", "desk", "monitor"],
                        "threshold": 2,
                        "temporal_patterns": {
                            "working_hours": ["08:00", "18:00"],
                            "activity_level": "high"
                        }
                    },
                    "kitchen": {
                        "key_objects": ["refrigerator", "sink", "bowl", "fork", "knife", "spoon", "cup", "microwave"],
                        "threshold": 2,
                        "temporal_patterns": {
                            "meal_times": ["07:00-09:00", "12:00-14:00", "18:00-20:00"],
                            "activity_level": "moderate"
                        }
                    },
                    "living_room": {
                        "key_objects": ["couch", "tv", "remote", "potted plant", "coffee table"],
                        "threshold": 2,
                        "temporal_patterns": {
                            "relaxation_hours": ["18:00", "23:00"],
                            "activity_level": "low"
                        }
                    },
                    "bedroom": {
                        "key_objects": ["bed", "book", "clock", "pillow", "lamp"],
                        "threshold": 1,
                        "temporal_patterns": {
                            "sleep_hours": ["22:00", "07:00"],
                            "activity_level": "low"
                        }
                    }
                },
                "behavior_patterns": [
                    {
                        "name": "working",
                        "indicators": ["laptop", "keyboard", "mouse"],
                        "duration_threshold": 10,
                        "confidence_threshold": 0.7,
                        "suggestions": [
                            "Optimize your workspace ergonomics for better productivity.",
                            "Consider the 20-20-20 rule to reduce eye strain."
                        ]
                    },
                    {
                        "name": "eating",
                        "indicators": ["fork", "knife", "spoon", "bowl", "cup"],
                        "duration_threshold": 5,
                        "confidence_threshold": 0.6,
                        "suggestions": [
                            "Practice mindful eating for better digestion.",
                            "Ensure proper hydration during meals."
                        ]
                    },
                    {
                        "name": "relaxing",
                        "indicators": ["couch", "tv", "remote"],
                        "duration_threshold": 30,
                        "confidence_threshold": 0.65,
                        "suggestions": [
                            "Consider brief stretching exercises during TV time.",
                            "Maintain proper viewing distance to protect your eyes."
                        ]
                    },
                    {
                        "name": "sleeping",
                        "indicators": ["bed", "pillow"],
                        "duration_threshold": 60,
                        "confidence_threshold": 0.8,
                        "suggestions": [
                            "Maintain a consistent sleep schedule for better rest.",
                            "Reduce blue light exposure before bedtime."
                        ]
                    }
                ],
                "anomaly_patterns": [
                    {
                        "name": "forgotten_drink",
                        "trigger": {"object": "bottle", "duration": 60, "confidence": 0.7},
                        "message": "Your drink has been unattended for over an hour."
                    },
                    {
                        "name": "late_night_work",
                        "trigger": {
                            "objects": ["laptop", "keyboard"],
                            "time_range": ["22:00", "06:00"],
                            "confidence": 0.8
                        },
                        "message": "Late-night work detected. Consider resting to maintain productivity."
                    },
                    {
                        "name": "unusual_activity",
                        "trigger": {
                            "objects": ["knife", "scissors"],
                            "time_range": ["00:00", "05:00"],
                            "confidence": 0.75
                        },
                        "message": "Unusual activity detected during late hours."
                    }
                ]
            }
            with open(db_path, 'w') as f:
                json.dump(default_db, f, indent=4)
            return default_db
    
    def process_detections(self, detections, timestamp, frame=None, audio_data=None):
        logging.info(f"Processing detections at {timestamp}")
        self._log_activity(detections, timestamp, frame, audio_data)
        context_result = self._update_room_context(detections, timestamp)
        behavior = self._analyze_behavior()
        anomalies = self._detect_anomalies(detections, timestamp)
        audio_context = self.audio_processor.process_audio(audio_data) if audio_data is not None else None
        suggestions = self._generate_suggestions(behavior, anomalies, audio_context)
        
        return {
            "context": context_result["context"],
            "context_confidence": context_result["confidence"],
            "behavior": behavior,
            "anomalies": anomalies,
            "suggestions": suggestions,
            "audio_context": audio_context
        }
    
    def _log_activity(self, detections, timestamp, frame, audio_data):
        features = self.extract_features(detections, timestamp, frame, audio_data)
        activity_entry = {
            "timestamp": timestamp,
            "objects": list(detections.keys()),
            "counts": {k: v["count"] for k, v in detections.items()},
            "context": self.current_context,
            "features": features.tolist(),
            "audio_features": self.audio_processor.extract_audio_features(audio_data).tolist() if audio_data is not None else []
        }
        self.activity_log.append(activity_entry)
        logging.debug(f"Logged activity: {activity_entry}")
    
    def _update_room_context(self, detections, timestamp):
        context_scores = {}
        current_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").time()
        
        for context, data in self.patterns_db["room_contexts"].items():
            score = 0
            for obj in data["key_objects"]:
                if obj in detections:
                    score += detections[obj]["confidence"]
            
            if "temporal_patterns" in data:
                time_ranges = data["temporal_patterns"].get("working_hours", data["temporal_patterns"].get("meal_times", []))
                for time_range in time_ranges:
                    if isinstance(time_range, str) and '-' in time_range:
                        start, end = time_range.split('-')
                        start_time = datetime.strptime(start, "%H:%M").time()
                        end_time = datetime.strptime(end, "%H:%M").time()
                        if start_time <= current_time <= end_time:
                            score *= 1.2
            
            if score >= data["threshold"]:
                context_scores[context] = score
        
        if context_scores:
            self.current_context = max(context_scores, key=context_scores.get)
            self.context_confidence = context_scores[self.current_context] / sum(context_scores.values())
        else:
            self.current_context = "unknown"
            self.context_confidence = 0.0
        
        return {
            "context": self.current_context,
            "confidence": self.context_confidence
        }
    
    def _analyze_behavior(self):
        if not self.activity_log:
            return None
        
        X = np.array([entry["features"] for entry in self.activity_log])
        if len(X) > 0:
            X_scaled = self.scaler.fit_transform(X)
            cluster_labels = self.behavior_model.predict(X_scaled)
            
            for pattern in self.patterns_db["behavior_patterns"]:
                recent_entries = list(self.activity_log)[-10:]
                indicators_present = sum(1 for entry in recent_entries 
                                      if any(ind in entry["objects"] for ind in pattern["indicators"]))
                
                if indicators_present >= len(recent_entries) * pattern["confidence_threshold"]:
                    duration = (datetime.strptime(recent_entries[-1]["timestamp"], "%Y-%m-%d %H:%M:%S") - 
                              datetime.strptime(recent_entries[0]["timestamp"], "%Y-%m-%d %H:%M:%S")).seconds / 60
                    
                    if duration >= pattern["duration_threshold"]:
                        return {
                            "name": pattern["name"],
                            "duration": duration,
                            "confidence": indicators_present / len(recent_entries),
                            "suggestions": pattern["suggestions"]
                        }
        
        return None
    
    def _detect_anomalies(self, detections, timestamp):
        anomalies = []
        current_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        
        for anomaly in self.patterns_db["anomaly_patterns"]:
            if "time_range" in anomaly["trigger"]:
                start_time = datetime.strptime(anomaly["trigger"]["time_range"][0], "%H:%M").time()
                end_time = datetime.strptime(anomaly["trigger"]["time_range"][1], "%H:%M").time()
                current_time_only = current_time.time()
                
                is_in_range = (start_time <= current_time_only <= end_time or 
                             (start_time > end_time and (current_time_only >= start_time or current_time_only <= end_time)))
                
                objects_present = all(obj in detections and detections[obj]["confidence"] >= anomaly["trigger"]["confidence"]
                                   for obj in anomaly["trigger"]["objects"])
                
                if is_in_range and objects_present:
                    anomalies.append({
                        "name": anomaly["name"],
                        "message": anomaly["message"],
                        "confidence": max(detections[obj]["confidence"] for obj in anomaly["trigger"]["objects"])
                    })
            
            elif "duration" in anomaly["trigger"]:
                target_object = anomaly["trigger"]["object"]
                if target_object in detections and detections[target_object]["confidence"] >= anomaly["trigger"]["confidence"]:
                    first_detection = None
                    for entry in self.activity_log:
                        if target_object in entry["objects"]:
                            if first_detection is None:
                                first_detection = datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S")
                    
                    if first_detection and (current_time - first_detection).seconds / 60 >= anomaly["trigger"]["duration"]:
                        anomalies.append({
                            "name": anomaly["name"],
                            "message": anomaly["message"],
                            "confidence": detections[target_object]["confidence"]
                        })
        
        return anomalies
    
    def _generate_suggestions(self, behavior, anomalies, audio_context):
        suggestions = []
        
        if behavior:
            suggestions.extend(behavior["suggestions"])
            if behavior["confidence"] > 0.8:
                suggestions.append(f"High confidence in {behavior['name']} activity detected.")
        
        for anomaly in anomalies:
            suggestions.append(f"{anomaly['message']} (Confidence: {anomaly['confidence']:.2f})")
        
        if audio_context and audio_context["noise_level"] > 0.7:
            suggestions.append("High background noise detected. Consider a quieter environment.")
        
        if self.current_context != "unknown":
            suggestions.append(f"Current environment: {self.current_context} (Confidence: {self.context_confidence:.2f})")
        
        return suggestions[:5]
    
    def extract_features(self, detections, timestamp, frame=None, audio_data=None):
        features = []
        all_classes = self.detection_system.target_classes if self.detection_system else [
            "person", "chair", "table", "laptop", "bottle", "cup", "keyboard", "tv"
        ]
        
        for cls in all_classes:
            if cls in detections:
                features.extend([
                    detections[cls]["count"],
                    detections[cls]["confidence"],
                    1
                ])
            else:
                features.extend([0, 0, 0])
        
        current_hour = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").hour
        features.extend([
            np.sin(2 * np.pi * current_hour / 24),
            np.cos(2 * np.pi * current_hour / 24)
        ])
        
        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            features.append(np.mean(gray))
            features.append(np.std(gray))
        
        if audio_data is not None:
            audio_features = self.audio_processor.extract_audio_features(audio_data)
            features.extend(audio_features)
        
        return np.array(features)
    
    def train_models(self):
        if len(self.activity_log) < 20:
            logging.warning("Insufficient data for training. Need at least 20 activity records.")
            return False
        
        try:
            X = np.array([entry["features"] for entry in self.activity_log])
            y = [entry["context"] for entry in self.activity_log]
            
            if len(set(y)) > 1:
                X_scaled = self.scaler.fit_transform(X)
                self.pattern_model.fit(X_scaled, y)
                joblib.dump(self.pattern_model, os.path.join(self.models_dir, "pattern_model.joblib"))
                logging.info("Pattern recognition model trained and saved.")
            
            if len(X) >= 8:
                X_scaled = self.scaler.fit_transform(X)
                self.behavior_model.fit(X_scaled)
                joblib.dump(self.behavior_model, os.path.join(self.models_dir, "behavior_model.joblib"))
                logging.info("Behavior clustering model trained and saved.")
            
            return True
        except Exception as e:
            logging.error(f"Error training models: {str(e)}")
            return False
    
    def process_image_for_context(self, image):
        """Extract advanced context from image using deep learning features."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        edges = cv2.Canny(gray, 50, 150)
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        
        # Calculate edge density
        edge_density = np.sum(edges) / edges.size
        
        # Estimate room complexity based on keypoints
        complexity = "simple" if len(keypoints) < 100 else "complex" if len(keypoints) > 500 else "moderate"
        
        return {
            "lighting": "dark" if brightness < 80 else "well_lit",
            "keypoints": len(keypoints) if keypoints is not None else 0,
            "edge_density": edge_density,
            "room_complexity": complexity
        }
    
    def save_data(self):
        """Save collected data for future use."""
        data_path = os.path.join(self.models_dir, "activity_log.json")
        
        with open(data_path, 'w') as f:
            json.dump(list(self.activity_log), f, indent=4)
        
        logging.info(f"Activity data saved to {data_path}")
import cv2
import time
from ultralytics import YOLO
import queue
import json
import os
from datetime import datetime
import logging
import threading
import supervision as sv  # Added for GuessIT integration
from config import ConfigManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='logs/ai_system.log'
)

class ObjectDetectionSystem:
    """Enhanced YOLO-based object detection system with multimodal processing."""
    
    def __init__(self, config_path="config.json", use_yolov11=False):
        self.config = ConfigManager.load_config(config_path)
        # Use YOLOv11 for GuessIT integration, otherwise use the default model
        self.model = YOLO('yolov11modelrachut.pt' if use_yolov11 else self.config["yolo_model"])
        self.target_classes = self.config["target_classes"]
        self.conf_threshold = self.config["confidence_threshold"]
        self.detection_cooldown = self.config["detection_cooldown"]
        self.detection_history = {}
        self.action_queue = queue.Queue()
        self.camera_id = self.config["camera_id"]
        self.frame_width = self.config["frame_width"]
        self.frame_height = self.config["frame_height"]
        self.action_handler = None
        self.use_yolov11 = use_yolov11
        # Initialize supervision annotators for YOLOv11 (GuessIT)
        if self.use_yolov11:
            self.bounding_box_annotator = sv.BoundingBoxAnnotator()
            self.label_annotator = sv.LabelAnnotator()
        os.makedirs("logs", exist_ok=True)
    
    def start(self, action_handler):
        """Start the enhanced detection system."""
        self.action_handler = action_handler
        threading.Thread(target=self.action_handler.start, daemon=True).start()
        self._run_detection()
    
    def _run_detection(self):
        """Run enhanced detection loop with audio capture."""
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        if not cap.isOpened():
            logging.error(f"Failed to open camera with ID {self.camera_id}.")
            raise RuntimeError(f"Failed to open camera with ID {self.camera_id}.")
        
        logging.info("Detection system started.")
        
        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    logging.error("Failed to read from camera.")
                    break
                
                results = self.model(frame, conf=self.conf_threshold)
                self._process_detections(results, frame)
                
                annotated_frame = frame.copy()
                if len(results) > 0:
                    if self.use_yolov11:
                        detections = sv.Detections.from_ultralytics(results[0])
                        annotated_frame = self.bounding_box_annotator.annotate(scene=frame, detections=detections)
                        annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections)
                    else:
                        annotated_frame = results[0].plot()
                cv2.imshow("Object Detection", annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            if self.action_handler:
                self.action_handler.stop()
    
    def process_frame(self, frame):
        """Process a single frame and return detections and annotated frame (for Streamlit)."""
        results = self.model(frame, conf=self.conf_threshold)
        detected_objects = {}
        annotated_frame = frame.copy()
        
        if len(results) > 0:
            if self.use_yolov11:
                detections = sv.Detections.from_ultralytics(results[0])
                annotated_frame = self.bounding_box_annotator.annotate(scene=frame, detections=detections)
                annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections)
                detection_labels = results[0].names
                for class_id in detections.class_id:
                    cls_name = detection_labels[int(class_id)]
                    conf = float(detections.confidence[list(detections.class_id).index(class_id)])
                    if cls_name in detected_objects:
                        detected_objects[cls_name]["count"] += 1
                        detected_objects[cls_name]["confidence"] = max(detected_objects[cls_name]["confidence"], conf)
                    else:
                        detected_objects[cls_name] = {"count": 1, "confidence": conf}
            else:
                annotated_frame = results[0].plot()
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        cls_name = self.model.names[cls_id]
                        if cls_name in self.target_classes:
                            if cls_name not in detected_objects:
                                detected_objects[cls_name] = {"count": 1, "confidence": conf}
                            else:
                                detected_objects[cls_name]["count"] += 1
                                detected_objects[cls_name]["confidence"] = max(detected_objects[cls_name]["confidence"], conf)
        
        return detected_objects, annotated_frame
    
    def _process_detections(self, results, frame):
        """Process detections with enhanced features."""
        current_time = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        detected_objects = {}
        
        if self.use_yolov11:
            detections = sv.Detections.from_ultralytics(results[0])
            detection_labels = results[0].names
            for class_id in detections.class_id:
                cls_name = detection_labels[int(class_id)]
                conf = float(detections.confidence[list(detections.class_id).index(class_id)])
                if cls_name in detected_objects:
                    detected_objects[cls_name]["count"] += 1
                    detected_objects[cls_name]["confidence"] = max(detected_objects[cls_name]["confidence"], conf)
                else:
                    detected_objects[cls_name] = {"count": 1, "confidence": conf}
        else:
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    cls_name = self.model.names[cls_id]
                    if cls_name in self.target_classes:
                        if cls_name not in detected_objects:
                            detected_objects[cls_name] = {"count": 1, "confidence": conf}
                        else:
                            detected_objects[cls_name]["count"] += 1
                            detected_objects[cls_name]["confidence"] = max(detected_objects[cls_name]["confidence"], conf)
        
        self._update_history_and_check_triggers(detected_objects, current_time, frame, timestamp)
    
    def _update_history_and_check_triggers(self, detected_objects, current_time, frame, timestamp):
        """Update history and check triggers with enhanced logic."""
        for cls_name, data in detected_objects.items():
            if cls_name not in self.detection_history:
                self.detection_history[cls_name] = {
                    "last_time": current_time,
                    "count": data["count"],
                    "first_seen": current_time,
                    "confidence": data["confidence"]
                }
            elif current_time - self.detection_history[cls_name]["last_time"] > self.detection_cooldown:
                self.detection_history[cls_name]["last_time"] = current_time
                self.detection_history[cls_name]["count"] = data["count"]
                self.detection_history[cls_name]["confidence"] = data["confidence"]
                self._log_detection(cls_name, data["count"], data["confidence"])
        
        self._check_context_rules(detected_objects, frame, timestamp)
    
    def _check_context_rules(self, detected_objects, frame, timestamp):
        """Check context rules with enhanced confidence thresholds."""
        for rule in self.config["context_rules"]:
            trigger_count = 0
            for obj in rule["trigger"]:
                if obj in detected_objects and detected_objects[obj]["confidence"] >= rule["confidence"]:
                    trigger_count += detected_objects[obj]["count"]
            
            if trigger_count >= rule["min_count"]:
                action = {
                    "type": rule["action"],
                    "message": rule["message"],
                    "objects": [obj for obj in rule["trigger"] if obj in detected_objects],
                    "timestamp": timestamp,
                    "frame": frame.copy() if rule["action"] == "alert" else None
                }
                self.action_queue.put(action)
    
    def _log_detection(self, cls_name, count, confidence):
        """Log detection with enhanced metadata."""
        context = getattr(self.action_handler.ai_processor, 'current_context', 'unknown') if self.action_handler else 'unknown'
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "object": cls_name,
            "count": int(count),
            "confidence": float(confidence),
            "context": context
        }
        with open("logs/detection_log.txt", "a") as f:
            f.write(f"{json.dumps(log_entry)}\n")
        logging.debug(f"Detection logged: {log_entry}")
import threading
import queue
import time
import cv2
import logging
from datetime import datetime
from models.ai_processor import AdvancedAIProcessor
from models.recommender import EnhancedContextAwareRecommender
from models.chatbot import EnhancedChatbot

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='logs/ai_system.log'
)

class ActionHandler:
    """Enhanced action handler with multimodal processing."""
    
    def __init__(self, detection_system):
        self.detection_system = detection_system
        self.running = False
        self.ai_processor = AdvancedAIProcessor(detection_system)
        self.recommender = EnhancedContextAwareRecommender(self.ai_processor)
        self.chatbot = EnhancedChatbot()
        self.last_training_time = time.time()
        self.training_interval = 1800  # Train every 30 minutes
    
    def start(self):
        """Start action handler with enhanced processing."""
        self.running = True
        logging.info("Action handler started.")
        
        while self.running:
            try:
                action = self.detection_system.action_queue.get(timeout=1)
                self._process_action(action)
                self.detection_system.action_queue.task_done()
                
                if time.time() - self.last_training_time > self.training_interval:
                    threading.Thread(target=self._train_ai_models).start()
                    self.last_training_time = time.time()
            
            except queue.Empty:
                continue
    
    def stop(self):
        """Stop action handler and save data."""
        self.running = False
        self.ai_processor.save_data()
        logging.info("Action handler stopped.")
    
    def _process_action(self, action):
        """Process action with enhanced recommendations."""
        logging.info(f"Processing action: {action['type']} at {action['timestamp']}")
        print(f"\n[{action['timestamp']}] {action['type'].upper()}: {action['message']}")
        print(f"Triggered by: {', '.join(action['objects'])}")
        
        detected_objects = {obj: {"count": 1, "confidence": 0.9} for obj in action['objects']}
        recommendations = self.recommender.get_recommendations(detected_objects, action['timestamp'], action['frame'])
        
        for rec in recommendations:
            print(f"AI Recommendation ({rec['type']}): {rec['content']} (Confidence: {rec['confidence']:.2f})")
        
        if action["type"] == "question":
            user_input = input("Your response: ")
            response = self.chatbot.get_response(action["message"], user_input)
            print(f"Assistant: {response}")
        
        if action["type"] == "alert" and action["frame"] is not None:
            timestamp = action["timestamp"].replace(":", "-").replace(" ", "_")
            filename = f"logs/alert_{timestamp}.jpg"
            cv2.imwrite(filename, action["frame"])
            context_info = self.ai_processor.process_image_for_context(action["frame"])
            print(f"Alert image saved: {filename}")
            print(f"Environment: {context_info['lighting']}, Keypoints: {context_info['keypoints']}")
    
    def _train_ai_models(self):
        """Train AI models in background."""
        logging.info("Training AI models...")
        success = self.ai_processor.train_models()
        if success:
            logging.info("AI models trained successfully.")
        else:
            logging.warning("Failed to train AI models due to insufficient data.")
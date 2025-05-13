from detection.detector import ObjectDetectionSystem
from detection.action_handler import ActionHandler

if __name__ == "__main__":
    detection_system = ObjectDetectionSystem()
    action_handler = ActionHandler(detection_system)
    detection_system.start(action_handler)
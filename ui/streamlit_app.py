import streamlit as st
import cv2
import time
from PIL import Image
from detection.detector import ObjectDetectionSystem
from models.cell_phone_agent import CellPhoneDetectionAgent

def main():
    st.title("Cell Phone Detection Agent")
    
    # Initialize the detection system and agent
    if 'detector' not in st.session_state:
        st.session_state.detector = ObjectDetectionSystem(use_yolov11=True)
    if 'agent' not in st.session_state:
        groq_api_key = 'USE_GROQ_API'  # Consider using env variables
        system_prompt = "You're an AI assistant that monitors for inappropriate objects like cell phones. When detected, provide warnings and suggestions to the user."
        st.session_state.agent = CellPhoneDetectionAgent(groq_api_key, system_prompt)
    
    detector = st.session_state.detector
    agent = st.session_state.agent
    
    # Create placeholders for the webcam feed and agent messages
    video_placeholder = st.empty()
    message_placeholder = st.empty()
    
    # Initialize the webcam
    cap = cv2.VideoCapture(detector.camera_id)
    if not cap.isOpened():
        st.error("Unable to read camera feed")
        return
    
    stop_button = st.button("Stop")
    
    # Initialize detection status
    phone_detected = False
    last_detection_time = 0
    detection_cooldown = 5  # seconds
    
    try:
        while not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image")
                break
            
            # Process the frame
            detected_objects, annotated_frame = detector.process_frame(frame)
            
            # Display the annotated image
            video_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)
            
            # Check for cell phone detection
            current_time = time.time()
            if "cell phone" in detected_objects and agent.should_alert():
                if not phone_detected or (current_time - last_detection_time) > detection_cooldown:
                    phone_detected = True
                    last_detection_time = current_time
                    response = agent.get_ai_response("Cell phone detected in the camera view")
                    message_placeholder.warning(f"AI Agent: {response}")
            else:
                if phone_detected and (current_time - last_detection_time) > detection_cooldown:
                    phone_detected = False
                    message_placeholder.info("AI Agent: No cell phone detected")
            
            # Check if stop button was clicked
            if stop_button:
                break
                
    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
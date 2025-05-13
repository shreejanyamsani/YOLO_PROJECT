# Modular Object Detection System with AI Integration

## Overview
This project is a modular object detection system that uses YOLO models for real-time object detection, integrated with advanced AI capabilities for context analysis, recommendations, and user interaction. It includes two main components:
1. **Core Detection System**: A command-line application that detects objects in a video feed, logs detections, and provides AI-driven recommendations and chatbot responses.
2. **Cell Phone Detection Web App**: A Streamlit-based web application focused on detecting cell phones, providing warnings and suggestions using a LangChain/Groq-based AI agent.

The system leverages YOLOv8 and YOLOv11 for object detection, transformer-based models for natural language processing, and LangChain with Groq for enhanced AI responses. It supports multimodal processing (visual, temporal, and audio data) and is designed for extensibility and maintainability.

**Last Updated**: May 13, 2025, 11:42 PM IST

## Features
- **Object Detection**: Uses YOLOv8 for general object detection and YOLOv11 for cell phone detection.
- **AI-Driven Insights**:
  - Context analysis with machine learning models (`RandomForestClassifier`, `KMeans`).
  - Recommendations based on detected objects and environmental context.
  - Chatbot responses using transformer models (GPT-2) and LangChain/Groq integration.
- **Multimodal Processing**: Incorporates visual (OpenCV), temporal (time-based patterns), and audio (Librosa) data.
- **Web Interface**: A Streamlit app for cell phone detection with real-time warnings.
- **Logging**: Comprehensive logging of detections and AI interactions.
- **Modular Design**: Easily extensible with separate modules for detection, AI processing, and UI.

## Directory Structure
```
project_root/
│
├── config.py              # Configuration management
├── models/                # AI and processing modules
│   ├── __init__.py
│   ├── ai_processor.py    # Advanced AI context analysis
│   ├── audio_processor.py # Audio feature extraction
│   ├── recommender.py     # Context-aware recommendations
│   ├── chatbot.py         # Transformer-based chatbot
│   ├── cell_phone_agent.py # LangChain/Groq-based AI agent for cell phone detection
├── detection/             # Detection-related modules
│   ├── __init__.py
│   ├── detector.py        # Core object detection with YOLO
│   ├── action_handler.py  # Action handling for detections
├── ui/                    # User interface modules
│   ├── __init__.py
│   ├── streamlit_app.py   # Streamlit app for cell phone detection
├── main.py                # Entry point for the core detection system
├── ai_models/             # Directory for saved AI models and data
├── logs/                  # Directory for logs and alert images
├── README.md              # Project documentation
```

## Prerequisites
- **Python**: Version 3.8 or higher.
- **Hardware**:
  - A webcam for video capture.
  - Optional: GPU for faster inference with YOLO and transformer models (requires CUDA-compatible `torch`).
- **External Model**: The YOLOv11 model file (`yolov11modelrachut.pt`) for cell phone detection must be placed in the project root directory.
- **Groq API Key**: Required for the LangChain/Groq integration in the cell phone detection app. You must provide your own Groq API key (see Configuration section).

## Installation
1. **Clone the Repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd <project-directory>
   ```

2. **Set Up a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Create a `requirements.txt` file with the following libraries and install them:
   ```
   numpy
   scikit-learn
   joblib
   transformers
   torch
   opencv-python
   librosa
   ultralytics
   streamlit
   supervision
   Pillow
   langchain
   langchain-groq
   ```
   Install using:
   ```bash
   pip install -r requirements.txt
   ```

   **Notes**:
   - Verify the latest stable versions of these libraries as of May 2025.
   - If using a GPU, ensure `torch` is compatible with your CUDA version (e.g., `pip install torch==2.3.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html` for CUDA 12.1).
   - If `librosa` encounters issues, you may need `soundfile` and `ffmpeg`:
     ```bash
     pip install soundfile
     sudo apt-get install ffmpeg  # On Ubuntu; adjust for other OS
     ```

4. **Verify Camera Access**:
   Ensure your webcam is accessible. The default camera ID is `0` (configurable in `config.json`).

## Usage
The project has two main entry points:

### 1. Core Detection System
This runs the original object detection system with AI-driven context analysis and recommendations.

- **Run the Application**:
  ```bash
  python main.py
  ```
- **What to Expect**:
  - A window displaying the webcam feed with annotated detections.
  - Console output with AI recommendations and chatbot responses.
  - Logs saved in the `logs/` directory.
- **Stop the Application**:
  Press `q` in the OpenCV window to exit.

### 2. Cell Phone Detection Web App
This launches a Streamlit-based web app focused on cell phone detection.

- **Run the Application**:
  ```bash
  streamlit run ui/streamlit_app.py
  ```
- **What to Expect**:
  - A web interface displaying the webcam feed with annotated cell phone detections.
  - AI-generated warnings and suggestions when a cell phone is detected.
- **Stop the Application**:
  Click the "Stop" button in the web interface.

## Configuration
- **`config.json`**: Contains settings for the core detection system (e.g., `camera_id`, `yolo_model`, `target_classes`). Adjust as needed.
- **Groq API Key**: The cell phone detection web app requires a Groq API key for the LangChain/Groq integration. The current key in `ui/streamlit_app.py` is a placeholder. Replace it with your own Groq API key:
  1. Obtain a Groq API key from the Groq platform (https://groq.com).
  2. Open `ui/streamlit_app.py` and locate the line:
     ```python
     groq_api_key = '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'
     ```
  3. Replace the placeholder with your own key:
     ```python
     groq_api_key = 'your-groq-api-key'
     ```
  4. For better security, consider using environment variables:
     ```bash
     export GROQ_API_KEY='your-groq-api-key'
     ```
     Update `ui/streamlit_app.py` to use:
     ```python
     import os
     groq_api_key = os.getenv("GROQ_API_KEY")
     ```

## Dependencies
The project relies on the following libraries:
- `numpy`: Numerical operations.
- `scikit-learn`: Machine learning models (`RandomForestClassifier`, `KMeans`).
- `joblib`: Model persistence.
- `transformers`: Transformer-based NLP (GPT-2).
- `torch`: Backend for transformers.
- `opencv-python`: Video capture and image processing.
- `librosa`: Audio processing.
- `ultralytics`: YOLO models (YOLOv8 and YOLOv11).
- `streamlit`: Web interface for cell phone detection.
- `supervision`: Detection annotations for YOLOv11.
- `Pillow`: Image handling in Streamlit.
- `langchain`: Language model chaining.
- `langchain-groq`: Groq LLM integration.

## Notes
- **YOLOv11 Model**: Ensure `yolov11modelrachut.pt` is in the project root directory for cell phone detection.
- **Performance**: The system supports GPU acceleration for YOLO and transformer models. Configure `torch` accordingly if using a GPU.
- **Extensibility**: Add new modules under `models/` or `detection/` to extend functionality (e.g., new sensors, AI models).
- **Logging**: Detections and AI interactions are logged in the `logs/` directory for debugging and analysis.

## Troubleshooting
- **Camera Issues**: If the webcam fails to open, check the `camera_id` in `config.json` or ensure the camera is connected.
- **Dependency Conflicts**: Use `pipdeptree` to diagnose conflicts:
  ```bash
  pip install pipdeptree
  pipdeptree
  ```
- **Groq API Key Errors**: If the cell phone detection app fails to generate responses, ensure you have replaced the placeholder Groq API key in `ui/streamlit_app.py` with your own valid key. Verify the key is active and has the necessary permissions on the Groq platform.
- **Streamlit Errors**: If the web app fails to load, ensure `streamlit` is installed and the port (default 8501) is not in use.

## Contributing
Contributions are welcome! Please fork the repository, create a new branch, and submit a pull request with your changes.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details (if applicable).

## Acknowledgments
- Built with contributions from various open-source libraries: Ultralytics YOLO, Transformers, LangChain, Streamlit, and more.
- Special thanks to the xAI team for inspiring AI-driven solutions.

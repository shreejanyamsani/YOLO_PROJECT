import numpy as np
import librosa

class AudioContextProcessor:
    """Processes audio input for additional context."""
    
    def __init__(self):
        self.sr = 16000  # Sample rate for audio processing
    
    def extract_audio_features(self, audio_data):
        """Extract audio features for context analysis."""
        if audio_data is None or len(audio_data) == 0:
            return [0] * 5
        
        try:
            y = audio_data.astype(np.float32)
            mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=5)
            return np.mean(mfcc, axis=1).tolist()
        except:
            return [0] * 5
    
    def process_audio(self, audio_data):
        """Process audio data for context information."""
        if audio_data is None:
            return {"noise_level": 0.0, "context": "silent"}
        
        try:
            y = audio_data.astype(np.float32)
            rms = librosa.feature.rms(y=y)[0]
            noise_level = np.mean(rms)
            
            return {
                "noise_level": min(noise_level, 1.0),
                "context": "noisy" if noise_level > 0.5 else "quiet"
            }
        except:
            return {"noise_level": 0.0, "context": "silent"}
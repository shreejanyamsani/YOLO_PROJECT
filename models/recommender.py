from transformers import pipeline
import torch
from collections import deque
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='logs/ai_system.log'
)

class EnhancedContextAwareRecommender:
    """Advanced context-aware recommender with NLP and temporal analysis."""
    
    def __init__(self, ai_processor):
        self.ai_processor = ai_processor
        self.recommendation_history = deque(maxlen=200)
        self.user_feedback = {}
        self.nlp = pipeline("text-generation", model="gpt2", device=0 if torch.cuda.is_available() else -1)
    
    def get_recommendations(self, detections, timestamp, frame=None, audio_data=None):
        """Get enhanced context-aware recommendations."""
        state = self.ai_processor.process_detections(detections, timestamp, frame, audio_data)
        base_recommendations = self._generate_base_recommendations(state)
        filtered_recommendations = self._filter_recommendations(base_recommendations)
        self._update_history(filtered_recommendations)
        
        return filtered_recommendations
    
    def _generate_base_recommendations(self, state):
        """Generate enhanced recommendations with NLP augmentation."""
        recommendations = []
        
        if "suggestions" in state:
            recommendations.extend([{"type": "suggestion", "content": s, "confidence": state["context_confidence"]} 
                                 for s in state["suggestions"]])
        
        if state["behavior"]:
            behavior = state["behavior"]["name"]
            prompt = f"Generate a recommendation for someone {behavior} in a {state['context']} environment."
            generated = self.nlp(prompt, max_length=50, num_return_sequences=1)[0]["generated_text"]
            recommendations.append({
                "type": "ai_suggestion",
                "content": generated,
                "confidence": state["behavior"]["confidence"]
            })
        
        if state["audio_context"] and state["audio_context"]["noise_level"] > 0.7:
            recommendations.append({
                "type": "environment",
                "content": "Consider noise-canceling headphones for better focus.",
                "confidence": state["audio_context"]["noise_level"]
            })
        
        return recommendations
    
    def _filter_recommendations(self, recommendations):
        """Filter recommendations with sentiment analysis."""
        filtered = []
        for rec in recommendations:
            rec_hash = hash(rec["content"])
            sentiment = self.ai_processor.sentiment_analyzer(rec["content"])[0]
            
            if (self.recommendation_history and rec_hash in self.recommendation_history[-10:] or
                rec_hash in self.user_feedback and self.user_feedback[rec_hash] < 0 or
                sentiment["label"] == "NEGATIVE" and sentiment["score"] > 0.7):
                continue
            
            filtered.append(rec)
        
        return filtered[:3]
    
    def _update_history(self, recommendations):
        """Update recommendation history."""
        for rec in recommendations:
            self.recommendation_history.append(hash(rec["content"]))
    
    def provide_feedback(self, recommendation, score):
        """Handle user feedback with sentiment analysis."""
        self.user_feedback[hash(recommendation)] = score
        logging.info(f"User feedback recorded: {recommendation} - Score: {score}")
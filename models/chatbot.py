from transformers import pipeline
import torch
import random

class EnhancedChatbot:
    """Advanced chatbot with transformer-based NLP."""
    
    def __init__(self):
        self.nlp = pipeline("text-generation", model="gpt2", device=0 if torch.cuda.is_available() else -1)
        self.responses = {
            "productivity": [
                "Setting up a productivity timer for focused work.",
                "Would you like to try the Pomodoro technique?",
                "How about a 25-minute focused work session?"
            ],
            "hydration": [
                "Stay hydrated! Want regular water reminders?",
                "Hydration is key! Keep sipping that water.",
                "Try to drink a glass of water every hour."
            ],
            "group": [
                "Group mode activated. Minimal notifications enabled.",
                "Multiple people detected. Want to start a group activity?",
                "Switching to group mode for collaboration."
            ],
            "default": [
                "How can I assist you today?",
                "Anything specific you need help with?",
                "I'm here to make your day better!"
            ]
        }
    
    def get_response(self, context, user_input):
        """Generate response with transformer-based augmentation."""
        user_input = user_input.lower()
        category = "default"
        
        if "productivity" in context.lower() or any(word in user_input for word in ["timer", "work", "focus"]):
            category = "productivity"
        elif "hydrat" in context.lower() or any(word in user_input for word in ["water", "drink", "thirsty"]):
            category = "hydration"
        elif "group" in context.lower() or any(word in user_input for word in ["people", "meeting", "group"]):
            category = "group"
        
        if user_input.strip():
            prompt = f"User said: {user_input}. Context: {context}. Generate a helpful response."
            generated = self.nlp(prompt, max_length=50, num_return_sequences=1)[0]["generated_text"]
            return generated
        
        return random.choice(self.responses[category])
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
import time
from .chatbot import EnhancedChatbot

class CellPhoneDetectionAgent:
    """AI agent for detecting cell phones and providing warnings/suggestions."""
    
    def __init__(self, api_key, system_prompt, model_name='llama3-8b-8192'):
        self.groq_api_key = api_key
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.memory = ConversationBufferWindowMemory(k=3, memory_key="chat_history", return_messages=True)
        self.groq_chat = ChatGroq(groq_api_key=self.groq_api_key, model_name=self.model_name)
        self.last_alert_time = 0
        self.alert_cooldown = 5  # seconds between alerts
        self.chatbot = EnhancedChatbot()  # Integrate with existing chatbot
    
    def get_ai_response(self, detection_info, use_groq=True):
        """Get AI response using either Groq (LangChain) or the existing chatbot."""
        if use_groq:
            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=self.system_prompt),
                    MessagesPlaceholder(variable_name="chat_history"),
                    HumanMessagePromptTemplate.from_template(f"Detection: {detection_info}")
                ]
            )
            conversation = LLMChain(
                llm=self.groq_chat,
                prompt=prompt,
                verbose=True,
                memory=self.memory,
            )
            response = conversation.predict(human_input=detection_info)
            return response.strip().capitalize()
        else:
            return self.chatbot.get_response("cell phone detection", detection_info)
    
    def should_alert(self):
        """Check if enough time has passed to issue a new alert."""
        current_time = time.time()
        if current_time - self.last_alert_time >= self.alert_cooldown:
            self.last_alert_time = current_time
            return True
        return False
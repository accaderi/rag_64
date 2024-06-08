# chat/consumers.py

import json
import uuid
from channels.generic.websocket import WebsocketConsumer
from .models import ChatSession, ChatMessage
import os
from .llm import *

# API keys
groq_api_key = 'your groq api key'
os.environ["TAVILY_API_KEY"] = "your tavily api key"
os.environ["GOOGLE_CSE_ID"] = "your google cse id"
os.environ["GOOGLE_API_KEY"] = "your google api key"


# Global variables
switch_state = None
local_llm, model_name = None, None
question_router = None
ensemble_retriever = None
rag_chain = None
web_search_tool = None
wikipedia = None
pubmed = None
app = None


class ChatConsumer(WebsocketConsumer):
    
    def connect(self):
        self.session_title = self.scope['url_route']['kwargs']['chat_session']
        session_id = str(uuid.uuid4())
        self.session = ChatSession.objects.create(session_id=session_id, session_title=self.session_title)
        self.accept()
        
        global switch_state, local_llm, model_name, question_router, ensemble_retriever
        global rag_chain, web_search_tool, wikipedia, pubmed, app

        switch_state = switch_state_reader()
        local_llm, model_name = llm_chooser(switch_state)
        question_router = router(switch_state, local_llm, model_name, groq_api_key)
        _, ensemble_retriever = retriever_creator(switch_state)
        rag_chain = generator(switch_state, local_llm, model_name, groq_api_key)
        web_search_tool = web_search_chooser(switch_state)
        wikipedia = wikipedia_search_creator()
        pubmed = pubmed_search_creator()
        app = create_workflow()

        # Send message back to frontend connected
        self.send(text_data=json.dumps({
            'message': 'WebSocket connection established. AI is ready.',
            'sender': 'server'
        }))

    def disconnect(self, close_code):
        pass

    # Receive message from WebSocket
    def receive(self, text_data):
        text_data_json = json.loads(text_data)
        message = text_data_json['message']

        # Create a new chat message (user)
        ChatMessage.objects.create(session=self.session, message=message, sender='user')
        inputs = {"switch_state": switch_state, "question_router": question_router, "ensemble_retriever": ensemble_retriever,
                  "rag_chain": rag_chain, "web_search_tool": web_search_tool, "wikipedia": wikipedia, "pubmed": pubmed,
                  "question": str(message)}

        for output in app.stream(inputs):
            for key, value in output.items():
                pass
        
        ChatMessage.objects.create(session=self.session, message=value["generation"], sender='ai')

        # Send message back to the same WebSocket connection
        self.send(text_data=json.dumps({
            'message': value["generation"],
            'sender': 'ai',
            'sam': text_data_json['sam']
        }))

    # Function to send a message from the server
        # Not adapted

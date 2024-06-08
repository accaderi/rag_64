from django.db import models

class SwitchState(models.Model):
    routing_switch = models.BooleanField(default=False)
    retriever_switch = models.BooleanField(default=False)
    wikipedia_switch = models.BooleanField(default=False)
    pubmed_switch = models.BooleanField(default=False)
    web_search_switch = models.CharField(max_length=50, choices=[('None', 'None'), ('Google', 'Google'), ('Tavily', 'Tavily')], default='None')
    retrieve_dir = models.CharField(max_length=100, default='')
    llm_switch = models.CharField(max_length=50, choices=[('Ollama/llama3-8b-8192', 'Ollama/llama3-8b-8192'), ('Ollama/phi3-mini-128K', 'Ollama/phi3-mini-128K'), ('Groq/llama3-8b-8192', 'Groq/llama3-8b-8192'), ('Groq/llama3-70b-8192', 'Groq/llama3-70b-8192'),
                                                          ('Groq/mixtral-8x7b-32768', 'Groq/mixtral-8x7b-32768'),
                                                          ('Groq/gemma-7b-it', 'Groq/gemma-7b-it')], default='Ollama/phi3-mini-128K')
    files_in_retrieve_dir = models.CharField(max_length=500, default='')
    files_retrieve_from_changed = models.BooleanField(default=False)

class ChatSession(models.Model):
    session_id = models.CharField(max_length=36, unique=True)
    session_title = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)

class ChatMessage(models.Model):
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages')
    message = models.TextField()
    sender = models.CharField(max_length=10)  # 'user' or 'server'
    timestamp = models.DateTimeField(auto_now_add=True)
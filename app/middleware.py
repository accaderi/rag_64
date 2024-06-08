from django.contrib import messages
from asgiref.sync import async_to_sync
from .consumers import TextConsumer

class MessageMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        for message in messages.get_messages(request):
            async_to_sync(TextConsumer.send_message)(message.message)
        return response
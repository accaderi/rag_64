from django.contrib import admin

from .models import SwitchState, ChatSession, ChatMessage

admin.site.register(SwitchState)
admin.site.register(ChatSession)
admin.site.register(ChatMessage)


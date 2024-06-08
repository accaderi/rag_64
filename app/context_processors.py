# A context processor file is used to provide variables (switch_state) to all HTML pages in the app.
# This avoids the need to pass the variable to each HTML page individually.
# The link of this file to be added to the setting.py in the context_processors section.

from .models import SwitchState

def switch_state_provider(request):
    return {'switch_state': SwitchState.objects.first()}
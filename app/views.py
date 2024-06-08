# views.py

from django.shortcuts import render, redirect
from django.contrib.auth import login, get_user_model, authenticate, logout
from django.contrib.sites.shortcuts import get_current_site
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.template.loader import render_to_string
from django.utils.encoding import force_bytes, force_str
from django.contrib.auth.tokens import default_token_generator
from django.core.mail import EmailMessage
from .forms import NewUserForm
from django.http import HttpResponse, JsonResponse
from .models import ChatSession, SwitchState
from django.contrib.auth.decorators import login_required
import os
from datetime import datetime
from .llm import *

def index_page(request):
    return render(request, template_name='app/index.html')

def logout_view(request):
    logout(request)
    return redirect('/')  # Redirect to login page after logout

def register_request(request):
    if request.method == "POST":
        form = NewUserForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.is_active = False  # Set user as inactive by default
            user.save()
            email = form.cleaned_data.get('email')
            current_site = get_current_site(request)
            mail_subject = 'Activate your account.'
            message = render_to_string('app/acc_active_email.html', {
                'user': user,
                'domain': current_site.domain,
                'uid': urlsafe_base64_encode(force_bytes(user.pk)),
                'token': default_token_generator.make_token(user),
            })
            email = EmailMessage(
                        mail_subject, message, to=[email]
            )
            email.send()
            return redirect('email_verification')
    else:
        form = NewUserForm()
    return render (request=request, template_name="app/register.html", context={"register_form":form})


def activate_account(request, uidb64, token):
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        user = get_user_model().objects.get(pk=uid)
    except(TypeError, ValueError, OverflowError, get_user_model().DoesNotExist):
        user = None
    if user is not None and default_token_generator.check_token(user, token):
        user.is_active = True
        user.save()
        return redirect('registration_complete')
    else:
        return HttpResponse('Activation link is invalid!')
    

def email_verification(request):
    return render(request, 'app/email_verification.html')

def registration_complete(request):
    return render(request, 'app/registration_complete.html')

def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('index')

    return render(request, 'app/login.html')

@login_required
def workflow_view(request):
    switch_state, _ = SwitchState.objects.get_or_create(id=1)
    
    return render(request, 'app/workflow.html', {'switch_state': switch_state,
                                                 'files_to_retrieve_dir': switch_state.retrieve_dir.replace('\\\\', '\\')})


def check_pdf_in_directory(directory_path):
    # Check if the directory path is valid and is a directory
    if not os.path.isdir(directory_path):
        return {'exists': False, 'pdf': False}
    else:
    
        # List all files in the directory
        files = os.listdir(directory_path)
    
        # Check if there is any file with a .pdf extension
        for file in files:
            if file.lower().endswith('.pdf'):
                return {'exists': True, 'pdf': True}
        
        return {'exists': True, 'pdf': False}
    

def list_files_with_timestamps(directory):
    result = []
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file():
                info = entry.stat()
                creation_time = datetime.fromtimestamp(info.st_ctime)
                modification_time = datetime.fromtimestamp(info.st_mtime)
                file_info = f"{entry.name} Created: {creation_time} Modified: {modification_time}"
                result.append(file_info)
    return "<-+->".join(result)
    

def set_db_switch_state(switch_state, switch_id, switch_value, pdf_chk = ''):
    # Convert the switch value to a boolean
    switch_value = switch_value.lower() == 'true'
    # For other switches
    setattr(switch_state, switch_id, switch_value)
    switch_state.save()
    if pdf_chk != 'exists':
        return {'status': 'error', 'message': 'Directory does not exist.'}
    elif pdf_chk == 'exists':
        return {'status': 'error', 'message': 'No PDF files found in the directory.'}


@login_required
def update_switch_state(request):
    switch_id = request.POST.get('switchId')
    switch_value = request.POST.get('switchValue')

    switch_state, _ = SwitchState.objects.get_or_create(id=1)

    if 'webSearch' in switch_id:
        # For web search switch, set the switch value directly
        switch_state.web_search_switch = switch_id[9:]
        switch_state.save()
        return JsonResponse({'message': 'Switch state updated successfully.'})
    elif 'M_' in switch_id:
        switch_state.llm_switch = switch_id[2:]
        switch_state.save()
        return JsonResponse({'message': 'Switch state updated successfully.'})
    else:
        if 'retriever_switch' == switch_id and switch_value.lower() == 'true':
            try:
                files_to_retrieve_dir = request.POST.get('filesToRetrieveDir')
                if not files_to_retrieve_dir:
                    files_to_retrieve_dir = ''
                else:
                    files_to_retrieve_dir = files_to_retrieve_dir.replace('\\', '\\\\').replace('"', '')
                pdf_check = check_pdf_in_directory(files_to_retrieve_dir)
                print(pdf_check, pdf_check['exists'], pdf_check['pdf'])
                if not pdf_check['exists'] or not pdf_check['pdf']:
                    switch_state.retrieve_dir = ''
                    return JsonResponse(set_db_switch_state(switch_state, switch_id, switch_value='false', pdf_chk='exists' if pdf_check['exists'] else 'no_pdf'))
                else:
                    try:
                        files = list_files_with_timestamps(files_to_retrieve_dir)
                        if files and files != switch_state.files_in_retrieve_dir:
                            switch_state.files_in_retrieve_dir = files
                            switch_state.files_retrieve_from_changed = True
                        else:
                            switch_state.files_retrieve_from_changed = False
                        switch_state.save()
                    except:
                        print('Could not list files in the directory.')
                        pass
                    switch_state.retrieve_dir = files_to_retrieve_dir
                    set_db_switch_state(switch_state, switch_id, switch_value)
                    return JsonResponse({'status': 'success', 'message': 'PDF files found in the directory.'})
            except:
                pass

        set_db_switch_state(switch_state, switch_id, switch_value)

    return JsonResponse({'message': 'Switch state updated successfully.'})



@login_required
def chat_view(request, chat_session):

    return render(request, "app/chat.html")


@login_required
def get_chat_session(request, chat_session):
    try:
        session = ChatSession.objects.get(session_title=chat_session)
        messages = session.messages.values('message', 'sender', 'timestamp')
        return JsonResponse({'messages': list(messages)})
    except ChatSession.DoesNotExist:
        return JsonResponse({'error': 'Chat session not found'}, status=404)
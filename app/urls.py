from django.urls import path
from app import views

urlpatterns = [
    path('', views.index_page, name='index'),
    path('logout/', views.logout_view, name='logout'),
    path('register/', views.register_request, name='register'),
    path('activate/<uidb64>/<token>/', views.activate_account, name='activate'),
    path('email-verification/', views.email_verification, name='email_verification'),
    path('registration_complete/', views.registration_complete, name='registration_complete'),
    path('login/', views.login_view, name='login'),
    path('workflow/', views.workflow_view, name='workflow'),
    path('update_switch_state/', views.update_switch_state, name='update_switch_state'),
    path("chat/<str:chat_session>/", views.chat_view, name="chat"),
    path('chat/get_chat_session/<str:chat_session>/', views.get_chat_session, name='chat_session')
]
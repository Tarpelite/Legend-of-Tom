from django.urls import path
from django.conf.urls import url

from . import views

app_name = "TextWeb"
urlpatterns = [
    path('Choice/', views.Choice, name='Choice'),
    url(r'^$', views.predict_based_time, name='Answer')
]
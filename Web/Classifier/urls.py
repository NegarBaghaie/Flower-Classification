from django.urls import path
from .views import classify_view

app_name = 'Classifier'

urlpatterns = [
    path('', classify_view, name='classify')
]

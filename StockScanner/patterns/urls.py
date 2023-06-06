from django.urls import path
from . import views

urlpatterns = [
    path('predict_trend_reversal/', views.predict_trend_reversal, name='predict_trend_reversal'),
]

from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
 #   path('candlestick_chart/', views.index, name='index'),
]
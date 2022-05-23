from django.urls import path

from . import views

urlpatterns = [
    path('', views.main, name='main'),
    path('kmeans/', views.KmeansAPIView.as_view(), name="kmeans"),
]

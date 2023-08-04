from django.urls import path

from . import views

app_name = "dashboard"
urlpatterns = [
    path('', views.home, name="home"),
    path('devices/', views.DeviceListView.as_view(), name='devices'),
    path('runs/', views.RunListView.as_view(), name='runs'),
    path('reports/', views.ReportListView.as_view(), name='reports'),
    path('reservoirs', views.ReservoirListView.as_view(), name='reservoirs'),
    path('compare_runs', views.compare_runs, name='compare_runs'),
    path('device/<int:pk>', views.DeviceDetailView.as_view(), name='device-detail'),
    path('run/<int:pk>', views.RunDetailView.as_view(), name='run-detail'),
    path('report/<int:pk>', views.ReportDetailView.as_view(), name='report-detail'),
    path('test', views.test, name="test"),
]

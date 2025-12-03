from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from api import views as api_views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', api_views.map_view, name='map'),  # homepage = map
    path('api/predict/', api_views.predict_view, name='api_predict'),
    path('api/predictions/', api_views.predictions_view, name='api_predictions'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

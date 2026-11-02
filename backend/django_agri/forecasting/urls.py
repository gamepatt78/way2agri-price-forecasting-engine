from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    CommodityPriceViewSet, ModelRegistryViewSet,
    PriceForecastViewSet, CorporateActionViewSet, DashboardViewSet
)

router = DefaultRouter()
router.register(r'prices', CommodityPriceViewSet, basename='price')
router.register(r'models', ModelRegistryViewSet, basename='model')
router.register(r'forecasts', PriceForecastViewSet, basename='forecast')
router.register(r'actions', CorporateActionViewSet, basename='action')
router.register(r'dashboard', DashboardViewSet, basename='dashboard')

urlpatterns = [
    path('api/', include(router.urls)),
]

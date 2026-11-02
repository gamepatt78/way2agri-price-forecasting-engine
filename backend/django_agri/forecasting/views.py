from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.db.models import Avg, Max, Min
from .models import CommodityPrice, ModelRegistry, PriceForecast, CorporateAction
from .serializers import (
    CommodityPriceSerializer, ModelRegistrySerializer,
    PriceForecastSerializer, CorporateActionSerializer
)

class CommodityPriceViewSet(viewsets.ModelViewSet):
    queryset = CommodityPrice.objects.all().order_by('date')
    serializer_class = CommodityPriceSerializer

    @action(detail=False, methods=['get'])
    def summary(self, request):
        count = CommodityPrice.objects.count()
        if count == 0:
            return Response({"error": "No price data available"}, status=status.HTTP_404_NOT_FOUND)
        stats = CommodityPrice.objects.aggregate(
            avg_price=Avg('avg_monthly_price'),
            max_price=Max('avg_monthly_price'),
            min_price=Min('avg_monthly_price')
        )
        latest = CommodityPrice.objects.latest('date')
        return Response({
            "total_months": count,
            "latest_date": latest.date,
            "latest_price": latest.avg_monthly_price,
            "average_price": round(stats['avg_price'], 2),
            "max_price": stats['max_price'],
            "min_price": stats['min_price'],
            "anomalies_detected": CommodityPrice.objects.filter(is_anomaly=True).count()
        })


class ModelRegistryViewSet(viewsets.ModelViewSet):
    queryset = ModelRegistry.objects.all().order_by('-trained_at')
    serializer_class = ModelRegistrySerializer

    @action(detail=True, methods=['post'])
    def activate(self, request, pk=None):
        model = self.get_object()
        ModelRegistry.objects.update(is_active=False)
        model.is_active = True
        model.save()
        return Response({"status": f"Model {model.model_name} activated for production."})


class PriceForecastViewSet(viewsets.ModelViewSet):
    queryset = PriceForecast.objects.all().order_by('target_date')
    serializer_class = PriceForecastSerializer

    @action(detail=False, methods=['get'], url_path='forward-curve')
    def forward_curve(self, request):
        """Returns the 12-month forward price forecast curve from the active model."""
        active_model = ModelRegistry.objects.filter(is_active=True).first()
        if not active_model:
            active_model = ModelRegistry.objects.first()
        
        if not active_model:
            return Response({"error": "No trained model found"}, status=status.HTTP_404_NOT_FOUND)

        forecasts = PriceForecast.objects.filter(model=active_model).order_by('target_date')[:12]
        serializer = PriceForecastSerializer(forecasts, many=True)
        return Response({
            "model_name": active_model.model_name,
            "algorithm_type": active_model.algorithm_type,
            "metrics": {
                "mae": active_model.mae,
                "rmse": active_model.rmse,
                "mape": active_model.mape,
            },
            "horizon_months": len(forecasts),
            "forecast_curve": serializer.data
        })


class CorporateActionViewSet(viewsets.ModelViewSet):
    queryset = CorporateAction.objects.all().order_by('-created_at')
    serializer_class = CorporateActionSerializer


class DashboardViewSet(viewsets.ViewSet):
    @action(detail=False, methods=['get'])
    def stats(self, request):
        active_model = ModelRegistry.objects.filter(is_active=True).first()
        latest_price = CommodityPrice.objects.order_by('-date').first()
        next_month_forecast = None
        if active_model:
            next_month_forecast = PriceForecast.objects.filter(model=active_model).order_by('target_date').first()

        recommended_actions = CorporateAction.objects.filter(status='RECOMMENDED').count()

        return Response({
            "latest_historical_price": latest_price.avg_monthly_price if latest_price else None,
            "latest_historical_date": latest_price.date if latest_price else None,
            "active_model": active_model.model_name if active_model else "None",
            "active_model_mape": active_model.mape if active_model else None,
            "next_month_forecast": next_month_forecast.predicted_price if next_month_forecast else None,
            "pending_recommended_actions": recommended_actions
        })

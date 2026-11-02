from rest_framework import serializers
from .models import CommodityPrice, ModelRegistry, PriceForecast, CorporateAction

class CommodityPriceSerializer(serializers.ModelSerializer):
    class Meta:
        model = CommodityPrice
        fields = '__all__'

class PriceForecastSerializer(serializers.ModelSerializer):
    model_name = serializers.CharField(source='model.model_name', read_only=True)
    algorithm_type = serializers.CharField(source='model.algorithm_type', read_only=True)

    class Meta:
        model = PriceForecast
        fields = '__all__'

class ModelRegistrySerializer(serializers.ModelSerializer):
    forecast_count = serializers.IntegerField(source='forecasts.count', read_only=True)

    class Meta:
        model = ModelRegistry
        fields = '__all__'

class CorporateActionSerializer(serializers.ModelSerializer):
    forecast_details = PriceForecastSerializer(source='forecast', read_only=True)

    class Meta:
        model = CorporateAction
        fields = '__all__'

from django.contrib import admin
from .models import CommodityPrice, ModelRegistry, PriceForecast, CorporateAction

@admin.register(CommodityPrice)
class CommodityPriceAdmin(admin.ModelAdmin):
    list_display = ('date', 'avg_monthly_price', 'is_anomaly', 'imputed_price', 'created_at')
    list_filter = ('is_anomaly', 'date')
    search_fields = ('date',)
    ordering = ('-date',)

@admin.register(ModelRegistry)
class ModelRegistryAdmin(admin.ModelAdmin):
    list_display = ('model_name', 'version', 'algorithm_type', 'mae', 'rmse', 'mape', 'is_active', 'trained_at')
    list_filter = ('algorithm_type', 'is_active')
    search_fields = ('model_name', 'version')

@admin.register(PriceForecast)
class PriceForecastAdmin(admin.ModelAdmin):
    list_display = ('target_date', 'predicted_price', 'lower_bound', 'upper_bound', 'horizon_months', 'model')
    list_filter = ('model', 'horizon_months')
    ordering = ('target_date',)

@admin.register(CorporateAction)
class CorporateActionAdmin(admin.ModelAdmin):
    list_display = ('action_type', 'status', 'trigger_condition', 'expected_savings_inr', 'created_at')
    list_filter = ('action_type', 'status')

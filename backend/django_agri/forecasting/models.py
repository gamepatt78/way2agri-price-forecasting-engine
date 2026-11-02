from django.db import models
from django.utils import timezone

class CommodityPrice(models.Model):
    date = models.DateField(unique=True, db_index=True)
    avg_monthly_price = models.FloatField(help_text="Historical average monthly price in INR/quintal")
    is_anomaly = models.BooleanField(default=False, help_text="Flag indicating if the record is a detected anomaly")
    imputed_price = models.FloatField(null=True, blank=True, help_text="Imputed price if original was anomalous")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['date']
        verbose_name = "Commodity Price Record"
        verbose_name_plural = "Commodity Price Records"

    def __str__(self):
        return f"{self.date.strftime('%Y-%m')}: INR {self.avg_monthly_price:.2f}"


class ModelRegistry(models.Model):
    model_name = models.CharField(max_length=100)
    version = models.CharField(max_length=50)
    algorithm_type = models.CharField(max_length=100, help_text="e.g. SARIMAX, XGBoost, Random Forest, Ensemble")
    mae = models.FloatField(help_text="Mean Absolute Error on validation set")
    rmse = models.FloatField(help_text="Root Mean Squared Error")
    mape = models.FloatField(help_text="Mean Absolute Percentage Error (%)")
    directional_accuracy = models.FloatField(default=0.0, help_text="Directional Accuracy (%)")
    parameters = models.JSONField(default=dict, help_text="Hyperparameters and feature config")
    artifact_path = models.CharField(max_length=255, help_text="Path to saved model artifact .pkl/.joblib")
    is_active = models.BooleanField(default=False, help_text="Active model used for production predictions")
    trained_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ['-trained_at']
        verbose_name = "Model Registry"
        verbose_name_plural = "Model Registries"

    def __str__(self):
        status = "ACTIVE" if self.is_active else "ARCHIVED"
        return f"{self.model_name} v{self.version} ({self.algorithm_type}) - [{status}]"


class PriceForecast(models.Model):
    model = models.ForeignKey(ModelRegistry, on_delete=models.CASCADE, related_name='forecasts')
    forecast_date = models.DateField(help_text="The date when the forecast was generated")
    target_date = models.DateField(db_index=True, help_text="The future month being predicted")
    predicted_price = models.FloatField(help_text="Point forecast in INR")
    lower_bound = models.FloatField(help_text="95% confidence interval lower bound")
    upper_bound = models.FloatField(help_text="95% confidence interval upper bound")
    horizon_months = models.IntegerField(help_text="Forecast horizon in months (1-12)")
    actual_price = models.FloatField(null=True, blank=True, help_text="Actual realized price when available for drift check")
    absolute_error = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['target_date']
        unique_together = ('model', 'forecast_date', 'target_date')
        verbose_name = "Price Forecast"
        verbose_name_plural = "Price Forecasts"

    def __str__(self):
        return f"{self.target_date.strftime('%Y-%m')}: Predicted INR {self.predicted_price:.2f} (H={self.horizon_months})"


class CorporateAction(models.Model):
    ACTION_TYPES = [
        ('FORWARD_CONTRACT', 'Forward / Futures Contract Lock-in'),
        ('INVENTORY_BUFFER', 'Warehouse Stockpile / Strategic Buffer Accumulation'),
        ('DYNAMIC_PRICING', 'Dynamic End-Product Price Adjustment'),
        ('SUPPLIER_RENEGOTIATION', 'Supplier Re-negotiation / Diversification'),
        ('HEDGE_PROTECTION', 'Financial Derivatives / Options Hedging'),
    ]

    STATUS_CHOICES = [
        ('RECOMMENDED', 'Recommended by AI'),
        ('UNDER_REVIEW', 'Under Procurement Committee Review'),
        ('EXECUTED', 'Executed'),
        ('REJECTED', 'Rejected'),
    ]

    forecast = models.ForeignKey(PriceForecast, on_delete=models.CASCADE, related_name='recommended_actions')
    action_type = models.CharField(max_length=50, choices=ACTION_TYPES)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='RECOMMENDED')
    trigger_condition = models.CharField(max_length=255, help_text="e.g. Predicted price increase > 15% in Q2")
    action_details = models.TextField()
    expected_savings_inr = models.FloatField(default=0.0, help_text="Projected cost avoidance or margin enhancement")
    actual_savings_inr = models.FloatField(null=True, blank=True, help_text="Realized post-execution savings")
    roi_percentage = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.get_action_type_display()} - Status: {self.status}"

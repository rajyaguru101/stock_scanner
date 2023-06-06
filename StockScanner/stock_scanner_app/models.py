# stock_scanner_app/models.py code :
from django.db import models

class CompanyInfo(models.Model):
    symbol = models.CharField(max_length=20, unique=True)
    exchange = models.CharField(max_length=100)
    sector = models.CharField(max_length=100, blank=True, null=True)
    company_name = models.CharField(max_length=200)
    country = models.CharField(max_length=100)

    def __str__(self):
        return self.company_name
    
class ScannerConfiguration(models.Model):
    name = models.CharField(max_length=100, unique=True)
    scanner_type = models.CharField(max_length=50)
    preferences = models.JSONField(null=True, blank=True)  # Requires Django 3.1 or higher

    def __str__(self):
        return self.name

class HistoricalData(models.Model):
    company = models.ForeignKey(CompanyInfo, on_delete=models.CASCADE)
    symbol = models.CharField(max_length=20) # new field
    date = models.DateField()
    open = models.FloatField()
    high = models.FloatField()
    low = models.FloatField()
    close = models.FloatField()
    volume = models.FloatField()

    def __str__(self):
        return f"{self.symbol} - {self.date}"

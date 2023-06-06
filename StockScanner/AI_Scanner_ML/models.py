
#AI_Scanner_ML/models.py
from django.db import models

class StockData(models.Model):
    date = models.DateField()
    symbol = models.CharField(max_length=10)
    open = models.FloatField()
    high = models.FloatField()
    low = models.FloatField()
    close = models.FloatField()
    volume = models.IntegerField()

    def __str__(self):
        return f'{self.symbol} {self.date}'


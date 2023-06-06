# run_dojimodel.py

from django.core.management.base import BaseCommand
from stock_scanner_app.doji_model import train_and_test_model


class Command(BaseCommand):
    help = 'Train and test the Doji candlestick model using the HistoricalData table'

    def handle(self, *args, **options):
        
        trained_model, accuracy, cm = train_and_test_model()

        self.stdout.write(self.style.SUCCESS("Model trained."))
        self.stdout.write("Accuracy: {:.2f}".format(accuracy))
        self.stdout.write("Confusion Matrix:\n{}".format(cm))




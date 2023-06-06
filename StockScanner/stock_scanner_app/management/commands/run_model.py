from django.core.management.base import BaseCommand
from models.testing.test_model import test_model, save_results
import datetime

class Command(BaseCommand):
    help = 'Train and test the Doji model'

    def handle(self, *args, **options):
        accuracy, report = test_model()
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H_%M')
        result_file = f'results/model_1_{timestamp}.csv'
        with open(result_file, 'w') as f:
            f.write(f'Accuracy: {accuracy}\n')
            f.write(f'{report}\n')

        self.stdout.write(self.style.SUCCESS(f'Trained and tested the Doji model. Results saved in: {result_file}'))

# python manage.py run_dojimodel
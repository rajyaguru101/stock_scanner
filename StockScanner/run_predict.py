import os
import sys
from django.core.wsgi import get_wsgi_application

# Set up the Django environment
os.environ['DJANGO_SETTINGS_MODULE'] = 'StockScanner.settings'
application = get_wsgi_application()

# Import and run the doji model script
from patterns.models import predict_doji

if __name__ == '__main__':
    predict_doji.main()

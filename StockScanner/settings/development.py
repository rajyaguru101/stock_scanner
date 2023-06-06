from .base import *

DEBUG = True

ALLOWED_HOSTS = ['4.18.48.173']

# Configure your local database settings here
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Additional development-specific settings

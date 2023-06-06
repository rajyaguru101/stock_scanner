from .base import *

DEBUG = False

# Replace with your production domain or Google App Engine domain
ALLOWED_HOSTS = ['aialgo-379410.appspot.com', '34.18.38.50']

# Replace with your actual secret key
SECRET_KEY = '5d2_##^rs*b-28^1^w+_v!p18rwxoi#^l(f)n0jbqsef35#5-&'

# Configure your production database settings here (e.g., Google Cloud SQL)
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Add security settings, such as SSL/TLS, secure cookies, etc.

# Configure static files handling for production
STATIC_ROOT = os.path.join(BASE_DIR, 'static')

# If you're using Google Cloud Storage for serving static/media files, you can configure it here
# STATICFILES_STORAGE = 'storages.backends.gcloud.GoogleCloudStorage'
# GS_BUCKET_NAME = 'your_bucket_name'
# GS_PROJECT_ID = 'your_project_id'
# GS_DEFAULT_ACL = 'publicRead'

# If you're using Google Cloud Storage for serving media files
# DEFAULT_FILE_STORAGE = 'storages.backends.gcloud.GoogleCloudStorage'
# MEDIA_URL = 'https://storage.googleapis.com/your_bucket_name/'

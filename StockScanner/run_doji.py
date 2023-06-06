import os
import sys
from django.core.wsgi import get_wsgi_application
from patterns.models.doji import prepare_data, preprocess_data, create_model, train_model, evaluate_model
import joblib
import django

# Set up the Django environment
os.environ['DJANGO_SETTINGS_MODULE'] = 'StockScanner.settings'
django.setup()
application = get_wsgi_application()

if __name__ == "__main__":
    # Prepare and preprocess the data
    features, labels = prepare_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(features, labels)

    # Create, train, and evaluate the model
    input_shape = (X_train.shape[1], 1)

    model = create_model(input_shape)
    train_model(model, X_train, y_train)
    evaluate_model(model, X_test, y_test)

    # Save the model and the scaler
    model.save('doji_model.h5')
    joblib.dump(scaler, 'doji_scaler.pkl')
# scanner_preferences.py

from collections import defaultdict

SCANNER_PREFERENCES = {
    'doji': ['doji_tolerance'],
    'spike_volume': ['spike_threshold'],
    'volatility': ['atr_multiplier', 'atr_window_size'],
    'percentage_stock_scanner': ['percentage_stock_scanner'],
    # Add other scanner types and their preference fields here
}

def get_preferences(form, scanner_type):
    # Get the base_preferences
    base_preferences = {
        'timeframe': form.cleaned_data['timeframe'],

    }
    
    # Get the specific_preferences for the scanner_type
    specific_preferences = {}
    for pref_field in SCANNER_PREFERENCES.get(scanner_type, []):
        specific_preferences[pref_field] = form.cleaned_data[pref_field]

    # Combine base_preferences and specific_preferences
    final_preferences = {**base_preferences, **specific_preferences}
    return final_preferences

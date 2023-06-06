def extract_features(data):
    X = data.drop(columns=['doji'])
    y = data['doji']
    return X, y
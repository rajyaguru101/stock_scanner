from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models.doji import make_prediction

@api_view(['POST'])
def predict_trend_reversal(request):
    symbol = request.data.get('symbol')
    features = request.data.get('features')

    if not symbol or not features or len(features) != 4:
        return Response({"error": "Invalid input data"}, status=400)

    prediction = make_prediction(symbol, features)
    return Response({"symbol": symbol, "predicted_trend_reversal": bool(prediction)})

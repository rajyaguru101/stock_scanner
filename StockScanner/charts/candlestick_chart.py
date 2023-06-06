# candlestick_chart.py
import plotly.graph_objects as go
from .base_chart import BaseChart

class CandlestickChart(BaseChart):
    def __init__(self, df, title='Candlestick Chart'):
        self.df = df
        self.title = title

    def create_chart(self):
        fig = go.Figure(data=[go.Candlestick(x=self.df['Date'],
                                             open=self.df['Open'],
                                             high=self.df['High'],
                                             low=self.df['Low'],
                                             close=self.df['Close'])])

        fig.update_layout(title=self.title,
                          xaxis_title='Date',
                          yaxis_title='Price',
                          xaxis_rangeslider_visible=False)

        return fig

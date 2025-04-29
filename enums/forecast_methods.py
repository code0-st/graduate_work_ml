from enum import Enum

class ForecastMethod(str, Enum):
    SMA = "SMA"
    EMA = "EMA"
    HOLT_WINTERS = "HoltWinters"
    LINEAR_REGRESSION = "LinearRegression"
    NEURAL_NETWORK = "NeuralNetwork"
    ARIMA = "ARIMA"
    SARIMA = "SARIMA"
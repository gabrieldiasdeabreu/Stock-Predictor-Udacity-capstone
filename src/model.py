import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
import datetime 

def make_window(series, window, ahead):
    """
        series: a numpy time series with shape (n, 1)
        window : size of past values each step of LSTM will see
        ahead : size of future values each step of LSTM will predict
    """
    size = len(series)
    X = np.array([series[i - window:i] for i in range(window, size - ahead)])
    y = np.array([series[i + ahead] for i in range(window, size - ahead)])
    X_train = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y



def create_model(X_train, y_train):
    """
        X_train: X train set to train our model with shapes (n_observations, sequence_length, 1)
        y_train: X train set to train our model with shapes (n_observations, 1)
    """

    model = Sequential()

    # LSTM Layer
    model.add(LSTM(units = 50, input_shape = (X_train.shape[1], 1)))

    # Regularization Layer
    model.add(Dropout(0.2))

    model.add(Dense(units = 10, activation='linear'))

    # output Layer
    model.add(Dense(units = 1, activation='linear'))

    # Compile
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    # Fit
    model.fit(X_train, y_train, epochs = 200, validation_split=0.1, batch_size = 400, callbacks=[
        EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    ])
    return model



def train_and_execute_model(stock_name, window, ahead): 
    """
        stock_name: valid stock name to yahoo api
        window: window size to prediction, how many past values are going see to predict
        ahead: how far (days) in the future model will predict
    """
    # get yahoo data
    ticker = yf.Ticker(stock_name)
    df = ticker.history(period="50y")
    
    ## uses all values for future prediction
    train_set_size = int(df.shape[0]) 
    close_stock_serie = df['Close'].values.reshape(-1,1)

    ## scales
    sc = MinMaxScaler(feature_range = (.2, .8))
    training_set_scaled = sc.fit_transform(close_stock_serie[:train_set_size])    
   
    # make windows, compile and fit model
    X_train, y_train = make_window(training_set_scaled, window, ahead)
    model = create_model(X_train, y_train)

    # gets data to predict ahead
    stock_serie = df['Close']
    last_windows = stock_serie[-(window + ahead  ):].values.reshape(-1,1)
    last_windows_scaled = sc.transform(last_windows)

    # windows to predict
    X_last_windows = np.array([last_windows_scaled[i - window:i] for i in range(window, len(last_windows_scaled))])
    X_last_windows_to_lstm = np.reshape(X_last_windows, (X_last_windows.shape[0], X_last_windows.shape[1], 1))

    # inverse scale to return predictions
    future_ahead_prediction = sc.inverse_transform(model.predict(X_last_windows_to_lstm))

    # calcule dates
    tomorow = pd.to_datetime(stock_serie.index[-1])
    days_ahead = [tomorow + datetime.timedelta(days=n) for n in range(1, ahead+1)]

    return  days_ahead, future_ahead_prediction.ravel()

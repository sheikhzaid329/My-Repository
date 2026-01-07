
# Select which model to run (set only one to True)

RNN_code = True

LSTM_code = False

GRU_code = False


from tensorflow.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import SimpleRNN, GRU, LSTM, Dense, Dropout

if  RNN_code:
    # Data Loading, Preparing and Scaling
    data = pd.read_csv('FinalAssessment/Section-B-Q1-MSFT_1986-03-13_2025-04-06.csv',parse_dates=['date'],index_col='date')

    Feature_col = data[['high', 'low', 'open','adj_close', 'volume']].astype(float)#.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['close']])

    window_size = 256

    X = Feature_col

    y = data['close']

# RNN models require 3D input in the form (samples, time steps, features)
    # Creating Sequences and Train-Test Split
    window_size = 12
    X = []
    y = []
    target_dates = data.index[window_size:]

    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i - window_size:i, 0])
        y.append(scaled_data[i, 0])

    X = np.array(X)
    y = np.array(y)
    # Splitting the Data into Training and Testing Sets
    X_train,X_test, y_train, y_test = train_test_split(
         X, y, test_size = 0.2,random_state=42
    )

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    # Building the RNN Model
    model = keras.models.Sequential()
    model.add(keras.layers.SimpleRNN(64, return_sequences=True))
    model.add(keras.layers.SimpleRNN(64))
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(1))

    METRICS = [ 
                    'mean_squared_error',
                    'mean_absolute_error']

    model.compile(optimizer='adam', loss='mean_squared_error',metrics = METRICS)
    
    # Training and Evaluating the Model
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=10)])

    predictions = model.predict(X_test)

    rmse = np.sqrt(np.mean((y_test - predictions)**2))
    print(f'RMSE: {rmse:.2f}')

    """
    Visualizing Model Performance
    In this step, we visualize the actual vs predicted values. A plot is generated to compare the actual Stocks against the predicted values, allowing us to evaluate how well the model performs over time.
    """
    plt.figure(figsize=(12, 6))
    plt.plot( y_test, label='Actual stocks')
    plt.plot( predictions, label='Predicted stocks ')
    plt.title('Actual vs Predicted stocks')
    plt.xlabel('date')
    plt.ylabel('Stocks')
    plt.legend()
    plt.show()
    # End of RNN code

"""
RNN:
    RNN is simple and works well for short sequences but struggles with long-term dependencies.

LSTM:
    LSTM is more complex but effectively captures long-term dependencies and provides better performance on most sequence-based tasks.

Therefore, we will use LSTM for more acuracy

 """
if  LSTM_code:


    """
     Data Loading, Preparing and Scaling
    Here we are using a dataset of Stocks investment data using LSTM.

    """
    data = pd.read_csv('FinalAssessment/Section-B-Q1-MSFT_1986-03-13_2025-04-06.csv')
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    Features = data[['high', 'low', 'open','adj_close', 'volume']]
    close = data[['close']]

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(close)

    """
     Creating Sequences and Train-Test Split
    Here we generate sequences of input data and split the dataset into training and testing sets.

    We use a sliding window of 256 trading days to predict the next month's Stocks.
    The dataset is split into training and testing sets and reshaped to match the LSTM input shape.
    We split 80% data for training and 20% for testing purposes.
    """
    window_size = 256
    X = []
    y = []
    target_dates = data.index[window_size:]

    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i - window_size:i, 0])
        y.append(scaled_data[i, 0])

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
        X, y, target_dates, test_size=0.2, shuffle=False
    )

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    """
     Building the LSTM Model
    This step involves defining and building the LSTM model architecture.

    The model consists of two LSTM layers, each with 128 units and a dropout layer after each to prevent overfitting.
    The model concludes with a Dense layer to predict a single value (next month's Stock prediction).
    """
    model = Sequential()
    model.add(LSTM(units=128, return_sequences=True,
          input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=128))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    METRICS = [ 
                    'mean_squared_error',
                    'mean_absolute_error']

    model.compile(optimizer='adam', loss='mean_squared_error',metrics = METRICS)
    """
     Training and Evaluating the Model
    In this step, we train the model on the training data and evaluate its performance.

    The model is trained for 100 epochs using a batch size of 32, with 10% of the training data used for validation.
    After training the model is used to make predictions on the test set and we calculate the Root Mean Squared Error (RMSE) to evaluate performance.
    """
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=10)])

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions).flatten()
    y_test = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()

    rmse = np.sqrt(np.mean((y_test - predictions)**2))
    print(f'RMSE: {rmse:.2f}')

    """
     Visualizing Model Performance
    In this step, we visualize the actual vs predicted values. A plot is generated to compare the actual stocks against the predicted values, allowing us to evaluate how well the model performs over time.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dates_test, y_test, label='Actual Stocks')
    plt.plot(dates_test, predictions, label='Predicted Stocks')
    plt.title('Actual vs Predicted Stocks Prices')
    plt.xlabel('Time')
    plt.ylabel('price')
    plt.legend()
    plt.show()
    # End of LSTM code
"""

LSTM:
    Preferred when working with complex time-series data where long-term patterns matter,
 such as stock prices or speech signals, but it requires more training time.

GRU:
    Commonly used when faster training and lower computational cost are needed,
 often delivering similar results on time-series forecasting tasks.

"""

if GRU_code:

    """
 Loading the Dataset
The dataset we’re using is a time-series dataset containing daily stocks data for Microsoft (MSFT) from March 13, 1986, to April 6, 2025.

"""
df = pd.read_csv('FinalAssessment/Section-B-Q1-MSFT_1986-03-13_2025-04-06.csv', parse_dates=['date'], index_col='date')
print(df.head())

"""
 Preprocessing the Data

MinMaxScaler(): This scales the data to a range of 0 to 1. This is important because neural networks perform better when input features are scaled properly.

"""
target_col = 'close'
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[[target_col]])


""""
 Preparing Data for GRU
create_dataset(): Prepares the dataset for time-series forecasting. It creates sliding windows of time_step length to predict the next time step.
X.reshape(): Reshapes the input data to fit the expected shape for the GRU which is 3D: [samples, time steps, features].
"""
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0]) 
        y.append(data[i + time_step, 0]) 
    return np.array(X), np.array(y)

time_step = 100 
X, y = create_dataset(scaled_data, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)


"""
 Building the GRU Model
GRU(units=50): Adds a GRU layer with 50 units (neurons).
return_sequences=True: Ensures that the GRU layer returns the entire sequence (required for stacking multiple GRU layers).
Dense(units=1): The output layer which predicts a single value for the next time step.
Adam(): An adaptive optimizer commonly used in deep learning.

"""
model = Sequential()
model.add(Dropout(0.2))
model.add(GRU(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(GRU(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='linear')) 

METRICS = [ 
                    'mean_squared_error',
                    'mean_absolute_error']

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error',metrics = METRICS)

""""
model.fit(): Trains the model on the prepared dataset. The epochs=10 specifies the number of iterations over the entire dataset, and batch_size=32 defines the number of samples per batch.

"""
model.fit(X, y, epochs=10, batch_size=32,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=10)])

"""" Making Predictions
Input Sequence: The code takes the last 100 Stock values from the dataset (scaled_data[-time_step:]) as an input sequence.
Reshaping the Input Sequence: The input sequence is reshaped into the shape (1, time_step, 1) because the GRU model expects a 3D input: [samples, time_steps, features]. Here samples=1 because we are making one prediction, time_steps=100 (the length of the input sequence) and features=1 because we are predicting only the stock value.
model.predict(): Uses the trained model to predict future values based on the input data.

"""

input_sequence = scaled_data[-time_step:, 0].reshape(1, time_step, 1)
predicted_values = model.predict(input_sequence)


"""" Inverse Transforming the Predictions
Inverse Transforming the Predictions refers to the process of converting the scaled (normalized) predictions back to their original scale."""

predicted_values = scaler.inverse_transform(predicted_values.reshape(-1, 1)).flatten()
print(f"The predicted close for the next day is: {predicted_values[0]:.2f}")
# End of GRU code
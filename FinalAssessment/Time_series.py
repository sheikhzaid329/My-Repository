
from tensorflow.keras.models import Sequential
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from keras.metrics import Precision, Recall


data = pd.read_csv('FinalAssessment/Section-B-Q1-MSFT_1986-03-13_2025-04-06.csv',parse_dates=['date'],index_col='date')

Feature_col = data[['high', 'low', 'open','adj_close', 'volume']].astype(float)#.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(Feature_col)

window_size = 12

X = Feature_col

y = data['close']

'The error occurred because the target variable was reshaped incorrectly; only the input data needs reshaping in an RNN.'
window_size = 12
X = []
y = []
target_dates = data.index[window_size:]

for i in range(window_size, len(scaled_data)):
    X.append(scaled_data[i - window_size:i, 0])
    y.append(scaled_data[i, 0])

X = np.array(X)
y = np.array(y)

X_train,X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2,random_state=42
)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

model = keras.models.Sequential()
model.add(keras.layers.SimpleRNN(64, return_sequences=True))
model.add(keras.layers.SimpleRNN(64))
model.add(keras.layers.Dense(128, activation="relu"))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(1, activation="sigmoid"))

METRICS = metrics=['accuracy', 
                   Precision(name='precision'),
                   Recall(name='recall')]

model.compile("rmsprop",
              "binary_crossentropy",
               metrics = METRICS)

model.compile(optimizer='adam', loss='mean_squared_error')

#The error occurred because the input to the RNN was two-dimensional, whereas an RNN requires three-dimensional input data.
history = model.fit(X_train, y_train, epochs=10)

predictions = model.predict(X_test)

rmse = np.sqrt(np.mean((y_test - predictions)**2))
print(f'RMSE: {rmse:.2f}')

"""
6. Visualizing Model Performance
In this step, we visualize the actual vs predicted values. A plot is generated to compare the actual Stocks against the predicted values, allowing us to evaluate how well the model performs over time.
"""
plt.figure(figsize=(12, 6))
plt.plot( y_test, label='Actual stocks')
plt.plot( predictions, label='Predicted stocks ')
plt.title('Actual vs Predicted stocks')
plt.xlabel('Date')
plt.ylabel('Stocks')
plt.legend()
plt.show()

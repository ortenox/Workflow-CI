import mlflow
import tensorflow as tf
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt

def create_dataset(series, window_size=10):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    return np.array(X), np.array(y)

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True, help="Path to dataset in CSV format")
args = parser.parse_args()

mlflow.tensorflow.autolog()

# Load CSV
df = pd.read_csv(args.data, parse_dates=["Timestamp"])

# Ambil hanya kolom TrafficCount
traffic = df['TrafficCount'].values

# Buat X, y dari time series (windowed data)
window_size = 10
X, y = create_dataset(traffic, window_size)
X = X.reshape((X.shape[0], X.shape[1], 1))  # (samples, timesteps, features)

# Parameter
batch_size = 64
learning_rate = 1e-3

# Buat dataset TensorFlow
train_set = tf.data.Dataset.from_tensor_slices((X, y))
train_set = train_set.shuffle(1000).batch(batch_size).repeat().prefetch(1)

with mlflow.start_run():
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("window_size", window_size)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(X.shape[1], 1)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1)
    ])

    model.compile(
        loss=tf.keras.losses.Huber(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["mae"]
    )

    steps_per_epoch = len(X) // batch_size
    model.fit(train_set, epochs=10, steps_per_epoch=steps_per_epoch)

    mlflow.tensorflow.log_model(model, artifact_path="lstm_model")
    model.save("lstm_model.h5")

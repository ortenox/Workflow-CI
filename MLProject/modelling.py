import mlflow
import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt

# Argumen input dataset
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True, help="Path to preprocessed dataset CSV")
args = parser.parse_args()

# Autologging
mlflow.tensorflow.autolog()

# Load data
df = pd.read_csv(args.data)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.set_index('Timestamp')
df = df.resample('1h').mean().interpolate()
traffic = df['TrafficCount'].values

# Normalisasi
min_val = traffic.min()
max_val = traffic.max()
traffic = (traffic - min_val) / (max_val - min_val)

# Windowing
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[-1:]))
    return ds.batch(batch_size).repeat().prefetch(1)

# Parameter
window_size = 168
batch_size = 64
shuffle_buffer = 1000
split_time = int(len(traffic) * 0.8)
train_series = traffic[:split_time]
test_series = traffic[split_time:]
train_set = windowed_dataset(train_series, window_size, batch_size, shuffle_buffer)

# MLflow run
with mlflow.start_run():
    mlflow.log_param("window_size", window_size)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("shuffle_buffer", shuffle_buffer)
    mlflow.log_param("learning_rate", 1e-3)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(window_size, 1)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1)
    ])

    model.compile(
        loss=tf.keras.losses.Huber(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=["mae"]
    )

    steps_per_epoch = len(train_series) // batch_size
    model.fit(train_set, epochs=10, steps_per_epoch=steps_per_epoch)

    mlflow.tensorflow.log_model(model, artifact_path="lstm_model")

    # Forecast
    predictions = []
    for i in range(len(test_series) - window_size):
        input_window = test_series[i:i + window_size]
        input_window = input_window.reshape(1, window_size, 1)
        pred = model.predict(input_window, verbose=0)
        predictions.append(pred[0, 0])

    # Un-normalisasi
    test_actual = test_series[window_size:] * (max_val - min_val) + min_val
    predictions = np.array(predictions) * (max_val - min_val) + min_val

    # Plot hasil
    plt.figure(figsize=(15, 6))
    plt.plot(test_actual, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.title("Forecasting Web Traffic")
    plt.xlabel("Time Step")
    plt.ylabel("Traffic Count")
    plt.legend()
    plt.savefig("forecast_plot.png")
    plt.close()

    mlflow.log_artifact("forecast_plot.png")
import mlflow
import tensorflow as tf
import numpy as np
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True, help="Path to preprocessed dataset (.npz)")
args = parser.parse_args()

mlflow.tensorflow.autolog()

# Load preprocessed data
data = np.load(args.data)
X, y = data["X"], data["y"]

# Parameter
batch_size = 64
learning_rate = 1e-3

# Buat dataset TensorFlow
train_set = tf.data.Dataset.from_tensor_slices((X, y))
train_set = train_set.shuffle(1000).batch(batch_size).repeat().prefetch(1)

with mlflow.start_run():
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("learning_rate", learning_rate)

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

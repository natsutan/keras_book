import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib import tpu
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras import models
from tensorflow.keras import layers

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# (404, 13)
print(train_data.shape)

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
train_data /= std


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

# cross validation
k = 4
num_val_samples = len(train_data) // k
num_epochs = 500
all_scores = []
all_mae_histories = []

tpu_grpc_url = tf.contrib.cluster_resolver.TPUClusterResolver(tpu=[os.environ['TPU_NAME']])

strategy = tf.contrib.tpu.TPUDistributionStrategy(tpu_grpc_url)

for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    partial_train_data = np.concatenate(
        [train_data[: i * num_val_samples], train_data[(i+1) * num_val_samples:]],
        axis=0
    )
    partial_train_targets = np.concatenate(
        [train_targets[: i * num_val_samples], train_targets[(i+1) * num_val_samples:]],
        axis=0
    )

    model = build_model()

    tpu_model = tf.contrib.tpu.keras_to_tpu_model(
        model,
        strategy=strategy
    )
    
    history = tpu_model.fit(partial_train_data, partial_train_targets,
                        validation_data = (val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0)
    mae_history =   history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)
    


average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]


plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.savefig('plot.png')


tpu.shutdown_system()

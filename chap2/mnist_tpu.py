import os
import tensorflow as tf
from tensorflow.contrib import tpu
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# (60000, 28, 28)
#print(train_images.shape)

# (10000, 28, 28)
#print(test_images.shape)

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))

# loss function
# optimizer
# metrics

network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# normalization
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255

# one hot encoding
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

tpu_grpc_url = tf.contrib.cluster_resolver.TPUClusterResolver(tpu=[os.environ['TPU_NAME']])

strategy = tf.contrib.tpu.TPUDistributionStrategy(tpu_grpc_url)


tpu_model = tf.contrib.tpu.keras_to_tpu_model(
    network,
    strategy=strategy
)

tpu_model.fit(train_images, train_labels, epochs=5, batch_size=128)
test_loss, test_acc = tpu_model.evaluate(test_images, test_labels)

tpu.shutdown_system()


# test_acc: , 0.9786
print("test_acc: ,", test_acc)


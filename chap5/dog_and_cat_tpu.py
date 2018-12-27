import os
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib import tpu
from tensorflow.contrib.tpu.python.tpu import keras_support
from tensorflow.keras import layers
from tensorflow.keras import models
#from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

original_dataset_dir = 'D:/data/dog_and_cat/train'
base_dir = '/home/natsutan0/myproj/keras_book/chap5/tmp/dog_and_cat_small'

train_dir = os.path.join(base_dir, 'train')
os.makedirs(train_dir, exist_ok=True)
validation_dir = os.path.join(base_dir, 'validation')
os.makedirs(validation_dir, exist_ok=True)
test_dir = os.path.join(base_dir, 'test')
os.makedirs(test_dir, exist_ok=True)


train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=tf.train.RMSPropOptimizer(1e-4) ,metrics=['acc'])

              
tpu_grpc_url = tf.contrib.cluster_resolver.TPUClusterResolver(tpu=[os.environ['TPU_NAME']])

strategy = tf.contrib.tpu.TPUDistributionStrategy(tpu_grpc_url)

tpu_model = tf.contrib.tpu.keras_to_tpu_model(
    model,
    strategy=strategy
)


     
history = tpu_model.fit_generator(train_generator, steps_per_epoch=100, epochs=50,
                              validation_data=validation_generator, validation_steps=50)



# 結果の保存
#model.save('cat_adn_dogs_small_1_tpu.h5')
with open("history_tpu.pickle", mode='wb') as fp:
    pickle.dump(history.history, fp)


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

def dummy():
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and Validation accuracy')
    plt.legend()
    plt.savefig('plot1_tpu.png')


    plt.plot(epochs, loss, 'bo', label='Training logg')
    plt.plot(epochs, val_loss, 'b', label='Validation acc')
    plt.title('Training and Validation loss')
    plt.legend()

    plt.savefig('plot2_tpu.png')

tpu.shutdown_system()

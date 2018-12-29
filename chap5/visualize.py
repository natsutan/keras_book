import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.preprocessing import image

model_name = 'save/dog_and_cat_small_1_v2.h5'
img_path = 'save/wiki_cat1.jpg'

# https://stackoverflow.com/questions/53183865/unknown-initializer-glorotuniform-when-loading-keras-model
with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    model = load_model(model_name)
    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.0
  #  print(img_tensor.shape)

 #   plt.imshow(img_tensor[0])
 #   plt.show()

    layer_outputs = [layer.output for layer in model.layers[:8]]
#    model.summary()
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model(img_tensor)

    first_layer_activation = activations[0]

    print(first_layer_activation[0,0,0,0])
#    print(first_layer_activation[0,0,0,1])

    mat = first_layer_activation[0, :, :, 3].eval()
    plt.matshow(mat, cmap='viridis')
    plt.show()
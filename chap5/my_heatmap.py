import matplotlib as plt
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.initializers import glorot_uniform
# from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
import numpy as np
import cv2
import os

IMAGE_IS_CAT = True

#model_name = 'save/dog_and_cat_small_1_v2.h5'
model_name = 'save/cat_and_dogs_all.h5'
save_dir = 'save'
output_dir = 'output'


if IMAGE_IS_CAT:
    img_path = os.path.join(save_dir, 'wiki_cat1.jpg')
else:
    img_path = 'save/wiki_dog1.jpg'

img = image.load_img(img_path, target_size=(150, 150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x /= 255.0

with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    model = load_model(model_name)

    preds = model.predict(x)
    print('Predicted:', preds)

model.summary()

last_conv_layer = model.get_layer('conv2d_3')
print(model.output.shape)
model_output = model.output[:, 0]

grads = K.gradients(model_output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))

iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_value = iterate([x])

for i in range(128):
    conv_layer_value[:, :, 1] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_value, axis=-1)
heetmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

cv2.imwrite(os.path.join(output_dir, 'my_hm1.png'), heatmap)
superimposed_img = heatmap * 0.4 + img
cv2.imwrite(os.path.join(output_dir, 'my_heatmap1.png'), superimposed_img)


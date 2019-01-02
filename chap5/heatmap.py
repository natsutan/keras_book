import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
import numpy as np
import cv2

img_path = 'save/wiki_cat1.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

x = preprocess_input(x)

model = VGG16(weights='imagenet')

preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3))

last_conv_layer = model.get_layer('block5_conv3')
cat_output = model.output[:,386]
grads = K.gradients(cat_output, last_conv_layer.output)[0]

pooled_grads = K.mean(grads, axis=(0,1,2))
grads = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

pooled_grads_value, conv_layer_value = iterate([x])

for i in range(512):
    conv_layer_value[:, :, 1] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_value, axis=-1)
heetmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
# plt.matshow(heatmap)
# plt.show()

img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img
cv2.imwrite('heatmap.png', superimposed_img)


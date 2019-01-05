from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.initializers import glorot_uniform
# from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
import numpy as np
import cv2

model_name = 'save/dog_and_cat_small_1_v2.h5'
img_path = 'save/wiki_cat1.jpg'
#img_path = 'save/wiki_dog1.jpg'

img = image.load_img(img_path, target_size=(150, 150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x /= 255.0

with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    model = load_model(model_name)

    preds = model.predict(x)
    print('Predicted:', preds)



# For benchmarking against DEFER, try this file that uses Single Device Inference

from keras.applications import ResNet50
from keras.applications.resnet import preprocess_input
import numpy as np
import time
import tensorflow as tf

model = ResNet50(weights='imagenet', include_top=True)

img_path = 'pizza.jpeg'
img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
x = tf.keras.utils.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

time_run = 1
in_sec = time_run * 60
start = time.time()
result_count = 0
while (time.time() - start) < in_sec:
    model.predict(x)
    result_count += 1
print(f"In {time_run} min, {result_count} results")
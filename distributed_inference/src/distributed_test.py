from master import DistributedInference
import threading
import tensorflow as tf
import keras.applications
from keras.applications import ResNet50
from keras.applications.resnet import preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np
import queue
import time

numWorkers = 2
workerNodeIPs = ["172.16.0.1","172.16.0.2"]
distributedInference = DistributedInference(workerNodeIPs)

model = ResNet50(weights='imagenet', include_top=True)
# Depending on the number of compute nodes, use different number of partitions
# "part_at" is where the graph is split, so there should be one less element in this
# list than the number of partitions you want
part_at = ["conv3_block4_out"]
img_path = 'pizza.jpeg'
img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
x = tf.keras.utils.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

time_min = 1
run = time_min * 60

start = time.time()
def print_result(q):
    res_count = 0
    while (time.time() - start) < run:
        res = q.get()
        res_count += 1
        print(res.shape)
    print(f"{res_count} results in {time_min} min")
    print(f"Throughput: {res_count / run}")
    exit(0)

input_q = queue.Queue(10)
output_q = queue.Queue(10)

a = threading.Thread(target = distributedInference.start, args=(model, part_at, input_q, output_q), daemon=True)
b = threading.Thread(target=print_result, args=(output_q,))
a.start()
b.start()

for i in range(10):
    # Whatever input you want
    input_q.put(x)

b.join()

from dispatcher import DEFER
import threading
import tensorflow as tf
import keras.applications
from keras.applications import ResNet50
from keras.applications.resnet import preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np
import queue
import time
from core.emulator.coreemu import CoreEmu
from core.emulator.data import IpPrefixes
from core.emulator.enumerations import EventTypes
from core.nodes.base import CoreNode, Position
from core.nodes.network import SwitchNode

# ip nerator for example
ip_prefixes = IpPrefixes(ip4_prefix="10.0.0.0/24")

# create emulator instance for creating sessions and utility methods
coreemu = CoreEmu()
session = coreemu.create_session()

# must be in configuration state for nodes to start, when using "node_add" below
session.set_state(EventTypes.CONFIGURATION_STATE)

# create switch
position = Position(x=200, y=200)
switch = session.add_node(SwitchNode, position=position)

# create nodes
position = Position(x=100, y=100)
n1 = session.add_node(CoreNode, position=position)
n1.set_model(model="PC")
n1.set_command("sudop python3 node.py")
position = Position(x=300, y=100)
n2 = session.add_node(CoreNode, position=position)
n2.set_command("sudop python3 node.py")
position = Position(x=200, y=100)
n3 = session.add_node(CoreNode, position=position)
n3.set_command("sudop python3 node.py")
position = Position(x=400, y=100)
n4 = session.add_node(CoreNode, position=position)
n4.set_command("sudop python3 node.py")

# link nodes to switch
iface1 = ip_prefixes.create_iface(n1)
session.add_link(n1.id, switch.id, iface1)
print("n1:", iface1.ip4)
iface1 = ip_prefixes.create_iface(n2)
session.add_link(n2.id, switch.id, iface1)
print("n2:", iface1.ip4)
iface1 = ip_prefixes.create_iface(n3)
session.add_link(n3.id, switch.id, iface1)
print("n3:", iface1.ip4)
iface1 = ip_prefixes.create_iface(n4)
session.add_link(n4.id, switch.id, iface1)
print("n4:", iface1.ip4)

# start session
session.instantiate()

try:

    numNodes = 2
    computeNodes = ["10.0.0.2","10.0.0.3"]
    defer = DEFER(computeNodes)

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

    a = threading.Thread(target=defer.run_defer, args=(model, part_at, input_q, output_q), daemon=True)
    b = threading.Thread(target=print_result, args=(output_q,))
    a.start()
    b.start()

    for i in range(1000):
        # Whatever input you want
        input_q.put(x)

    b.join()

    session.shutdown()
except Exception as e:
    session.shutdown()

import orjson as json
import queue
import socket
from threading import Thread
from queue import Queue
import select
import time

import numpy as np
import tensorflow as tf

from sock_comm import socket_recv, socket_send
from worker_state import WorkerState

import zfpy
import lz4.frame


class WorkerNode:

    def _wsocket(self, worker_state):
    
        chunk_size = worker_state.chunk_size
        weights_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        weights_server.bind(("0.0.0.0", 5002))
        weights_server.listen(1)
        weights_cli = weights_server.accept()[0]
        weights_cli.setblocking(0)
        
        model_weights = self._retrieve_weights(weights_cli, chunk_size)
        worker_state.weights = model_weights
        weights_server.close()

    def _retrieve_weights(self, connection, buffer_size):
        remaining_size = 8
        byte_array = bytearray()
    
        while remaining_size > 0:
            try:
                received = connection.recv(min(remaining_size, 8))
                remaining_size -= len(received)
                byte_array.extend(received)
            except socket.error as err:
                if err.errno != socket.EAGAIN:
                    raise err
                select.select([connection], [], [])
    
        array_length = int.from_bytes(byte_array, 'big')
    
        weights_list = []
        for _ in range(array_length):
            received_bytes = bytes(socket_recv(connection, buffer_size))
            weights_list.append(self._decompressData(received_bytes))

        return weights_list

        
    def _compressData(self, arr):
        return lz4.frame.compress(zfpy.compress_numpy(arr))
        
    def _decompressData(self, byts):
        return zfpy.decompress_numpy(lz4.frame.decompress(byts))
        
    def _server(self, worker_state, to_send):
    
        chunk_size = worker_state.chunk_size
        data_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        data_server.bind(("0.0.0.0", 5000))
        data_server.listen(1) 
        data_cli = data_server.accept()[0]
        data_cli.setblocking(0)

        while True: 
            inpt = self._decompressData(bytes(socket_recv(data_cli, chunk_size)))
            to_send.put(inpt)

    def _client(self, worker_state, payload):
    
        while not worker_state.next_node:
            time.sleep(5)  

        model = worker_state.model
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((worker_state.next_node, 5000))
        client_socket.setblocking(0)

        while True:
            input_data = payload.get()
            prediction = model.predict(input_data)
            compressed_data = self._compressData(prediction)
            socket_send(compressed_data, client_socket, worker_state.chunk_size)
            
    def _msocket(self, worker_state):
    
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind(("0.0.0.0", 5001))
        print("Model socket running")
        server_socket.listen(1) 
        client_socket = server_socket.accept()[0]
        client_socket.setblocking(0)

        model_data = socket_recv(client_socket, worker_state.chunk_size)
        next_node_data = socket_recv(client_socket, chunk_size=1)
    
        partial_model = tf.keras.models.model_from_json(model_data)
    
        while not worker_state.weights:  
            time.sleep(5)

        partial_model.set_weights(worker_state.weights)
        partial_model.make_predict_function()
        worker_state.model = partial_model
    
        worker_state.next_node = next_node_data.decode()
        select.select([], [client_socket], [])
        client_socket.send(b'\x06')
        server_socket.close()

    def start(self):
    
        workerState = WorkerState(chunk_size = 512 * 1000)
        
        modelThread = Thread(target=self._msocket, args=(workerState,))
        weightsThread = Thread(target=self._wsocket, args=(workerState,))
        
        to_send = queue.Queue(5000) 
        server = Thread(target=self._server, args=(workerState, to_send))
        client = Thread(target=self._client, args=(workerState, to_send))
        
        modelThread.start()
        weightsThread.start()
        
        server.start()
        client.start()
        
        modelThread.join()
        weightsThread.join()
        
        server.join()
        client.join()

worker_node = WorkerNode()
worker_node.start()


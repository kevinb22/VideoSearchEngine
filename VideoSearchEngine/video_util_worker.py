import video_utils
import pickle
import socket
import sys
from _thread import start_new_thread
import threading
import time
import torch
import os
import numpy as np
import tqdm
import ObjectDetection.TinyYolo as TinyYolo
import ImageCaptioningYolo.train as image_train
import ImageCaptioningYolo.im_args as im_args
import ImageCaptioningYolo.sample as image_sample
from ImageCaptioningYolo.build_vocab import Vocabulary
import ObjectDetection.Yolo as Yolo
from ImageCaptioner import ImageCaptioner
import video_utils
import random

def thread_main(conn, captioner, count, host, port):
    # Accept the byte chunks sent by VideoDistributer.py and join aggregate toegther
    data_list = []
    data = conn.recv(1024)
    data_list.append(data)
    while data:
        data = conn.recv(1024)
        data_list.append(data)
    conn.close()
    all_data = b''.join(data_list)

    # De-pickle bytes to reconstruct array of images, manipulate as needed.
    try:
       unpickled_data = pickle.loads(all_data)
    except Exception as e:
       print(e)
       unpickled_data = []

    # Insert metadata header and generate captions
    metadata = unpickled_data[0]
    unpickled_cluster_filename = metadata["file_name"]  # unpickled_data[0]
    unpickled_cluster_num = metadata["cluster_num"]  # unpickled_data[1]
    total_clusters = metadata["total_clusters"]
    unpickled_data = unpickled_data[1:]
    summaries = []
    frame_clusters = video_utils.group_semantic_frames(unpickled_data)
    for frame_cluster in tqdm.tqdm(frame_clusters):
        frames = random.choices(frame_cluster, k=10)
        for frame in frames:
            frame = np.array([np.array(frame)])
            if frame is torch.cuda.FloatTensor:
                frame = frame.cpu()
            caption = captioner.get_caption(frame)
            summaries.append(caption)
       
    # Pickle the array of summaries.
    summaries.insert(0, {"file_name": unpickled_cluster_filename, "cluster_num": unpickled_cluster_num, "total_clusters": total_clusters})
    data = pickle.dumps(summaries)

    # Create a socket object
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Send pickle file over the network to server.
    print("Sending cluster " + str(unpickled_cluster_num) + " to collector: " + str(host) + ":" + str(port))
    s.connect((host, port))
    s.sendall(data)
    s.close() 
    
def load_necessary():
    captioner = ImageCaptioner()
    captioner.load_models()
    return captioner

'''
Usage:
python video_util_worker.py <localhost:port_to_listen_on> <host_to_send_to:port_to_send_to>
python VideoSearchEngine/video_util_worker.py localhost:24448 lobster.cs.washington.edu:1234
'''
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python video_util_worker.py <localhost:port_to_listen_on> <host_to_send_to:port_to_send_to>")
        exit()

    host = ''                 # Symbolic name meaning all available interfaces 
    port = int(sys.argv[1].split(':')[1])

    collector_host = sys.argv[2].split(':')[0]
    collector_port = int(sys.argv[2].split(':')[1])

    image_captioner = load_necessary()
    print("Worker started, listening on: " + socket.gethostname()  + ":" + str(port) + " sending to: " + collector_host + ":" + str(collector_port))
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, port))
    s.listen(5)
    count = 0
    while True:
        #establish connection with client
        conn, addr = s.accept()
        # Start new thread
        start_new_thread(thread_main, (conn, image_captioner, count, collector_host, collector_port,))
        count = count + 1
    s.close()
        


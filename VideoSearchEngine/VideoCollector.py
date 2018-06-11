import video_utils
import pickle
import socket
import sys
from _thread import start_new_thread
import threading
import time
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
import database_utils
import SummaryJoiner
from nltk.corpus import brown, stopwords

map_lock = threading.Lock()
video_summary = {}

def finished(filename, summary, total_num):
    print("Finished", filename)
    summary_list = []
    for i in range(total_num):
        for summ in summary[i]:
            summary_list.append(summ)
    token_strings = summary_list # [s.split(" ") for s in summary_list]
    res = SummaryJoiner.textrank(token_strings, top_n=5, stopwords=stopwords.words('english'))
    print(res)
    database_utils.upload_new_summary(os.path.basename(filename), '\n'.join(res), filename)
    print("done uploading", filename)


def thread_main(conn, count):
    # Accept the pickle file sent by VideoDistributer.py and write/cache to local copy.
    data_list = []
    data = conn.recv(1024)
    data_list.append(data)
    while data:
        data = conn.recv(1024)
        data_list.append(data)
    conn.close()
    all_data = b''.join(data_list)

    # De-pickle file to reconstruct array of images, manipulate as needed.
    try:
        unpickled_data = pickle.loads(all_data)
    except Exception as e:
        print(e)
        unpickled_data = []
    
    metadata = unpickled_data[0]
    cluster_filename = metadata["file_name"]
    cluster_num = metadata["cluster_num"]
    total_num = metadata["total_clusters"]
    # cluster_filename = unpickled_data[0]
    # cluster_num = unpickled_data[1]
    unpickled_data = unpickled_data[1:]
    print(cluster_num, "finished")
    map_lock.acquire()

    if cluster_filename not in video_summary:
        video_summary[cluster_filename] = {}
    
    if cluster_num in video_summary[cluster_filename]:
        print("WTFFFFFFFFFFFFF")
    video_summary[cluster_filename][cluster_num] = unpickled_data
    print(len(video_summary[cluster_filename]), total_num)
    if len(video_summary[cluster_filename]) == total_num:
        finished(cluster_filename, video_summary[cluster_filename], total_num)
        del video_summary[cluster_filename]
    map_lock.release()

'''
Usage:
python VideoColleector.py <localhost:port_to_listen_on>
python VideoSearchEngine/VideoCollector.py localhost:24448 
'''
if __name__ == '__main__':
    if len(sys.argv) != 2:
      print("Usage: python video_util_worker.py <localhost:port_to_listen_on> ")
      exit()

    host = '' # Symbolic name meaning all available interfaces 
    port = int(sys.argv[1].split(':')[1])
    print("Collector started, listening on: " + socket.gethostname() + ":" + str(port))
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, port))
    s.listen(64)
    count = 0
    while True:
        #establish connection with client
        conn, addr = s.accept()
        # Start new thread
        start_new_thread(thread_main, (conn, count,))
        count = count + 1
    s.close()
        

import requests
import cv2
from time import sleep
import pickle

from threading import Thread
import SocketServer
from SimpleHTTPServer import SimpleHTTPRequestHandler
from RemoteFaceClassifier import *
from RemoteFaceClassifier.Client import *


class FaceClassifierResultsListener:
    class ClientListenHandler(SimpleHTTPRequestHandler):
        def _set_response(self):
            self.send_response(code=200)
            self.send_header("Content-type", "text/html")
            self.end_headers()

        def do_POST(self):
            self._set_response()
            content_length = int(self.headers['Content-Length'])  # <--- Gets the size of data
            post_data = self.rfile.read(content_length)  # <--- Gets the data itself
            frame_idx, maxI, predictions = pickle.loads(post_data)
            print("Recv results for frame #{}: {}, {}".format(frame_idx, maxI, predictions))

    def start(self):
        SocketServer.TCPServer.allow_reuse_address = True
        httpd = SocketServer.TCPServer((CLIENT, CLIENT_RES_PORT), self.ClientListenHandler)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        httpd.server_close()


class FaceClassifierClient:

    def __init__(self):
        pass

    def recv_confidence_result(self):
        """ Recv confidence results from server """
        FaceClassifierResultsListener().start()

    def start(self):
        """Start sending video frames to server"""
        frame_idx = 0
        vidcap = cv2.VideoCapture(CLIENT_VIDEO_PATH)
        success, img = vidcap.read()
        if not success:
            print("Fail to read video frames")
            exit(1)

        try:
            # Spawn another thread receiving responses
            t = Thread(target=self.recv_confidence_result)
            t.daemon = True
            t.start()
            while success:
                frame_idx += 1
                success, img = vidcap.read()
                requests.post("http://{}:{}".format(SERVER, str(SERVER_FRAME_PORT)), data=pickle.dumps((frame_idx, img)))
                sleep(CLIENT_FRAME_INTERVAL / 1000.0)

        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    FaceClassifierClient().start()

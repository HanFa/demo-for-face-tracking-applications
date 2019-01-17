import requests
import cv2
from time import sleep
import pickle

from threading import Thread
from collections import OrderedDict
import SocketServer
from SimpleHTTPServer import SimpleHTTPRequestHandler
from RemoteFaceClassifier import *
from RemoteFaceClassifier.Client import *

imgs = OrderedDict()

class FaceClassifierResultsListener:
    class ClientListenHandler(SimpleHTTPRequestHandler):

        def _print_res(self, frame_idx, maxI_lst, predictions_lst, bb_bl_lst, bb_tr_lst):
            """Print the results for face classification.
                Input:
                @frame_idx: index for the current frame
                @maxI_lst: classification labels for each face
                @predictions_lst: classification confidences for each face
                @bb_bl_lst: face boundaries
            """
            assert len(maxI_lst) == len(predictions_lst)

            print("Recv results for frame #{}".format(frame_idx))

            img = imgs[frame_idx]
            for i in range(len(maxI_lst)):
                print("\t Person id: {} \t confidence: {} \t Bound: {} {}"
                      .format(maxI_lst[i], predictions_lst[i], bb_bl_lst[i], bb_tr_lst[i]))

                valid = (predictions_lst[i][maxI_lst[i]] > CLASSIFY_THRESHOLD)
                img = cv2.rectangle(img, thickness=5, pt1=bb_bl_lst[i], pt2=bb_tr_lst[i],
                                    color=FACE_RECT_COLORS[maxI_lst[i]]
                                        if valid and maxI_lst[i] in FACE_RECT_COLORS.keys() else (255, 255, 255))

                img = cv2.putText(
                    img,
                    "Person: {} Conf: {}".format(maxI_lst[i], predictions_lst[i][maxI_lst[i]]) if valid else 'Unknown',
                    bb_tr_lst[i], cv2.FONT_HERSHEY_PLAIN, 1, LABEL_COLOR, 2
                )


            # show gui
            if SHOW_GUI:
                cv2.imshow(WINDOW_NAME, imgs[frame_idx])
                cv2.waitKey(1)


        def _set_response(self):
            self.send_response(code=200)
            self.send_header("Content-type", "text/html")
            self.end_headers()

        def do_POST(self):
            self._set_response()
            content_length = int(self.headers['Content-Length'])  # <--- Gets the size of data
            post_data = self.rfile.read(content_length)  # <--- Gets the data itself


            frame_idx, maxI_lst, predictions_lst, bb_bl_lst, bb_tr_lst = pickle.loads(post_data)
            self._print_res(frame_idx, maxI_lst, predictions_lst, bb_bl_lst, bb_tr_lst)


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
                success, img = vidcap.read()

                if frame_idx % VIDEO_SAMPLE_RATE == 0:
                    print("Send frame #{}".format(frame_idx))
                    imgs[frame_idx] = img
                    requests.post("http://{}:{}".format(SERVER, str(SERVER_FRAME_PORT)),
                                  data=pickle.dumps((frame_idx, img)))

                frame_idx += 1

        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    FaceClassifierClient().start()

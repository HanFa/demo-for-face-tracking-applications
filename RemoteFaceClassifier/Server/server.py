import sys, pickle, shutil, time
import requests
import SocketServer
import pandas as pd
import numpy as np
from threading import Thread, Condition
from SimpleHTTPServer import SimpleHTTPRequestHandler

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from RemoteFaceClassifier import *
from RemoteFaceClassifier.Server import *
from RemoteFaceClassifier.Server.classifier import full_search_face_boxes, getRep
import RemoteFaceClassifier.Server.globals as globals
from RemoteFaceClassifier.Server.profile import MEASURE_TYPE, profiler

# global variables
handler_has_setup = False
clf = None
labels = None
embeddings = None

class ServerHandler(SimpleHTTPRequestHandler):

    def _set_response(self):
        self.send_response(code=200)
        self.send_header("Content-type", "text/html")
        self.end_headers()


    def init_handler(self):
        global handler_has_setup, clf, labels, embeddings
        print("Starting ServerHandler, loading pretrained model classifier.pkl...")
        with open(SERVER_PRETRAINED, 'rb') as f:
            if sys.version_info[0] < 3:
                _, clf = pickle.load(f)
            else:
                _, clf = pickle.load(f, encoding='latin1')

        fname = "{}/labels.csv".format(os.path.dirname(SERVER_PRETRAINED))
        labels = pd.read_csv(fname, header=None).as_matrix()[:, 0]
        fname = "{}/reps.csv".format(os.path.dirname(SERVER_PRETRAINED))
        embeddings = pd.read_csv(fname, header=None).as_matrix()
        print("Finish loading pre-trained model classifier.pkl")
        handler_has_setup = True



    def train(self):
        global handler_has_setup, clf, labels, embeddings
        le = LabelEncoder().fit(labels)
        labelsNum = le.transform(labels)
        clf = SVC(C=1, kernel='linear', probability=True)
        clf.fit(embeddings, labelsNum)


    def infer(self, img_array):
        global handler_has_setup, clf, labels, embeddings

        # Predict using current model
        reps = getRep(img_array, SERVER_MULT_FACE_INFER)
        maxI_lst, predictions_lst, bb_lst, rep_lst = [], [], [], []

        for r in reps:
            rep = r[1].reshape(1, -1)
            rep_lst.append(rep)
            predictions = clf.predict_proba(rep).ravel()
            maxI = np.argmax(predictions)
            predictions_lst.append(predictions)
            maxI_lst.append(maxI)
            bb_lst.append(r[2])


        if SERVER_MODE == 'Stateful':
            # Add the faces in current frame into the additional dataset
            print("labels: {}".format(labels))
            print("embedding: {}".format(embeddings))

            for bb, rep in zip(bb_lst, rep_lst):
                # @TODO: more generic ground truth labelling support
                # in this case, we use the following ground truth in the video:
                # Joe face is always on the left while Obama on the right
                maxI = 1 if bb.center().x < 512L else 2
                labels = np.append(labels, maxI)
                embeddings = np.append(embeddings, rep, axis=0)

            print("*labels: {}".format(labels.shape))
            print("*embedding: {}".format(embeddings.shape))

            self.train()

        return maxI_lst, predictions_lst, bb_lst


    def do_POST(self):
        global handler_has_setup, clf, labels, embeddings, profiler

        if not handler_has_setup: self.init_handler()
        profiler.inform_transmission_time_start(MEASURE_TYPE.TOTAL)

        self._set_response()
        content_length = int(self.headers['Content-Length'])  # <--- Gets the size of data

        profiler.inform_transmission_time_start(MEASURE_TYPE.TRANSMISSION)
        post_data = self.rfile.read(content_length)  # <--- Gets the data itself
        profiler.inform_transmission_time_stop(MEASURE_TYPE.TRANSMISSION)

        frame_idx, img = pickle.loads(post_data)

        if SERVER_MODE in ["Stateless", "Stateful"]:
            maxI_lst, predictions_lst, bb_lst = self.infer(img)
        else:
            raise Exception("Invalid SERVER_MODE {}: ['Stateful', 'Stateless']".format(SERVER_MODE))

        # convert boundaries from dlib.rectanngle to corner points
        bb_bl_lst = []
        bb_tr_lst = []
        for bb in bb_lst:
            bb_bl_lst.append((bb.bl_corner().x, bb.bl_corner().y))
            bb_tr_lst.append((bb.tr_corner().x, bb.tr_corner().y))

        # Return the results to client
        requests.post("http://{}:{}".format(CLIENT, str(CLIENT_RES_PORT)),
                      data=pickle.dumps((frame_idx, maxI_lst, predictions_lst, bb_bl_lst, bb_tr_lst)))

        profiler.inform_transmission_time_stop(MEASURE_TYPE.TOTAL)
        profiler.update_log()

        if globals.frame_num % 3 == 2:
            Thread(target=full_search_face_boxes, args=(img,)).start()  # full search asyncronously

        globals.frame_num += 1


class FaceClassifierServer:

    def start(self):
        SocketServer.TCPServer.allow_reuse_address = True
        httpd = SocketServer.TCPServer((SERVER, SERVER_FRAME_PORT), ServerHandler)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        httpd.server_close()


if __name__ == '__main__':
    FaceClassifierServer().start()
    pass

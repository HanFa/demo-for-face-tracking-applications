import pickle, shutil
import requests
import SocketServer
from threading import Thread, Condition
from SimpleHTTPServer import SimpleHTTPRequestHandler
from RemoteFaceClassifier import *
from RemoteFaceClassifier.Server import *
from RemoteFaceClassifier.Server.classifier import stateless_infer, stateful_infer, train
from RemoteFaceClassifier.Server.globals import frame_num, align_dir_cv


class ServerHandler(SimpleHTTPRequestHandler):

    def _set_response(self):
        self.send_response(code=200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

    def do_POST(self):
        self._set_response()
        content_length = int(self.headers['Content-Length'])  # <--- Gets the size of data
        post_data = self.rfile.read(content_length)  # <--- Gets the data itself
        frame_idx, img = pickle.loads(post_data)

        if SERVER_MODE == "Stateless":
            maxI_lst, predictions_lst, bb_lst = stateless_infer(img, SERVER_STATELESS)
        elif SERVER_MODE == "Stateful":
            maxI_lst, predictions_lst, bb_lst = stateful_infer(img, SERVER_STATEFUL, frame_idx)
        else:
            raise Exception("Invalid SERVER_MODE {}: ['Stateful', 'Stateless']".format(SERVER_MODE))

        # convert boundaries from dlib.rectanngle to corner points
        bb_bl_lst = []
        bb_tr_lst = []
        for bb in bb_lst:
            bb_bl_lst.append((bb.bl_corner().x, bb.bl_corner().y))
            bb_tr_lst.append((bb.tr_corner().x, bb.tr_corner().y))

        requests.post("http://{}:{}".format(CLIENT, str(CLIENT_RES_PORT)),
                      data=pickle.dumps((frame_idx, maxI_lst, predictions_lst, bb_bl_lst, bb_tr_lst)))

        global frame_num
        frame_num += 1


class FaceClassifierServer:

    def update_stateful_model(self):

        while True:
            # Trigger an update of stateful model every n frames
            align_dir_cv.acquire()

            global frame_num
            while frame_num == 0 or frame_num % SERVER_UPDATE_FREQUENCY != 0:
                align_dir_cv.wait()

            # Get the reps
            os.system(
                "{} -model {} -outDir {} -data {}".format(
                    os.path.join(fileDir, "batch-represent", "main.lua"),
                    SERVER_OPENFACE_MODEL, SERVER_REPS_DIR, SERVER_ALIGN_DIR)
            )

            shutil.rmtree(SERVER_ALIGN_DIR)

            # Append the reps to model
            for csv_file in ["labels.csv", "reps.csv"]:
                with open(os.path.join(os.path.dirname(SERVER_STATEFUL), csv_file), 'a') as main:
                    with open(os.path.join(SERVER_REPS_DIR, csv_file), 'r') as current:
                        main.writelines(current.readlines())

            # Refit the model
            train()
            align_dir_cv.release()


    def start(self):

        # Spawn another thread to update stateful model regularly if stateful
        if SERVER_MODE == "Stateful":
            t = Thread(target=self.update_stateful_model)
            t.daemon = True
            t.start()

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

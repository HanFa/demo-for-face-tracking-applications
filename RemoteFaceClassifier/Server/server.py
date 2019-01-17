import pickle, shutil, time
import requests
import SocketServer
from threading import Thread, Condition
from SimpleHTTPServer import SimpleHTTPRequestHandler
from RemoteFaceClassifier import *
from RemoteFaceClassifier.Server import *
from RemoteFaceClassifier.Server.classifier import stateless_infer, stateful_infer, train, full_search_face_boxes
import RemoteFaceClassifier.Server.globals as globals
from RemoteFaceClassifier.Server.profile import MEASURE_TYPE, profiler


class ServerHandler(SimpleHTTPRequestHandler):

    def _set_response(self):
        self.send_response(code=200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

    def do_POST(self):
        global profiler
        profiler.inform_transmission_time_start(MEASURE_TYPE.TOTAL)

        self._set_response()
        content_length = int(self.headers['Content-Length'])  # <--- Gets the size of data

        profiler.inform_transmission_time_start(MEASURE_TYPE.TRANSMISSION)
        post_data = self.rfile.read(content_length)  # <--- Gets the data itself
        profiler.inform_transmission_time_stop(MEASURE_TYPE.TRANSMISSION)

        frame_idx, img = pickle.loads(post_data)

        if SERVER_MODE == "Stateless":
            maxI_lst, predictions_lst, bb_lst, _ = stateless_infer(img, SERVER_STATELESS)
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

        # Return the results to client
        requests.post("http://{}:{}".format(CLIENT, str(CLIENT_RES_PORT)),
                      data=pickle.dumps((frame_idx, maxI_lst, predictions_lst, bb_bl_lst, bb_tr_lst)))

        profiler.inform_transmission_time_stop(MEASURE_TYPE.TOTAL)
        profiler.update_log()

        if globals.frame_num % 3 == 2:
            Thread(target=full_search_face_boxes, args=(img,)).start()  # full search asyncronously

        globals.frame_num += 1



class FaceClassifierServer:

    def update_stateful_model(self):

        while True:
            # Trigger an update of stateful model every 5 seconds
            time.sleep(5.0)

            # Get the reps
            if not os.path.exists(SERVER_ALIGN_DIR):
                continue

            os.system(
                "{} -model {} -outDir {} -data {}".format(
                    os.path.join(fileDir, "batch-represent", "main.lua"),
                    SERVER_OPENFACE_MODEL, SERVER_REPS_DIR, SERVER_ALIGN_DIR)
            )

            for temp_dir in os.listdir(SERVER_ALIGN_DIR):
                temp_dir_full = os.path.join(SERVER_ALIGN_DIR, temp_dir)
                if not os.path.isdir(temp_dir_full):
                    os.remove(temp_dir_full)
                else:
                    shutil.rmtree(temp_dir_full)
                    os.mkdir(temp_dir_full)

            # Append the reps to model
            for csv_file in ["labels.csv", "reps.csv"]:
                with open(os.path.join(os.path.dirname(SERVER_STATEFUL), csv_file), 'a') as main:
                    with open(os.path.join(SERVER_REPS_DIR, csv_file), 'r') as current:
                        main.writelines(current.readlines())

            # Refit the model
            train()


    def start(self):

        # Spawn additional threads for stateful server
        stateful_additional_threads = [self.update_stateful_model]

        if SERVER_MODE == "Stateful":
            # one thread to update stateful model regularly if stateful,
            for func in stateful_additional_threads:
                t = Thread(target=func)
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

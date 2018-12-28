import pickle, cv2
import requests
import SocketServer
from SimpleHTTPServer import SimpleHTTPRequestHandler
from RemoteFaceClassifier import *
from RemoteFaceClassifier.Server import *
from RemoteFaceClassifier.Server.classifier import stateless_infer, stateful_infer


class FaceClassifierServer:
    class ServerHandler(SimpleHTTPRequestHandler):
        def _set_response(self):
            self.send_response(code=200)
            self.send_header("Content-type", "text/html")
            self.end_headers()


        def do_POST(self):
            print("receive POST")
            self._set_response()
            content_length = int(self.headers['Content-Length'])  # <--- Gets the size of data
            post_data = self.rfile.read(content_length)  # <--- Gets the data itself
            frame_idx, img = pickle.loads(post_data)

            if SERVER_MODE == "Stateless":
                maxI_lst, predictions_lst, bb_lst = stateless_infer(img, SERVER_STATELESS)
            elif SERVER_MODE == "Stateful":
                maxI_lst, predictions_lst, bb_lst = stateful_infer(img, SERVER_STATEFUL)
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


    def start(self):
        SocketServer.TCPServer.allow_reuse_address = True
        httpd = SocketServer.TCPServer((SERVER, SERVER_FRAME_PORT), self.ServerHandler)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        httpd.server_close()


if __name__ == '__main__':
    FaceClassifierServer().start()
    pass

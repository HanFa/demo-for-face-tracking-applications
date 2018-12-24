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
                maxI, predictions = stateless_infer(img, SERVER_STATELESS)
            elif SERVER_MODE == "Stateful":
                maxI, predictions = stateful_infer(img, SERVER_STATEFUL)
            else:
                raise Exception("Invalid SERVER_MODE {}: ['Stateful', 'Stateless']".format(SERVER_MODE))

            requests.post("http://{}:{}".format(CLIENT, str(CLIENT_RES_PORT)), data=pickle.dumps((frame_idx, maxI, predictions)))


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

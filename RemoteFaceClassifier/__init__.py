from RemoteFaceClassifier import *
import os

# face classification client
CLIENT = "localhost"
CLIENT_VIDEO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "video" + os.sep + "test_video.mp4")
CLIENT_RES_PORT = 50001

# face classification server
SERVER = "localhost"
SERVER_FRAME_PORT = 50002

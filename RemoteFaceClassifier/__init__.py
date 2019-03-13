from RemoteFaceClassifier import *
import os

# face classification client
CLIENT = "0.0.0.0"
CLIENT_VIDEO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "video" + os.sep + "test_video.mp4")
CLIENT_RES_PORT = 30001

# face classification server
SERVER = "0.0.0.0"
SERVER_FRAME_PORT = 20001

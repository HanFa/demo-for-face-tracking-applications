from RemoteFaceClassifier import *
import os

# face classification client
CLIENT = "10.5.6.1"
CLIENT_VIDEO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "video" + os.sep + "test_video.mp4")
CLIENT_RES_PORT = 10001

# face classification server
SERVER = "10.5.6.101"
SERVER_FRAME_PORT = 50001

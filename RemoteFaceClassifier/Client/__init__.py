VIDEO_SAMPLE_RATE = 24
CLIENT_FRAME_INTERVAL = 1000 # time between sending each frame (ms)
CLASSIFY_THRESHOLD = 0.7 # label a face when the confidence is greater than this

# Front end configurations
SHOW_GUI = False
WINDOW_NAME = "Face tracking & classification demo\t Thres={}".format(CLASSIFY_THRESHOLD)
LABEL_COLOR = (255, 255, 0)
FACE_RECT_COLORS = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255)
}


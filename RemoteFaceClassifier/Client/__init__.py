VIDEO_SAMPLE_RATE = 4
CLASSIFY_THRESHOLD = 0.7 # label a face when the confidence is greater than this

# Front end configurations
SHOW_GUI = True
WINDOW_NAME = "Face tracking & classification demo\t Thres={}".format(CLASSIFY_THRESHOLD)
LABEL_COLOR = (255, 255, 0)
FACE_RECT_COLORS = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255)
}


import os
import openface

fileDir = os.path.dirname(os.path.realpath(__file__))

SERVER_VERBOSE = False
SERVER_IMG_DIM = 96
SERVER_CUDA = False

SERVER_MODE = "Stateful"
SERVER_UPDATE_FREQUENCY = 3 # Trigger an update of stateful model every n frames

SERVER_CLASSIFIER = "LinearSvm"

SERVER_DLIB_FACEPREDICTOR = os.path.join(fileDir, "FacePredictor", "shape_predictor_68_face_landmarks.dat")  # Path to dlib's face predictor
SERVER_OPENFACE_MODEL = os.path.join(fileDir, "Openface", "nn4.small2.v1.t7") # Opencface torch net model

SERVER_STATELESS = os.path.join(fileDir, "Stateless", "classifier.pkl")
SERVER_STATEFUL = os.path.join(fileDir, "Stateful", "classifier.pkl")

# Output folder for stateful
SERVER_RAW_DIR = os.path.join(fileDir, "raw")
SERVER_ALIGN_DIR = os.path.join(fileDir, "align")
SERVER_REPS_DIR = os.path.join(fileDir, "CurrentReps")

SERVER_MULT_FACE_INFER = True

align = openface.AlignDlib(SERVER_DLIB_FACEPREDICTOR)
net = openface.TorchNeuralNet(SERVER_OPENFACE_MODEL, imgDim=SERVER_IMG_DIM,
                              cuda=SERVER_CUDA)

# Output folder for performance measure
SERVER_PROFILE_ENABLE = True
SERVER_PROFILE_DIR = os.path.join(fileDir, 'Profile')

# Parallel computing optimization
SERVER_POOL_SIZE = 4
SERVER_FACE_SEARCH_OPTIMIZE = True
SERVER_FACE_SEARCH_PADDING = 1
import os
import openface
from OpenFacePytorch.loadOpenFace import prepareOpenFace

fileDir = os.path.dirname(os.path.realpath(__file__))

SERVER_IMG_DIM = 80
SERVER_CUDA = False

SERVER_MODE = "Stateful"
SERVER_DLIB_FACEPREDICTOR = os.path.join(fileDir, "FacePredictor", "shape_predictor_68_face_landmarks.dat")  # Path to dlib's face predictor
SERVER_OPENFACE_MODEL = os.path.join(fileDir, "Openface", "nn4.small2.v1.t7") # Opencface torch net model

SERVER_PRETRAINED = os.path.join(fileDir, "Pretrained", "classifier.pkl")
SERVER_MULT_FACE_INFER = True

align = openface.AlignDlib(SERVER_DLIB_FACEPREDICTOR)

# Output folder for performance measure
SERVER_PROFILE_ENABLE = True
SERVER_PROFILE_DIR = os.path.join(fileDir, 'Profile')

# Parallel computing optimization
SERVER_FACE_SEARCH_OPTIMIZE = True
SERVER_FACE_SEARCH_PADDING = 0.5

SERVER_USE_PYTORCH = False
if SERVER_USE_PYTORCH:
    net = prepareOpenFace(useCuda=False).eval()
else:
    net = openface.TorchNeuralNet(SERVER_OPENFACE_MODEL, imgDim=SERVER_IMG_DIM,
                                cuda=SERVER_CUDA)
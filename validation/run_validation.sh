#!/bin/bash
python ./classifier.py  --dlibFacePredictor shape_predictor_68_face_landmarks.dat \
--networkModel nn4.small2.v1.t7 \
infer ./stateless/classifier.pkl \
./validation/160112-obama-1101p_1d16238ca868f5d9b1eb70c950d8f03f.nbcnews-fp-1200-800.jpg

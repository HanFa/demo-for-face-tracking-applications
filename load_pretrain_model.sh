#!/bin/bash

if [ ! -d "./dataset" ] || [ ! -d "./dataset/raw" ]; then
	echo "Please make sure dataset exist in ./dataset/raw "
	echo "person-1
		├── image-1.jpg
		├── image-2.png
		...
		└── image-p.png

		...

	person-m
		├── image-1.png
		├── image-2.jpg
		...
		└── image-q.png"

	exit
fi

# clear the caches
find ./dataset -name "cache.t7" -type f -delete

# align the faces
python ./utils/align-dlib.py  --dlibFacePredictor './RemoteFaceClassifier/Server/FacePredictor/shape_predictor_68_face_landmarks.dat' ./dataset/raw align outerEyesAndNose ./dataset/align --size 96

# extract the features
./RemoteFaceClassifier/Server/batch-represent/main.lua -model './RemoteFaceClassifier/Server/Openface/nn4.small2.v1.t7' -outDir ./dataset/feature -data ./dataset/align

# fit the model
python ./utils/classifier.py --dlibFacePredictor './RemoteFaceClassifier/Server/FacePredictor/shape_predictor_68_face_landmarks.dat' train ./dataset/feature

set -x
echo 'load the model to stateless server'
rm -rf ./RemoteFaceClassifier/Server/Stateless/*
cp ./dataset/feature/* ./RemoteFaceClassifier/Server/Stateless/







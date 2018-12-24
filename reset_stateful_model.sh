#!/bin/bash
set -x
rm -rf RemoteFaceClassifier/Server/raw/
rm -rf RemoteFaceClassifier/Server/align/
rm -rf RemoteFaceClassifier/Server/CurrentReps/
rm -rf RemoteFaceClassifier/Server/Stateful/
mkdir RemoteFaceClassifier/Server/Stateful/
cp -r RemoteFaceClassifier/Server/Stateless/* RemoteFaceClassifier/Server/Stateful/
echo "Reset the stateful model to its original pretrained states."

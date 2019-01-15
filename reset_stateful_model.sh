#!/bin/bash
set -x
rm -rf RemoteFaceClassifier/Server/raw/
rm -rf RemoteFaceClassifier/Server/align/
rm -rf RemoteFaceClassifier/Server/CurrentReps/
rm -rf RemoteFaceClassifier/Server/Stateful/
rm -rf RemoteFaceClassifier/Server/Profile/
mkdir RemoteFaceClassifier/Server/Stateful/
pkill -f run_server.sh
pkill -f server.py
cp -r RemoteFaceClassifier/Server/Stateless/* RemoteFaceClassifier/Server/Stateful/
chmod 777 RemoteFaceClassifier/Server/Stateful/*
echo "Reset the stateful model to its original pretrained states."

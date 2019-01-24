#!/bin/bash
set -x
rm -rf RemoteFaceClassifier/Server/Profile/
pkill -f run_server.sh
pkill -f server.py
echo "Reset Done."

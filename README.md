# Remote client-server architecture demo based on CMU face classification application
## Current Progress
Client will send a frame every `CLIENT_FRAME_INTERVAL` which is configurable at `./RemoteFaceClassifier/Client/__init__.py`. 
It will has another thread receiving results.

Both version has been implemented.

The stateless model is currently running with a pre-trained model `RemoteFaceClassifier\Stateless\classifier.pkl`, while the stateful model updates the pickle `RemoteFaceClassifier\Stateful\classifier.pkl` at every incoming frames. 
 

## Structure of the project
```
.
├── README.md
└── RemoteFaceClassifier
    ├── Client
    │   ├── __init__.py
    │   └── client.py
    ├── Server
    │   ├── FacePredictor
    │   │   ├── mean.csv
    │   │   ├── shape_predictor_68_face_landmarks.dat
    │   │   └── std.csv
    │   ├── Openface
    │   │   ├── celeb-classifier.nn4.small2.v1.pkl
    │   │   ├── nn2.def.lua
    │   │   ├── nn4.def.lua
    │   │   ├── nn4.small1.def.lua
    │   │   ├── nn4.small2.def.lua
    │   │   ├── nn4.small2.v1.t7
    │   │   ├── resnet1.def.lua
    │   │   ├── vgg-face.def.lua
    │   │   └── vgg-face.small1.def.lua
    │   ├── Stateless
    │   │   └── classifier.pkl
    │   ├── WorkDir
    │   ├── __init__.py
    │   ├── classifier.py
    │   └── server.py
    ├── __init__.py
    └── video
        └── test_video.mp4

8 directories, 21 files

```
## run the client

```bash
./run_client.sh
```

## run the server
```bash
./run_server.sh
```

## reset the stateful
This script will reset the stateful server to its initial state, which is identical with the pretrained stateless model.
```bash
./reset_stateful_model.sh
```

## update the pre-trained dataset
You should make sure the dataset exists in the folder `dataset/raw`, in the following format
```bash
raw
├── joe
│   ├── 1.jpg
│   ├── 2.jpg
│   ├── 3.jpg
│   ├── 4.jpg
│   ├── 5.jpg
│   └── 6.jpg
└── obama
    ├── 1.jpg
    ├── 2.jpg
    ├── 3.jpg
    ├── 4.jpg
    ├── 5.jpg
    └── 6.jpg

2 directories, 12 files
```
Then run the script `load_pretrain_model.sh`. This script will retrain both stateful and stateless models, with the dataset in `raw` folder.

## toggle between stateful/stateless
Change the configuration `SERVER_MODE = "Stateful"` or `SERVER_MODE = "Stateless"` in [./RemoteFaceClassifier/Server/__init__.py](./RemoteFaceClassifier/Server/__init__.py).

## demo
The demo video url is [DEMO_URL.txt](DEMO_URL.txt).
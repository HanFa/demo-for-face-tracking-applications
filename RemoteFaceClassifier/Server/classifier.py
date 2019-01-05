#!/usr/bin/env python2
#
# Example to classify faces.
# Brandon Amos
# 2015/10/11
#
# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

start = time.time()

import argparse
import cv2
import os
import pickle
import sys

from operator import itemgetter

import numpy as np

np.set_printoptions(precision=2)
import pandas as pd

import openface

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from RemoteFaceClassifier.Server import *
from RemoteFaceClassifier.Server.aligndlib import alignMain
from PIL import Image
import shutil


def getRep(img_array, multiple=False):
    """Return the representations and boundaries for each person."""
    start = time.time()
    rgbImg = img_array

    # rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    if SERVER_VERBOSE:
        print("  + Original size: {}".format(rgbImg.shape))
    if SERVER_VERBOSE:
        print("Loading the image took {} seconds.".format(time.time() - start))

    start = time.time()

    if multiple:
        bbs = align.getAllFaceBoundingBoxes(rgbImg)
    else:
        bb1 = align.getLargestFaceBoundingBox(rgbImg)
        bbs = [bb1]
    if len(bbs) == 0 or (not multiple and bb1 is None):
        print("Unable to find a face")
        return []
    if SERVER_VERBOSE:
        print("Face detection took {} seconds.".format(time.time() - start))

    reps = []
    for bb in bbs:
        start = time.time()
        alignedFace = align.align(
            SERVER_IMG_DIM,
            rgbImg,
            bb,
            landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if alignedFace is None:
            raise Exception("Unable to align image")
        if SERVER_VERBOSE:
            print("Alignment took {} seconds.".format(time.time() - start))
            print("This bbox is centered at {}, {}".format(bb.center().x, bb.center().y))

        start = time.time()
        rep = net.forward(alignedFace)
        if SERVER_VERBOSE:
            print("Neural network forward pass took {} seconds.".format(
                time.time() - start))
        reps.append((bb.center().x, rep, bb))
    sreps = sorted(reps, key=lambda x: x[0])
    return sreps


def train():
    fname = "{}/labels.csv".format(os.path.dirname(SERVER_STATEFUL))
    labels = pd.read_csv(fname, header=None).as_matrix()[:, 0]
    print(labels)
    fname = "{}/reps.csv".format(os.path.dirname(SERVER_STATEFUL))
    embeddings = pd.read_csv(fname, header=None).as_matrix()
    print("LABEL:" + str(labels))
    le = LabelEncoder().fit(labels)
    labelsNum = le.transform(labels)
    nClasses = len(le.classes_)
    print("LE:" + str(le))

    print("Training for {} classes.".format(nClasses))

    if SERVER_CLASSIFIER == 'LinearSvm':
        clf = SVC(C=1, kernel='linear', probability=True)

    clf.fit(embeddings, labelsNum)

    print("Saving classifier to '{}'".format(SERVER_STATEFUL))
    with open(SERVER_STATEFUL, 'w') as f:
        pickle.dump((le, clf), f)


def stateful_infer(img_array, stateful_model):
    # Predict using current model
    maxI_lst, predictions_lst, bb_lst = stateless_infer(img_array, stateful_model)

    # Dump the image
    if os.path.exists(SERVER_ALIGN_DIR):
        shutil.rmtree(SERVER_ALIGN_DIR)

    os.mkdir(SERVER_ALIGN_DIR)

    if len(bb_lst) == 0: return maxI_lst, predictions_lst, bb_lst

    for idx, bb in enumerate(bb_lst):
        img_sub_array = img_array[bb.top() : bb.bottom(), bb.left() : bb.right()]

        maxI = str(maxI_lst[idx])
        if not os.path.exists(os.path.join(SERVER_ALIGN_DIR, maxI)):
            os.mkdir(os.path.join(SERVER_ALIGN_DIR, maxI))

        Image.fromarray(cv2.cvtColor(img_sub_array, cv2.COLOR_BGR2RGB)).save(os.path.join(SERVER_ALIGN_DIR, maxI, 'temp.png'))

    # Get the reps
    os.system(
        "{} -model {} -outDir {} -data {}".format(os.path.join(fileDir, "batch-represent", "main.lua"), SERVER_OPENFACE_MODEL, SERVER_REPS_DIR, SERVER_ALIGN_DIR)
    )

    # Append the reps to model
    for csv_file in ["labels.csv", "reps.csv"]:
        with open(os.path.join(os.path.dirname(SERVER_STATEFUL), csv_file), 'a') as main:
            with open(os.path.join(SERVER_REPS_DIR, csv_file), 'r') as current:
                main.writelines(current.readlines())

    # Refit the model
    train()

    return maxI_lst, predictions_lst, bb_lst


def stateless_infer(img_array, model):
    with open(model, 'rb') as f:
        if sys.version_info[0] < 3:
            (le, clf) = pickle.load(f)
        else:
            (le, clf) = pickle.load(f, encoding='latin1')

    print("\n=== Stateless {} ===")
    reps = getRep(img_array, SERVER_MULT_FACE_INFER)
    print("reps : {}".format(reps))

    maxI_lst = []
    predictions_lst = []
    bb_lst = []

    if len(reps) > 1:
        print("List of faces in image from left to right")
    for r in reps:
        rep = r[1].reshape(1, -1)
        predictions = clf.predict_proba(rep).ravel()
        maxI = np.argmax(predictions)

        maxI_lst.append(maxI)
        predictions_lst.append(predictions)
        bb_lst.append(r[2])

    return maxI_lst, predictions_lst, bb_lst


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode', help="Mode")
    trainParser = subparsers.add_parser('train',
                                        help="Train a new classifier.")
    trainParser.add_argument('--ldaDim', type=int, default=-1)
    trainParser.add_argument(
        '--classifier',
        type=str,
        choices=[
            'LinearSvm',
            'GridSearchSvm',
            'GMM',
            'RadialSvm',
            'DecisionTree',
            'GaussianNB',
            'DBN'],
        help='The type of classifier to use.',
        default='LinearSvm')
    trainParser.add_argument(
        'workDir',
        type=str,
        help="The input work directory containing 'reps.csv' and 'labels.csv'. Obtained from aligning a directory with 'align-dlib' and getting the representations with 'batch-represent'.")

    inferParser = subparsers.add_parser(
        'infer', help='Predict who an image contains from a trained classifier.')
    inferParser.add_argument(
        'classifierModel',
        type=str,
        help='The Python pickle representing the classifier. This is NOT the Torch network model, which can be set with --networkModel.')
    inferParser.add_argument('imgs', type=str, nargs='+',
                             help="Input image.")
    inferParser.add_argument('--multi', help="Infer multiple faces in image",
                             action="store_true")

    if SERVER_VERBOSE:
        print("Argument parsing and import libraries took {} seconds.".format(
            time.time() - start))

    start = time.time()

    if SERVER_VERBOSE:
        print("Loading the dlib and OpenFace models took {} seconds.".format(
            time.time() - start))
        start = time.time()

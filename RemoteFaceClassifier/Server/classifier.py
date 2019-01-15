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

import numpy as np
from threading import Thread

np.set_printoptions(precision=2)
import pandas as pd
import dlib

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from PIL import Image

from RemoteFaceClassifier.Server import *
from RemoteFaceClassifier.Server.globals import *
from RemoteFaceClassifier.Server.profile import MEASURE_TYPE, profiler

prev_bbs = None # Optimization for finding bounding boxes, stateful model only


def extend_boundary_box(img_array, bb):
    """ Extend the face boundary box. Provide a reasonable region to search faces for next frame.
        @bb: dlib.rectangle
        @crop_img:
    """
    padding = SERVER_FACE_SEARCH_PADDING
    crop_img = img_array[
               max(bb.top() - int(bb.height() * padding), 0): bb.bottom() + int(bb.height() * padding),
               max(bb.left() - int(bb.width() * padding), 0): bb.right() + int(bb.width() * padding)]

    # Debug use snippet
    # cv2.imshow('debug', crop_img)
    # cv2.waitKey(0)

    return crop_img


def get_face_boxes(rgbImg, prev_bb, bbs_out):
    """A wrapper function for align.getAllFaceBoundingBoxes """
    res = align.getAllFaceBoundingBoxes(rgbImg)
    local_left = res[0].left()
    local_right = res[0].right()
    local_top = res[0].top()
    local_bottom = res[0].bottom()

    padding = SERVER_FACE_SEARCH_PADDING
    anchor_point = dlib.point(y=max(prev_bb.top() - int(prev_bb.height() * padding), 0),
                              x=max(prev_bb.left() - int(prev_bb.width() * padding), 0))
    if len(res) > 0:
        bbs_out.append(dlib.rectangle(left=local_left + anchor_point.x,
                                      right=local_right + anchor_point.x,
                                      top=local_top + anchor_point.y,
                                      bottom=local_bottom + anchor_point.y))
    return


def getRep(rgbImg, multiple=False):
    """Return the representations and boundaries for each person."""

    global profiler, prev_bbs

    profiler.inform_transmission_time_start(MEASURE_TYPE.LOCATE)
    if multiple:
        if prev_bbs and len(prev_bbs) != 0 and SERVER_FACE_SEARCH_OPTIMIZE: # face search optimization
            threads = []
            bbs = []
            for prev_bb in prev_bbs:
                t = Thread(target=get_face_boxes, args=(extend_boundary_box(rgbImg, prev_bb), prev_bb, bbs))
                threads.append(t)

            for t in threads:
                t.start()

            for t in threads:
                t.join()

        else:
            bbs = [bb for bb in align.getAllFaceBoundingBoxes(rgbImg)]

    else:
        bb1 = align.getLargestFaceBoundingBox(rgbImg)
        bbs = [bb1]


    prev_bbs = bbs
    profiler.inform_transmission_time_stop(MEASURE_TYPE.LOCATE)

    if len(bbs) == 0 or (not multiple and bb1 is None):
        print("Unable to find a face")
        return []

    profiler.inform_transmission_time_start(MEASURE_TYPE.CLASSIFY)
    reps = []
    for bb in bbs:
        alignedFace = align.align(
            SERVER_IMG_DIM,
            rgbImg,
            bb,
            landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

        if alignedFace is None:
            raise Exception("Unable to align image")

        rep = net.forward(alignedFace)
        reps.append((bb.center().x, rep, bb))

    profiler.inform_transmission_time_stop(MEASURE_TYPE.CLASSIFY)

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


def stateful_infer(img_array, stateful_model, frame_idx):
    # Predict using current model
    maxI_lst, predictions_lst, bb_lst = stateless_infer(img_array, stateful_model)

    # Dump the image
    if not os.path.exists(SERVER_ALIGN_DIR):
        os.mkdir(SERVER_ALIGN_DIR)

    # Add the faces in current frame into the additional dataset
    for idx, bb in enumerate(bb_lst):
        print bb
        img_sub_array = img_array[bb.top() : bb.bottom(), bb.left() : bb.right()]

        maxI = str(maxI_lst[idx])
        if not os.path.exists(os.path.join(SERVER_ALIGN_DIR, maxI)):
            os.mkdir(os.path.join(SERVER_ALIGN_DIR, maxI))

        Image.fromarray(cv2.cvtColor(img_sub_array, cv2.COLOR_BGR2RGB)).save(
            os.path.join(SERVER_ALIGN_DIR, maxI, str(frame_idx) + '.png')
        )

    return maxI_lst, predictions_lst, bb_lst


def stateless_infer(img_array, model):
    with open(model, 'rb') as f:
        if sys.version_info[0] < 3:
            (le, clf) = pickle.load(f)
        else:
            (le, clf) = pickle.load(f, encoding='latin1')

    reps = getRep(img_array, SERVER_MULT_FACE_INFER)

    maxI_lst = []
    predictions_lst = []
    bb_lst = []

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

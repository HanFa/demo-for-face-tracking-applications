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
import torch

from PIL import Image
from RemoteFaceClassifier.Server import *
from RemoteFaceClassifier.Server.profile import MEASURE_TYPE, profiler

prev_bbs = None # Optimization for finding bounding boxes, stateful model only
real_bbs = None


def extend_boundary_box(img_array, bb):
    """ Extend the face boundary box. Provide a reasonable region to search faces for next frame.
        @bb: dlib.rectangle
        @crop_img:
    """
    padding = SERVER_FACE_SEARCH_PADDING
    crop_img = img_array[
               max(bb.top() - int(bb.height() * padding), 0): bb.bottom() + int(bb.height() * padding),
               max(bb.left() - int(bb.width() * padding), 0): bb.right() + int(bb.width() * padding)]

    return crop_img


def get_face_boxes(rgbImg, prev_bb, bbs_out):
    """A wrapper function for align.getAllFaceBoundingBoxes """
    res = align.getAllFaceBoundingBoxes(rgbImg)
    if not res:
        return

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


def align_faces(rgbImg, bb, aligned_faces_out):
    """A wrapper function for align and net.forward"""
    alignedFace = align.align(
        SERVER_IMG_DIM,
        rgbImg,
        bb,
        landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

    if alignedFace is None:
        raise Exception("Unable to align image")
        return

    aligned_faces_out.append((bb, alignedFace))


def full_search_face_boxes(rgbImg):
    global real_bbs
    real_bbs = [bb for bb in align.getAllFaceBoundingBoxes(rgbImg)]
    return


def forward_to_net(aligned_face):
    """Forward the aligned face image to net and retrieve face representations."""

    def preprocess(img):
        img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_LINEAR)
        img = np.transpose(img, (2, 0, 1))
        img = img.astype(np.float32) / 255.0
        I_ = torch.from_numpy(img).unsqueeze(0)
        return I_

    if not SERVER_USE_PYTORCH:
        return net.forward(aligned_face)
    else:
        rep = net(preprocess(aligned_face))
        return rep[0].detach().numpy()[0]


def getRep(rgbImg, multiple=False):
    """Return the representations and boundaries for each person."""

    global profiler, prev_bbs, real_bbs

    profiler.inform_transmission_time_start(MEASURE_TYPE.LOCATE)
    if multiple:
        if prev_bbs and len(prev_bbs) != 0 and SERVER_FACE_SEARCH_OPTIMIZE: # face search optimization
            threads = []
            bbs = []
            for prev_bb in prev_bbs:
                get_face_boxes(extend_boundary_box(rgbImg, prev_bb), prev_bb, bbs)
                # threads.append(Thread(target=get_face_boxes, args=(extend_boundary_box(rgbImg, prev_bb), prev_bb, bbs)))

            for t in threads: t.start()
            for t in threads: t.join()

        else:
            bbs = [bb for bb in align.getAllFaceBoundingBoxes(rgbImg)]

    else:
        bb1 = align.getLargestFaceBoundingBox(rgbImg)
        bbs = [bb1]


    if real_bbs and len(real_bbs) > len(bbs):
        prev_bbs = real_bbs
        real_bbs = None
    else:
        prev_bbs = bbs

    profiler.inform_transmission_time_stop(MEASURE_TYPE.LOCATE)

    if len(bbs) == 0 or (not multiple and bb1 is None):
        print("Unable to find a face")
        return []

    profiler.inform_transmission_time_start(MEASURE_TYPE.CLASSIFY)
    reps = []
    threads = []
    aligned_faces = []

    for bb in bbs:
        align_faces(rgbImg, bb, aligned_faces)
        # threads.append(Thread(target=align_faces, args=(rgbImg, bb, aligned_faces)))

    for t in threads: t.start()
    for t in threads: t.join()

    for bb, aligned_face in aligned_faces:
        rep = forward_to_net(aligned_face)
        reps.append((bb.center().x, rep, bb, aligned_face))

    profiler.inform_transmission_time_stop(MEASURE_TYPE.CLASSIFY)

    sreps = sorted(reps, key=lambda x: x[0])
    return sreps

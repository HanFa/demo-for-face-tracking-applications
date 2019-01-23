#!/usr/bin/env python3

import os, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from operator import sub

SAVE_DIR = 'results'

def cdf(array):
    num_bins = len(array)
    counts, bin_edges = np.histogram(array, bins=num_bins, normed=True)
    cdf = np.cumsum(counts) / sum(counts)
    return bin_edges[1:], cdf


if __name__ == '__main__':


    csv_file = 'RemoteFaceClassifier/Server/Profile/measure.csv'
    out_dir = 'plots'

    transmit, locate, classify, total = [], [], [], []
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    if not os.path.exists(csv_file):
        print "Cannot find RemoteFaceClassifier/Server/Profile/measure.csv"
        exit(1)

    with open(csv_file) as f:
        lines = f.readlines()

    labels = lines[0].split()
    for line in lines[1:]:
        transmit.append(float(line.split()[1]))
        locate.append(float(line.split()[2]))
        classify.append(float(line.split()[3]))
        total.append(float(line.split()[4]))

    plt.figure(0)
    measures_lst = [transmit, locate, classify, total]
    for idx, measures in enumerate(measures_lst):
        x, y = cdf(measures)
        plt.plot(x, y, label=labels[idx + 1])

    plt.legend()
    plt.xlabel("Time (ms)")
    plt.savefig(os.path.join(SAVE_DIR, 'latency_CDFs'))

    # bar plot
    plt.figure(1)
    x = range(len(lines) - 1)
    total_bar = plt.bar(x, total, color=(0,0,0))
    transmit_bar = plt.bar(x, transmit)
    locate_bar = plt.bar(x, locate, bottom=transmit)
    classify_bar = plt.bar(x, classify, bottom=locate)

    plt.legend((total_bar[0], transmit_bar[0], locate_bar[0], classify_bar[0]), ('Total', 'Transmit', 'Locate Faces', 'Classify Faces'))
    plt.savefig(os.path.join(SAVE_DIR, 'latency_bar'))

    # average plot
    plt.figure(2)
    transmit_avg = sum(transmit) / len(transmit)
    locate_avg = sum(locate) / len(locate)
    classify_avg = sum(classify) / len(classify)
    total_avg = sum(total) / len(total)

    plt.bar(range(4), [total_avg, transmit_avg, locate_avg, classify_avg])
    plt.xticks(range(4), ['total latency', 'transmit', 'locate faces', 'classify faces'])
    plt.savefig(os.path.join(SAVE_DIR, 'latency_avg'))

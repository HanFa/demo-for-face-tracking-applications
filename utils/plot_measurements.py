#!/usr/bin/env python3

import os, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

    print transmit

    measures_lst = [transmit, locate, classify, total]
    for idx, measures in enumerate(measures_lst):
        x, y = cdf(measures)
        plt.plot(x, y, label=labels[idx + 1])

    plt.legend()
    plt.xlabel("Time (ms)")
    plt.savefig('latency_CDFs')

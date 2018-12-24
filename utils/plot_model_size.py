#!/usr/bin/env python3
import os, sys
import glob
from matplotlib import pyplot as plt

if __name__ == '__main__':
    person_num = 1

    for person_idx in range(1, person_num + 1):
        feature_folders = sorted(glob.glob("./models/features*"), key=lambda x : int(x.split('-')[-1]))
        model_idxs = []
        model_sizes = []
        for feature_folder in feature_folders:
            model_idx = int(feature_folder.split('-')[-1])
            model_size = os.path.getsize(os.path.join(feature_folder, "classifier.pkl"))
            model_idxs.append(model_idx)
            model_sizes.append(model_size)
            print("idx is {} \t {} \t {}".format(str(model_idx), str(model_size), str(feature_folder)))

        plt.scatter(model_idxs, model_sizes, label="person-{}".format(person_idx), s=1)

    plt.legend()
    plt.xlabel("train frame num")
    plt.ylabel("model size (Bytes)")
    plt.title("Results of stateful model size")
    plt.savefig("model_size.png")

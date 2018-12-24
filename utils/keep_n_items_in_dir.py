#!/usr/bin/env python3
import sys, os
import argparse
import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Remove files until n left")
    parser.add_argument('--dir', default='.', dest='dir', help='Path to the folder where only n files will be kept')
    parser.add_argument('--n', type=int, default=1, dest='n', help="the number n, the number of files after in this folder")


    args = parser.parse_args()
    target_dir = os.path.join(os.getcwd(), args.dir)

    files = os.listdir(args.dir)
    while args.n < len(files):
        file = files[-1]
        files.remove(file)
        os.remove(os.path.join(args.dir, file))

        

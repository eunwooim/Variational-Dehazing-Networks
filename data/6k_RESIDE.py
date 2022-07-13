import argparse
import os

import cv2
import h5py
import numpy as np


def walk(folder):
    for dirpath, _, files in os.walk(folder):
        for filename in files:
            yield dirpath, filename

def load(path):
    data = []
    for folder, filename in walk(path):
        img = cv2.imread(f'{folder}/{filename}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data.append(path)
        print(f'Loaded {path}')
    return np.array(data).astype(np.uint8)

def main(args):
    folder, filename = os.path.split(args.h5_path)
    os.makedirs(folder, exist_ok=True)
    if 'out' in filename: mode = 'out'
    else: mode = 'in'
    if 'train' in filename: mode = mode[0] + 'ts'
    else: mode = f'sots/{mode}door'
    path = f'{args.root}/{mode}'

def main(args):
    os.makedirs(args.h5_path, exist_ok=True)
    for file in os.listdir(f'{args.root}/train'):
        load(file)
        ############################


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/Users/eunu/Downloads/RESIDE-full')
    parser.add_argument('--h5_path', type=str, default='/Users/eunu/reside/in_test.h5')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    main(args)
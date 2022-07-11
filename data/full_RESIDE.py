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

def main(args, dic):
    os.makedirs(args.h5_path, exist_ok=True)
    for dataset in ['its', 'ots', 'sots']:
        root = os.path.join(args.root, dataset)
        for dir in os.listdir(root):
            if not os.path.isdir(dir): continue
            dic[f'{dataset}_{dir}'] = load(f'{root}/{dataset}/{dir}')
    
    h5 = h5py.File(f'{args.h5_path}/full.h5', 'w')
    for key in dic.keys():
        h5.create_dataset(key, data=dic[key])
    h5.close()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/Users/eunu/Downloads/')
    parser.add_argument('--h5_path', type=str, default='/Users/eunu/reside')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    main(args, {})

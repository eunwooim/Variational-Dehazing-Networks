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
        data.append(img.astype(np.uint8))
        print(f'Loaded {path}')
    return np.array(data)

def chname(path, dir):
    if dir == 'clear': 
        for filename in os.listdir(f'{path}/{dir}'):
            name, ext = filename.split('.')
            os.rename(f'{path}/{dir}/{filename}',
                    f'{path}/{dir}/{name.zfill(4)}.{ext}')
    elif dir == 'hazy':
        for filename in os.listdir(f'{path}/{dir}'):
            name, idx, beta = filename.split('_')
            os.rename(f'{path}/{dir}/{filename}',
                    f'{path}/{dir}/{name.zfill(4)}_{idx}_{beta}')
    elif dir == 'trans':
        for filename in os.listdir(f'{path}/{dir}'):
            name, ext = filename.split('.')
            name, idx = name.split('_')
            os.rename(f'{path}/{dir}/{filename}',
                    f'{path}/{dir}/{name.zfill(4)}_{idx}_.{ext}')

def main(args, data={}):
    os.makedirs(args.h5_path[:-len(args.h5_path.split('/')[-1])], exist_ok=True)
    for dir in os.listdir(args.root):
        chname(args.root, dir)
        data[dir] = load(f'{args.root}/{dir}')

    h5 = h5py.File(f'{args.h5_path}', 'w')
    for key in data.keys():
        h5.create_dataset(key, data=data[key],
                            compression='gzip', compression_opts=9)
    h5.close()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/Users/eunu/Downloads/RESIDE-full')
    parser.add_argument('--h5_path', type=str, default='/Users/eunu/reside/in_train.h5')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    main(args)
import argparse
import os

import cv2
import h5py
import numpy as np


def walk(folder):
    for dirpath, _, files in os.walk(folder):
        for filename in files:
            yield dirpath, filename
            
def load_rgb(path):
    data = []
    for filename in sorted(os.listdir(path)):
        img = cv2.imread(f'{path}/{filename}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data.append(img.astype(np.uint8))
        print(f'Loaded {path}/{filename}')
    return np.array(data)

def load_gray(path):
    data = []
    for filename in sorted(os.listdir(path)):
        img = cv2.imread(f'{path}/{filename}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        data.append(img.astype(np.uint8))
        print(f'Loaded {path}/{filename}')
    return np.array(data)

def chname(path):
    if 'trans' in path:
        for filename in os.listdir(path):
            file, idx = filename.split('_')
            idx, ext = idx.split('.')
            os.rename(f'{path}/{filename}',
                      f'{path}/{file.zfill(4)}_{idx}_.{ext}')
            
    elif 'hazy' in path:
        for filename in os.listdir(path):
            file, idx = filename.split('_')[0], filename.split('_')[1:]
            os.rename(f'{path}/{filename}',
                      f"{path}/{file.zfill(4)}_{'_'.join(idx)}")
            
    elif 'clear' in path:
        for filename in os.listdir(path):
            file, ext = filename.split('.')
            os.rename(f'{path}/{filename}',
                      f'{path}/{file.zfill(4)}.{ext}')

def main(args, data=dict()):
    for dir in os.listdir(args.root):
        if dir == 'trans':
            array = load_gray(f'{args.root}/{dir}')
            data[dir] = array
        else:
            array = load_rgb(f'{args.root}/{dir}')
            data[dir] = array

    h5 = h5py.File(args.h5_path, 'w')
    for key in data.keys():
        h5.create_dataset(key, data=data[key])
    h5.close()
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/Users/eunu/Downloads/RESIDE-full')
    parser.add_argument('--h5_path', type=str, default='/Users/eunu/reside/in_train.h5')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    for path in os.listdir(args.root):
        path = f'{args.root}/{path}'
        chname(path)
    
    h5_path = f'{args.root}/in_train.h5'
    main(args)

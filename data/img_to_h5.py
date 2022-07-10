import argparse
import os

import cv2
import h5py
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='')
    parser.add_argument('--save_path', type=str, default='')

    return parser.parse_args()

if __name__ == '__main__':
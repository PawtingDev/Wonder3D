import os
import sys
import argparse
import math
import numpy as np
from utils import opengl_utils_icon

def gen_pose_icon(num_views):
    for y in range(0, 360, 360 // num_views):
        R = opengl_utils_icon.make_rotate(0, math.radians(y), 0)
        # T = -np.matmul(R, center).reshape(3, 1)
        # W2C = np.concatenate([R, T], axis=1)
        print(R)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_views', type=int, default='4')

    args = parser.parse_args()

    gen_pose_icon(args.num_views)

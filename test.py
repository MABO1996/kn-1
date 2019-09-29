import numpy as np
import h5py

# -*- coding: utf-8 -*-

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--square", help="display a square of a given number", type=int)
parser.add_argument("--cubic", help="display a cubic of a given number", type=int)

args = parser.parse_args()

if args.square:
    print( args.square**2)

if args.cubic:
    print( args.cubic**3)

# 读取h5数据
file=h5py.File('zz800_chg.h5','r')
# # print('h5 is done')
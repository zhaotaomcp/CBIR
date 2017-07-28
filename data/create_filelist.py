#! /usr/bin/env python
#coding=utf-8

# Authors: zhaotao
# create train and val txt

import sys
#import numpy as np
import os
#from scipy.sparse import csr_matrix
#import cPickle as pickle
import string
import random

def get_all_files(root_dir):

    test_list = []
    train_list = []
    for path, subdirs, files in os.walk(root_dir):
        for name in files:
            if name.endswith('.jpg'):
                namenum = string.atoi(name[5:8], base=10)
                labelnum = string.atoi(name[0:3], base=10)
                labelnum -= 1
                if namenum < 11:
                    test_list += [os.path.join(path, name) + ' ' + str(labelnum)]
                else:
                    train_list += [os.path.join(path, name) + ' ' + str(labelnum)]

    print "reading all data files"
    list2txt(train_list, 0)
    list2txt(test_list, 1)


def list2txt(filelist, t):
    random.shuffle(filelist)
    pre = "\n".join(filelist)

    print pre
    
    if t==0:
        f=open("train.txt","w")
    else:
        f=open("test.txt", "w")
    f.write(pre)
    f.close()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "usage: create_filelist.py [root_dir]"
        exit(1)

    root_dir = sys.argv[1]
    get_all_files(root_dir)



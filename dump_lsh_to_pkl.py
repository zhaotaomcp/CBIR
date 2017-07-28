#! /usr/bin/env python
#coding=utf-8

# use lsh on the 10 bit binary pre-compute image index feature

import sys
import numpy as np
import os
from scipy.sparse import csr_matrix
import cPickle as pickle
from lshash import LSHash
from bitarray import bitarray

def dump_lsh_data_to_pickle(bits_tid_pickle, lsh_pickle):
	f = file(bits_tid_pickle, "rb")
	data = pickle.load(f)
	f.close()

	# '10' means the bit binary (github.com/kayzh/LSHash)
	lsh = LSHash(13, 10, num_hashtables=1)
	map(lambda x:lsh.index(np.array([int(tmp) for tmp in x])), data.keys())	
	out = file(lsh_pickle,"wb")
	pickle.dump(lsh, out, -1)
	out.close()

if __name__ == '__main__':
	if len(sys.argv) != 3:
		print "usage: python dump_lsh_data_to_pickle.py [257 bits index fea file] [lsh_pickle_file]"
		exit(1)

	index_fea_pickle = sys.argv[1]
	lsh_pickle = sys.argv[2]
	dump_lsh_data_to_pickle(index_fea_pickle, lsh_pickle)

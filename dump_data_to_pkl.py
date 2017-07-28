#! /usr/bin/env python
#coding=utf-8

# Authors: zhao tao
# gather the pre-computed 4096*1 feature and 128/20*1 feature to a big file

import sys
import numpy as np
import os
from scipy.sparse import csr_matrix
import cPickle as pickle

def get_all_files(root_dir):
        file_list = []
        for path, subdirs, files in os.walk(root_dir):
                for name in files:
                        if name[-4:]==".pkl":
                                file_list += [os.path.join(path, name)]
        print "reading all data files done!"
        return file_list

def generate_dics_from_data(file_list, img_fea_file, index_fea_file):
        tid_feature_dic = {}
        bits_to_tids_dic = {}
        count = 0
        for bin_file in file_list:
                f = file(bin_file,"rb")
                while True:
                        try:
                                data_matrix = pickle.load(f)
                                count += 1
                                if count%20 == 0:
                                        print "Finished:"+str(count*50)+"image files....."
                        except:
                                break
                        for x in data_matrix:
                                tid = str(int(x[:,0].data))
                                tid_feature_dic[tid] = x[:,1:4097]
                                bit_str = "".join(map(lambda x:str(x), x[:,4097:].astype(int).toarray().flatten()))
                                try:
                                        bits_to_tids_dic[bit_str] += [tid]
                                except Exception, e:
                                        bits_to_tids_dic[bit_str] = [tid]
        img_fea_out = file(img_fea_file,"wb")
        index_fea_out = file(index_fea_file,"wb")
        pickle.dump(tid_feature_dic, img_fea_out, -1)
	img_fea_out.close()
    	pickle.dump(bits_to_tids_dic, index_fea_out, -1)
    	index_fea_out.close()



if __name__ == '__main__':

	if len(sys.argv) != 4:

		print "usage: python dump_data_dict_to_pickle.py [root_dir] [img_fea_file] [index_fea_file]"

		exit(1)

	root_dir = sys.argv[1]

	img_fea_file = sys.argv[2]

	index_fea_file = sys.argv[3]

	files = get_all_files(root_dir)

	generate_dics_from_data(files, img_fea_file, index_fea_file)


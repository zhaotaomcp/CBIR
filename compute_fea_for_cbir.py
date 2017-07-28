#! /usr/bin/env python
#coding=utf-8

# Authors: zhao tao
# compute image feature by pre-trained model

import sys
# change the following path to your compiled caffe python path
sys.path.append("/opt/down/caffe-master/python")
import caffe
import numpy as np
import os
from scipy.sparse import csr_matrix
import cPickle
import logging
import datetime

if __name__ == '__main__':
	if len(sys.argv) != 6:
		print "usage: python compute_fea_for_image_retrieval.py [img_name_file] [net_def_prototxt] [trained_net_caffemodel] [image_mean_file] [out_put dir]"
		exit(1)

	batchsize = 50
	img_file = sys.argv[1]

	logfile = "log_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
	logging.basicConfig(filename=logfile, level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

	# prototxt-->net_def_prototxt
	net_def_prototxt = sys.argv[2]
	# pre-trained model-->trained_net_caffemodel
	trained_net_caffemodel = sys.argv[3]
	# gpu mode is on
	caffe.set_mode_gpu()
	# cpu mode is on
	#caffe.set_mode_cpu()

	# setup the net by prototxt and model 
	net = caffe.Net(net_def_prototxt, trained_net_caffemodel, caffe.TEST)

	# setup transformer
	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_transpose('data', (2,0,1)) # height*width*channel -> channel*height*width

	# load mean file
	mean_file = sys.argv[4]
	mean_file = np.load(mean_file).mean(1).mean(1)
	transformer.set_mean('data', mean_file) #### subtract mean ####
	transformer.set_raw_scale('data', 255) # pixel value range
	transformer.set_channel_swap('data', (2,1,0)) # RGB -> BGR

        # setup batchsize to blob
	data_blob_shape = net.blobs['data'].data.shape
	data_blob_shape = list(data_blob_shape)
	net.blobs['data'].reshape(batchsize, data_blob_shape[1], data_blob_shape[2], data_blob_shape[3])
	img_list = []
	tid_list = []
        file_count = 0
	count = 0

	# setup the output file for feature --> out_put dir
	out_dir = sys.argv[5]
	for line in open(img_file):
		img, tag = line.strip().split(' ')
		imgname = img.split('/')[-1]
		tid = imgname[0:imgname.rindex(".")].replace('_', '')
		print tid

		try:
			print img
			img_list += [caffe.io.load_image(img)]
			tid_list += [int(tid)]
			count += 1
            		file_count += 1
		except:
			print 'fail!!!'
			continue

		if count == batchsize:
			print file_count
			count = 0
			# load image files as batchsize 
			net.blobs['data'].data[...] = map(lambda x: transformer.preprocess('data',x), img_list)
			# net forward
			net.forward()
			# get the fc7 layer 4096 features
			fc7_fea = net.blobs["fc7"].data[:]
			print "fc7_fea:"
			print fc7_fea.shape[0]
			print fc7_fea.shape[1]


			# get fc8_encode feature and convert to 0 or 1 
			fc8_fea = net.blobs["fc8_encode"].data[:]
			fc8_fea = (fc8_fea>=0.5)*1
			print "fc8_fea:"
			print fc8_fea.shape[0]
			print fc8_fea.shape[1]


			tid_arr = np.array([tid_list]).T

			# save the result by numpy
			result = np.hstack((tid_arr, fc7_fea, fc8_fea))
			print "result:"
			print result

			img_list = []
			tid_list = []
			# dump the features to pickle files each batch 
            		if file_count%batchsize == 0:
				print out_dir+"/"+img_file[-2:]+"_"+str(file_count/batchsize)+".pkl"
				f = open(out_dir+"/"+img_file[-2:]+"_"+str(file_count/batchsize)+".pkl",'wb')
				cPickle.dump(csr_matrix(result),f,-1)
				f.close()
				logging.info(img_file+str(file_count)+"files have been saved")
			else:
				continue

	#f.close()
	#out = np.packbits(output, axis=-1)
	#print out

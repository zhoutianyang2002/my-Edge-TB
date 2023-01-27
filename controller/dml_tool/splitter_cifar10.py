"""
this file loads cifar10 from keras and cut the data into several pieces in sequence.
我们的ctl内存买的太小了，只有2GB，平时可用的也就九百多M
所以这个文件运行不了
"""
import os

import numpy as np
from tensorflow.keras.datasets import cifar10

from splitter_utils import split_data, save_data

if __name__ == '__main__':
	# all configurable parameters.
	train_batch = 100
	train_drop_last = True
	test_batch = 100
	test_drop_last = True
	one_hot = False
	# all configurable parameters.

	# load data from keras.
	(train_images, train_labels), (test_images, test_labels) = cifar10.load_data ()
	len_train_data, len_test_data = len(train_images), len(test_images)
	print('size of train datset', len_train_data)
	print('size of test datset', len_test_data)

	# normalize.
	train_images, test_images = train_images / 255.0, test_images / 255.0
	# print(type(train_images[0][0][0][0])) #<class 'numpy.float64'>
	# convert to float32. 注意我们的ctl内存只有2G，这一步无法完成
	# train_images = train_images.astype (np.float32)
	test_images = test_images.astype (np.float32)

	# for i in range(len_train_data):
	# 	train_images[i] = train_images[i].astype (np.float32)
	# print('ok')
	# for i in range(len_test_data):
	# 	test_images[i] = test_images[i].astype (np.float32)
	# # save in here.
	# path = os.path.abspath (os.path.join (os.path.dirname (__file__), '../dataset2/CIFAR10'))
	# train_path = os.path.join (path, 'train_data')
	# test_path = os.path.join (path, 'test_data')

	# # split and save.
	# train_images_loader, train_labels_loader = split_data (train_images, train_labels, train_batch, train_drop_last)
	# save_data (train_images_loader, train_labels_loader, train_path, one_hot)

	# test_images_loader, test_labels_loader = split_data (test_images, test_labels, test_batch, test_drop_last)
	# save_data (test_images_loader, test_labels_loader, test_path, one_hot)
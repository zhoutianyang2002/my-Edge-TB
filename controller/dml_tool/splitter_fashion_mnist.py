"""
this file loads fashion_mnist from keras and cut the data into several pieces in sequence.
ReadME里面没有写，但是是需要先单独运行这个程序获取数据集。

这个程序是把Fashion Mnist数据集的6+1万张划分成若干个batch，然后以batch为单位发送给各个worker，
起到了一个“打包压缩”的作用。
否则一个文件一个文件的传，每传一次都要重新建立IO流，速度会非常慢
"""
import os

import numpy as np
from tensorflow.keras.datasets import fashion_mnist

from splitter_utils import split_data, save_data

if __name__ == '__main__':
	# all configurable parameters.
	train_batch = 100 # 把训练数据划分成100份，相当于每个文件包含600张照片
	train_drop_last = True
	test_batch = 100
	test_drop_last = True
	one_hot = False
	# all configurable parameters.

	# load data from keras.
	(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data ()
	# normalize.
	train_images, test_images = train_images / 255.0, test_images / 255.0 
	# convert to float32 from np.float64.
	train_images, test_images = train_images.astype (np.float32), test_images.astype (np.float32)

	# save in here.
	path = os.path.abspath (os.path.join (os.path.dirname (__file__), '../dataset/FASHION_MNIST'))
	train_path = os.path.join (path, 'train_data')
	test_path = os.path.join (path, 'test_data')

	# split and save.
	train_images_loader, train_labels_loader = split_data (train_images, train_labels, train_batch, train_drop_last)
	save_data (train_images_loader, train_labels_loader, train_path, one_hot)
	#print("save train data finish")
	test_images_loader, test_labels_loader = split_data (test_images, test_labels, test_batch, test_drop_last)
	save_data (test_images_loader, test_labels_loader, test_path, one_hot)
	#print("save test data finish")

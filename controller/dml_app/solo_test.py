"""
you can use this file to test your neural network models
and datasets on a single computer.
"""


import numpy as np
import os
import json

from nns.nn_fashion_mnist import nn

if __name__ == '__main__':
	conf = {}
	x_list = []
	y_list = []
	dirname = os.path.abspath (os.path.dirname (__file__))
	conf_path = os.path.join(dirname, 'dml_file/conf').replace('\\', '/')
	node = ['n'+str(i) for i in range(1, 6)]

	for node_name in node:
		print("-----------" + node_name + "-----------")
		node_path = os.path.join(conf_path, node_name + '_dataset.conf').replace('\\', '/')
		with open(node_path, 'r') as f:
			conf.update (json.loads (f.read()))
		dataset_path = os.path.join(dirname, 'dataset').replace('\\', '/')
		if node_name == 'n1':
			print('data size = ' + str(conf["test_len"]))
			test_images, test_labels = np.load (dataset_path + '/' + node_name + '_images.npy'), np.load (dataset_path + '/' + node_name + '_labels.npy')
			print('shape of test_images' + str(test_images.shape))
			print('shape of test_labels' + str(test_labels.shape))
		else :
			print('data size = ' + str(conf["train_len"]))
			x, y = np.load (dataset_path + '/' + node_name + '_images.npy'), np.load (dataset_path + '/' + node_name + '_labels.npy')
			print('shape of train_images' + str(x.shape))
			print('shape of train_labels' + str(y.shape))
			x_list.append(x)
			y_list.append(y)

	train_images = np.concatenate (tuple (x_list))
	train_labels = np.concatenate (tuple (y_list))
	print('shape of train_images' + str(train_images.shape))
	print('shape of train_labels' + str(train_labels.shape))

	nn.model.fit (train_images, train_labels, epochs=1, batch_size=32)
	nn.model.evaluate(test_images, test_labels)
import io
import time
import os
import tarfile
import math

import tensorflow as tf
from tensorflow import keras
import numpy as np

import worker_utils

write = io.BytesIO ()


def load_data (path, start_index, _len, input_shape):
	x_list = []
	y_list = []
	for i in range (_len):
		x_list.append (np.load (path + '/images_' + str (start_index + i) + '.npy')
			.reshape (input_shape))
		y_list.append (np.load (path + '/labels_' + str (start_index + i) + '.npy'))
	images = np.concatenate (tuple (x_list))
	labels = np.concatenate (tuple (y_list))
	return images, labels

def load_data_by_index (path, idxs, input_shape):
	x_list = []
	y_list = []
	_len = 100 # 注意灵活修改
	for i in range (_len):
		x_list.append (np.load (path + '/images_' + str (1+i) + '.npy').reshape (input_shape))
		y_list.append (np.load (path + '/labels_' + str (1+i) + '.npy'))
	images = np.concatenate (tuple (x_list))
	labels = np.concatenate (tuple (y_list))
	return images[idxs], labels[idxs]

# 1.平凡的训练过程（FedAvg）
def train (model, images, labels, epochs, batch_size):
	h = model.fit (images, labels, epochs=epochs, batch_size=batch_size)
	return h.history ['loss']

# 2. FedProx的训练
def train_on_FedProx(model, train_images, train_labels, epochs, batch_size, mu = 0.1):
	'''
	与FedAvg的区别在于计算loss时需要加上一个正则项。所以要手写sgd
	其实相比于model.fit大概要慢30多秒
	'''
	global_model = model.get_weights() # tpye: ndarray
	steps_per_epoch = math.ceil( 1.0 * len(train_images) / batch_size )
	optimizer = keras.optimizers.Adam()
	metric = keras.metrics.Accuracy()

	def random_batch(x, y, batch_size=32):
		idx = np.random.randint(0, len(x), size=batch_size)
		return x[idx], y[idx]

	for epoch in range(epochs):
		worker_utils.log(f'training : epoch/epochs:{epoch}/{epochs} start.')
		s = time.time()
		metric.reset_states()
		tot_loss = 0
		for step in range(steps_per_epoch):
			x_batch, y_batch = random_batch(train_images, train_labels, batch_size)
			with tf.GradientTape() as tape:
				y_pred = model(x_batch)
				loss = tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(y_true=y_batch, y_pred=y_pred, from_logits=True))
				# 计算正则项
				proximal_term = 0.0
				local_model = model.get_weights()
				for w, w_t in zip(local_model, global_model): # 返回类型为list，所以要遍历
					proximal_term += np.linalg.norm(w - w_t)
				loss += 0.5 * mu * proximal_term
				tot_loss += loss
			grads = tape.gradient(loss, model.trainable_variables)
			optimizer.apply_gradients(zip(grads, model.trainable_variables))
		cost_time = time.time() - s
		worker_utils.log(f'epoch/epochs:{epoch}/{epochs}: cost time = {cost_time} s, loss = {tot_loss / steps_per_epoch}')
	
	return tot_loss / steps_per_epoch

# 3. FedNova的训练
def train_on_FedNova (model, images, labels, epochs, batch_size):
	'''
	忽略了发送数据集大小和迭代次数的过程（因为通信量相对于模型权重微不足道）
	注意返回的是globel model 和 local model 的差
	深复制与浅复制的问题：试过，两次get_weights返回的id不同
	'''
	global_model = model.get_weights()
	h = model.fit (images, labels, epochs=epochs, batch_size=batch_size)
	local_model = model.get_weights()
	delta_w = global_model - local_model
	assert delta_w.shape == global_model.shape
	return h.history ['loss'][-1], delta_w


# 4. SCAFFOLD的训练
def train_on_SCAFFOLD(model, images, labels, epochs, batch_size):
	pass

def test (model, images, labels):
	loss, acc = model.test_on_batch (images, labels)
	return loss, acc


def test_on_batch (model, images, labels, batch_size):
	sample_number = images.shape [0]
	batch_number = sample_number // batch_size
	last = sample_number % batch_size
	total_loss, total_acc = 0.0, 0.0
	for i in range (batch_number):
		loss, acc = model.test_on_batch (images [i * batch_size:(i + 1) * batch_size],
			labels [i * batch_size:(i + 1) * batch_size])
		total_loss += loss * batch_size
		total_acc += acc * batch_size
	loss, acc = model.test_on_batch (images [batch_number * batch_size:],
		labels [batch_number * batch_size:])
	total_loss += loss * last
	total_acc += acc * last
	return total_loss / sample_number, total_acc / sample_number


def parse_weights (weights):
	w = np.load (weights, allow_pickle=True)
	return w

def parse_c (C):
	delta_c = np.load (C, allow_pickle=True)
	return delta_c


# only store the weights at received_weights [0]
# and accumulate as soon as new weights are received to save space :-)
def store_weights (received_weights, new_weights, received_count):
	if received_count == 1:
		received_weights.append (new_weights)
	else:
		received_weights [0] = np.add (received_weights [0], new_weights)


def avg_weights (received_weights, received_count):
	return np.divide (received_weights [0], received_count)

# 上面单纯的加起来是欠妥的，没有考虑数据划分不平衡的情况
def avg_weights_by_size (received_weights, sample_size):
	return np.divide (received_weights [0], sample_size)


def assign_weights (model, weights):
	model.set_weights (weights)


def send_weights (weights, path, node_list, connect, forward=None, layer=-1):
	self = 0
	np.save (write, weights)
	write.seek (0)
	for node in node_list:
		if node == 'self':
			self = 1
			continue
		if node in connect:
			addr = connect [node]
			data = {'path': path, 'layer': str (layer)}
			# print("aggregator send weights to " + addr)
			send_weights_helper (write, data, addr, is_forward=False, node_name=node)
		elif forward:
			addr = forward [node]
			data = {'node': node, 'path': path, 'layer': str (layer)}
			send_weights_helper (write, data, addr, is_forward=True, node_name=node)
		else:
			Exception ('has not connect to ' + node)
		write.seek (0)
	write.truncate ()
	return self


def send_weights_helper (weights, data, addr, is_forward, node_name=''):
	s = time.time ()
	if not is_forward:
		worker_utils.send_data ('POST', data ['path'], addr, data=data, files={'weights': weights})
	else:
		worker_utils.log ('need ' + addr + ' to forward to ' + data ['node'] + data ['path'])
		worker_utils.send_data ('POST', '/forward', addr, data=data, files={'weights': weights})
	e = time.time ()
	worker_utils.log ('send weights to ' + node_name + ', cost=' + str (e - s))


def random_selection (node_list, number):
	return np.random.choice (node_list, number, replace=False)


def log_loss (loss, _round):
	"""
	we left a comma at the end for easy positioning and extending.
	this message can be parse by controller/ctl_utils.py, parse_log_file ().
	"""
	message = 'Train: loss={}, round={},'.format (loss, _round)
	worker_utils.log (message)
	return message


def log_acc (acc, _round, layer=-1):
	"""
	we left a comma at the end for easy positioning and extending.
	this message can be parsed by controller/ctl_utils.py, parse_log_file ().
	"""
	if layer != -1:
		message = 'Aggregate: accuracy={}, round={}, layer={},'.format (acc, _round, layer)
	else:
		message = 'Aggregate: accuracy={}, round={},'.format (acc, _round)
	
	worker_utils.log (message,True,'TrainTime')
	return message

class Timer: 
	"""记录多次运⾏时间"""
	def __init__(self):
		self.times = []
	def start(self):
		"""启动计时器"""
		self.tik = time.time()
	def stop(self):
		"""停⽌计时器并将时间记录在list中"""
		self.times.append(time.time() - self.tik)
		return self.times[-1]
	def printAggTimes(self):
		"""打印聚合信息"""
		outputs = [round(ele / 60.0, 2) for ele in self.times]
		worker_utils.log('cost time in each aggregation(unit:minute): ')
		worker_utils.log(str(outputs))
		avg_time = round(self.avg(), 2)
		worker_utils.log('average time: ' + str(avg_time) + 's')
		tot_time = round(self.sum(), 2)
		worker_utils.log('total time: ' + str(tot_time) + 's')
	def avg(self):
		"""返回平均时间"""
		return sum(self.times) / len(self.times) * 1.0
	def sum(self):
		"""返回时间总和"""
		return sum(self.times)
	def cumsum(self):
		"""返回累计时间"""
		return np.array(self.times).cumsum().tolist()


def decompress(tar_file_name, path=".", model="r:*"):
	"""
	将名为tar_file_name的压缩文件解压到path路径下
	"""
	with tarfile.open(tar_file_name, model) as tar_obj:
		tar_obj.extractall(path=path)
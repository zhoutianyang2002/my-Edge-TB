import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from flask import Flask, request

import dml_utils
import worker_utils
from nns.nn_fashion_mnist import nn  # configurable parameter, from nns.whatever import nn.

dirname = os.path.abspath (os.path.dirname (__file__))

dml_port = os.getenv ('DML_PORT')
ctl_addr = os.getenv ('NET_CTL_ADDRESS')
agent_addr = os.getenv ('NET_AGENT_ADDRESS')
node_name = os.getenv ('NET_NODE_NAME')

input_shape = nn.input_shape
_dir_path = os.path.join (dirname, '../dml_file/log/')
if (not os.path.exists(_dir_path)):
	os.mkdir(_dir_path)
log_file = os.path.abspath (os.path.join (dirname, '../dml_file/log/', node_name + '.log'))
worker_utils.set_log (log_file)
conf = {}
peer_list = []
# configurable parameter, specify the dataset path.
train_path = os.path.join (dirname, '../dataset/FASHION_MNIST/train_data')
train_images: np.ndarray
train_labels: np.ndarray
# configurable parameter, specify the dataset path.
test_path = os.path.join (dirname, '../dataset/FASHION_MNIST/test_data')
test_images: np.ndarray
test_labels: np.ndarray

app = Flask (__name__)
lock = threading.RLock ()
executor = ThreadPoolExecutor (1)


# if this is container, docker will send a GET to here every 30s
# this ability is defined in controller/base/node.py, Class Emulator, save_yml (), healthcheck.
@app.route ('/hi', methods=['GET'])
def route_hi ():
	# send a heartbeat to the agent.
	# when the agent receives the heartbeat of a container for the first time,
	# it will deploy the container's tc settings.
	# please ensure that your app implements this function, i.e.,
	# receiving docker healthcheck and sending heartbeat to the agent.
	worker_utils.heartbeat (agent_addr, node_name)
	return 'this is node ' + node_name + '\n'


@app.route ('/conf/dataset', methods=['POST'])
def route_conf_d ():
	f = request.files.get ('conf').read ()
	conf.update (json.loads (f))
	worker_utils.log ('POST at /conf/dataset')

	#1.保存conf文件
	worker_name = 'worker_' + node_name
	filename = os.path.join (dirname, '../dml_file/conf', node_name + '_dataset.conf')
	with open (filename, 'w') as fw:
		fw.writelines (json.dumps (conf, indent=2))
	worker_utils.log('save conf finish')

	#2.判断是否使用worker本地数据集
	global train_images, train_labels
	global test_images, test_labels
	useLocalData = conf['useLocalData']
	worker_utils.log('useLocalData = ' + useLocalData)
	
	## 使用本地数据集：
	if useLocalData == 'True':
		# 注意这里路径是包含数据集名称的硬编码
		worker_utils.log('start load data')
		try:
			local_dataset_path_train = os.path.join(dirname, "../local_dataset/FASHION_MNIST/train_data")
			local_dataset_path_test = os.path.join(dirname, "../local_dataset/FASHION_MNIST/test_data")
			test_images, test_labels = \
			dml_utils.load_data (local_dataset_path_test, conf ['test_start_index'], conf ['test_len'], input_shape)
			train_images, train_labels = \
			dml_utils.load_data (local_dataset_path_train, conf ['train_start_index'], conf ['train_len'], input_shape)
			worker_utils.log('load data finish')
			worker_utils.log('num of test images = ' + str(len(test_images)))
		except Exception as e:
			worker_utils.log(e)
			worker_utils.log(e.with_traceback())
			print(e)
			print(e.with_traceback())
		return ''
	
	## 不使用本地数据集：
	# 1.创建mydataset文件夹
	ctl_dataset_path = os.path.join(dirname, "../dataset/")
	my_dataset_train_path = os.path.join(dirname, "../my_dataset/train_data")
	my_dataset_test_path = os.path.join(dirname, "../my_dataset/test_data")
	if os.path.exists(my_dataset_train_path) == False:
		os.makedirs(my_dataset_train_path) # makedirs可以递归创建目录
	if os.path.exists(my_dataset_test_path) == False:
		os.makedirs(my_dataset_test_path)

	# 2.解压
	print("解压中")
	print("dataset contains: ",os.listdir(ctl_dataset_path))
	tar_file_name = os.path.join(ctl_dataset_path, worker_name + '_train.tar') 
	dml_utils.decompress(tar_file_name, path=my_dataset_train_path)
	tar_file_name = os.path.join(ctl_dataset_path, worker_name + '_test.tar') 
	dml_utils.decompress(tar_file_name, path=my_dataset_test_path)
	print("解压完")

	# 3.load
	print("加载中")
	train_images, train_labels = dml_utils.load_data (my_dataset_train_path, conf ['train_start_index'],
		conf ['train_len'], input_shape)
	test_images, test_labels = dml_utils.load_data (my_dataset_test_path, conf ['test_start_index'],
		conf ['test_len'], input_shape)
	print('num of train images and labels:')
	print(len(train_images),len(train_labels))

	return ''
	# f = request.files.get ('conf').read ()
	# conf.update (json.loads (f))
	# print ('POST at /conf/dataset')

	# global train_images, train_labels
	# train_images, train_labels = dml_utils.load_data (train_path, conf ['train_start_index'],
	# 	conf ['train_len'], input_shape)
	# global test_images, test_labels
	# test_images, test_labels = dml_utils.load_data (test_path, conf ['test_start_index'],
	# 	conf ['test_len'], input_shape)

	# filename = os.path.join (dirname, '../dml_file/conf', node_name + '_dataset.conf')
	# with open (filename, 'w') as fw:
	# 	fw.writelines (json.dumps (conf, indent=2))
	# return ''


@app.route ('/conf/structure', methods=['POST'])
def route_conf_s ():
	f = request.files.get ('conf').read ()
	conf.update (json.loads (f))
	print ('POST at /conf/structure')

	filename = os.path.join (dirname, '../dml_file/conf', node_name + '_structure.conf')
	with open (filename, 'w') as fw:
		fw.writelines (json.dumps (conf, indent=2))

	conf ['current_round'] = 0
	peer_list.extend (list (conf ['connect'].keys ()))
	return ''


@app.route ('/log', methods=['GET'])
def route_log ():
	executor.submit (on_route_log)
	return ''


def on_route_log ():
	worker_utils.send_log (ctl_addr, log_file, node_name)


@app.route ('/start', methods=['GET'])
def route_start ():
	print ('GET at /start')
	worker_utils.reset_time()
	worker_utils.log('At Start.',True,time_text='StartTime')
	executor.submit (on_route_start)
	return ''


def on_route_start ():
	_, init_acc = dml_utils.test_on_batch (nn.model, test_images, test_labels, conf ['batch_size'])
	msg = dml_utils.log_acc (init_acc, 0)
	worker_utils.send_print (ctl_addr, node_name + ': ' + msg)

	with lock:
		gossip ()


def gossip ():
	peer = dml_utils.random_selection (peer_list, 1)
	worker_utils.log ('gossip to ' + peer [0])
	dml_utils.send_weights (nn.model.get_weights (), '/gossip', peer, conf ['connect'])


@app.route ('/gossip', methods=['POST'])
def route_gossip ():
	print ('POST at /gossip')
	weights = dml_utils.parse_weights (request.files.get ('weights'))
	worker_utils.log ('Get Gossip',True)
	executor.submit (on_route_gossip, weights)
	return ''


def on_route_gossip (received_weights):
	with lock:
		new_weights = np.add (nn.model.get_weights (), received_weights) / 2
		dml_utils.assign_weights (nn.model, new_weights)

		conf ['current_round'] += 1
		loss_list = dml_utils.train (nn.model, train_images, train_labels, conf ['epoch'],
			conf ['batch_size'])
		last_epoch_loss = loss_list [-1]
		msg = dml_utils.log_loss (last_epoch_loss, conf ['current_round'])
		worker_utils.send_print (ctl_addr, node_name + ': ' + msg)

		_, acc = dml_utils.test_on_batch (nn.model, test_images, test_labels, conf ['batch_size'])
		msg = dml_utils.log_acc (acc, conf ['current_round'])
		worker_utils.send_print (ctl_addr, node_name + ': ' + msg)

		if conf ['current_round'] < conf ['sync']:
			gossip ()


app.run (host='0.0.0.0', port=dml_port, threaded=True)

import json
import os
import time
import math
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
log_file = os.path.abspath (os.path.join (dirname, '../dml_file/log/', node_name + '.log'))
worker_utils.set_log (log_file)
conf = {}
# configurable parameter, specify the dataset path.
train_path = os.path.join (dirname, '../dataset/FASHION_MNIST/train_data')
train_images: np.ndarray
train_labels: np.ndarray
local_c = 0

app = Flask (__name__)
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
	useLocalData = conf['useLocalData']
	
	## 使用本地数据集：
	if useLocalData == 'True':
		local_dataset_path = os.path.join(dirname, "../local_dataset/FASHION_MNIST/train_data")
		# local_dataset_path = os.path.join("/home/worker/local_dataset", datasetType, "train_data")
		worker_utils.log(local_dataset_path)
		train_images, train_labels = dml_utils.load_data_by_index (local_dataset_path, conf ["train_idxs"], input_shape)
		worker_utils.log('load data finish')
		worker_utils.log('the shape of train images = ' + str(train_images.shape))
		return ''
	
	## 不使用本地数据集：
	# 1.创建mydataset文件夹：
	# 	  在manager.py中，已经把每个worker的数据集打包成压缩包保存在dataset中，
	# 	  但是注意，worker对于挂载的dataset文件夹只有读权限，没有写权限，不可以直接在dataset文件夹下解压
	# 	  这就需要每个worker单独建一个mydataset文件夹保存解压的数据
	ctl_dataset_path = os.path.join(dirname, "../dataset/")
	my_dataset_path = os.path.join(dirname, "../my_dataset/")
	if os.path.exists(my_dataset_path) == False:
		os.makedirs(my_dataset_path)
   
	# 2.解压
	tar_file_name = os.path.join(ctl_dataset_path, worker_name + '.tar') 
	dml_utils.decompress(tar_file_name, path=my_dataset_path)

	# 3.load
	train_images, train_labels = dml_utils.load_data (my_dataset_path, conf ['train_start_index'],
		conf ['train_len'], input_shape)

	return ''


@app.route ('/conf/structure', methods=['POST'])
def route_conf_s ():
	f = request.files.get ('conf').read ()
	conf.update (json.loads (f))
	worker_utils.log ('POST at /conf/structure')

	# 这里我觉得写的有点问题，client保存在本地的nx_structure.conf不仅有structure信息，还包括dataset的信息
	filename = os.path.join (dirname, '../dml_file/conf', node_name + '_structure.conf')
	with open (filename, 'w') as fw:
		fw.writelines (json.dumps (conf, indent=2))

	conf ['current_round'] = 0

	worker_utils.log ('the fed algorithm is ' + conf['algorithm'])

	# for customized selection >>>
	# 不管是同步和异步，采样都是必须的
	executor.submit (perf_eval)
	# <<< for customized selection
	return ''


# for customized selection >>>

# 1.FedAvg, FedProx, FedNova, SCAFFOLD

def perf_eval (): # 意义在于获取prob_list
	worker_utils.log('worker is about to train')
	s = time.time ()
	dml_utils.train (nn.model, train_images, train_labels, 1, conf ['batch_size'])
	e = time.time () - s
	worker_utils.log('train finish')
	addr = conf ['connect'] [conf ['father_node']]
	path = '/time/train?node=' + node_name + '&time=' + str (e)
	worker_utils.log (node_name + ': train time=' + str (e))
	worker_utils.send_data ('GET', path, addr)

	s = time.time ()
	path = '/time/test'
	dml_utils.send_weights (nn.model.get_weights (), path, [conf ['father_node']],
		conf ['connect'])
	e = time.time () - s
	path = '/time/send?node=' + node_name + '&time=' + str (e)
	worker_utils.log (node_name + ': send time=' + str (e))
	worker_utils.send_data ('GET', path, addr)


# <<< for customized selection

@app.route ('/log', methods=['GET'])
def route_log ():
	executor.submit (on_route_log)
	return ''


def on_route_log ():
	worker_utils.send_log (ctl_addr, log_file, node_name)


@app.route ('/train', methods=['POST'])
def route_train ():
	worker_utils.log ('POST at /train')
	s_time = time.time()
	
	# 1. 读取权重
	weights = dml_utils.parse_weights (request.files.get ('weights'))
	# 2. 判断算法
	if conf['algorithm'] in ['FedAvg']:
		executor.submit (on_route_train, weights)
	elif conf['algorithm'] in ['FedNova']:
		executor.submit (train_FedNova_method, weights)
	elif conf['algorithm'] in ['FedProx']:
		executor.submit (train_FedProx_method, weights)	
	elif conf['algorithm'] in ['SCAFFOLD']:
		executor.submit (train_SCAFFOLD_method, weights, 0)

	cost_time = round(time.time() - s_time(), 2)
	worker_utils.log ('in this round, total cost time = ' + str(cost_time))
	return ''

# 1. 平凡的训练过程（FedAvg）
def on_route_train (received_weights):
	s_time = time.time()
	# 1. load 权重
	dml_utils.assign_weights (nn.model, received_weights)
	# 2. 训练
	loss_list = dml_utils.train (nn.model, train_images, train_labels, conf ['epoch'], conf ['batch_size'])
	# 3. 计时
	train_time = time.time() - s_time
	worker_utils.log ('in this round, training cost time = ' + str(train_time))
	conf ['current_round'] += 1
	# 4. 发送loss
	last_epoch_loss = loss_list [-1]
	msg = dml_utils.log_loss (last_epoch_loss, conf ['current_round'])
	worker_utils.send_print (ctl_addr, node_name + ': ' + msg)
	# 5. 发送权重(乘上client样本大小，便于aggregator处理)
	weights = nn.model.get_weights () * len(train_labels)
	dml_utils.send_weights (weights, '/combine', [conf ['father_node']], conf ['connect'])


# 2. FedProx
def train_FedProx_method(received_weights):
	s_time = time.time()
	# 1. load 权重
	dml_utils.assign_weights (nn.model, received_weights)
	# 2. 训练
	loss = dml_utils.train_on_FedProx (nn.model, train_images, train_labels, conf ['epoch'], conf ['batch_size'], mu = conf['parameters']['mu'])
	# 3. 计时
	train_time = time.time() - s_time
	worker_utils.log ('in this round, training cost time = ' + str(train_time))
	conf ['current_round'] += 1
	# 4. 发送loss
	msg = dml_utils.log_loss (loss, conf ['current_round'])
	worker_utils.send_print (ctl_addr, node_name + ': ' + msg)
	# 5. 发送权重(乘上client样本数量，便于aggregator处理)
	weights = nn.model.get_weights () * len(train_labels)
	dml_utils.send_weights (weights, '/combine', [conf ['father_node']], conf ['connect'])


# 3. FedNova
def train_FedNova_method (received_weights):
	s_time = time.time()
	# 1. load 权重
	dml_utils.assign_weights (nn.model, received_weights)
	# 2. 训练
	loss, delta_w = dml_utils.train (nn.model, train_images, train_labels, conf ['epoch'], conf ['batch_size'])
	# 3. 计时
	train_time = time.time() - s_time
	worker_utils.log ('in this round, training cost time = ' + str(train_time))
	conf ['current_round'] += 1
	# 4. 发送loss
	msg = dml_utils.log_loss (loss, conf ['current_round'])
	worker_utils.send_print (ctl_addr, node_name + ': ' + msg)
	# 5. 发送权重
	dml_utils.send_weights (delta_w, '/combine_FedNova', [conf ['father_node']], conf ['connect'])

# 4. SCAFFOLD
def train_SCAFFOLD_method (received_weights, global_c):
	s_time = time.time()
	# 1. load 权重
	dml_utils.assign_weights (nn.model, received_weights)
	# 2. 训练
	loss = dml_utils.train (nn.model, train_images, train_labels, conf ['epoch'], conf ['batch_size'], global_c)
	# 3. 计时
	train_time = time.time() - s_time
	worker_utils.log ('in this round, training cost time = ' + str(train_time))
	conf ['current_round'] += 1
	# 4. 发送loss
	msg = dml_utils.log_loss (loss, conf ['current_round'])
	worker_utils.send_print (ctl_addr, node_name + ': ' + msg)
	# 5. 发送权重
	weights = nn.model.get_weights () * len(train_labels)
	dml_utils.send_weights (weights, '/combine_SCAFFOLD', [conf ['father_node']], conf ['connect'])

# 5. FedAcc

# 6. FedAsync

# app.debug = True
app.run (host='0.0.0.0', port=dml_port, threaded=True)

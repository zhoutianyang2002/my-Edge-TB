import json
import os
import math
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

global_weights = nn.model.get_weights ()
global_c = 0
input_shape = nn.input_shape
log_file = os.path.abspath (os.path.join (dirname, '../dml_file/log/', node_name + '.log'))
worker_utils.set_log (log_file)
conf = {}
trainer_list = []
trainer_per_round = 0
# configurable parameter, specify the dataset path.
test_path = os.path.join (dirname, '../dataset/FASHION_MNIST/test_data')
test_images: np.ndarray
test_labels: np.ndarray
selected_trainers = []


app = Flask (__name__)
lock = threading.RLock ()
executor = ThreadPoolExecutor (1)
timer = dml_utils.Timer()

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
	global test_images, test_labels
	useLocalData = conf['useLocalData']
	worker_utils.log('useLocalData = ' + useLocalData)
	## 使用本地数据集：
	if useLocalData == 'True':
		local_dataset_path = os.path.join(dirname, "../local_dataset/FASHION_MNIST/test_data")
		test_images, test_labels = dml_utils.load_data (local_dataset_path, conf ["test_idxs"], input_shape)
		worker_utils.log('load data finish')
		worker_utils.log('size of test images = ' + str(len(test_images)))
		worker_utils.log('shape of test images = ' + str(test_images.shape))
		return ''
	
	## 不使用本地数据集：
	# 1.创建mydataset文件夹
	ctl_dataset_path = os.path.join(dirname, "../dataset/")
	my_dataset_path = os.path.join(dirname, "../my_dataset/")
	if os.path.exists(my_dataset_path) == False:
		os.makedirs(my_dataset_path)
   
	# 2.解压
	tar_file_name = os.path.join(ctl_dataset_path, worker_name + '.tar') 
	dml_utils.decompress(tar_file_name, path=my_dataset_path)

	# 3.load
	test_images, test_labels = dml_utils.load_data (my_dataset_path, conf ['test_start_index'],
		conf ['test_len'], input_shape)

	return ''


@app.route ('/conf/structure', methods=['POST'])
def route_conf_s ():
	f = request.files.get ('conf').read ()
	conf.update (json.loads (f))
	worker_utils.log ('POST at /conf/structure')

	filename = os.path.join (dirname, '../dml_file/conf', node_name + '_structure.conf')
	with open (filename, 'w') as fw:
		fw.writelines (json.dumps (conf, indent=2))

	worker_utils.log ('the fed algorithm is ' + conf['algorithm'])

	global trainer_per_round
	conf ['current_round'] = 0
	conf ['received_number'] = 0
	conf ['received_weights'] = []
	trainer_list.extend (conf ['child_node'])
	trainer_per_round = int (len (trainer_list) * conf ['trainer_fraction'])
	return ''


# for customized selection >>>

total_time = {}
send_time = {}
name_list = []
prob_list = []


@app.route ('/time/train', methods=['GET'])
def route_time_train ():
	worker_utils.log ('GET at /time/train')
	node = request.args.get ('node')
	_time = request.args.get ('time', type=float)
	worker_utils.log ('train: ' + node + ' use ' + str (_time))
	with lock:
		total_time [node] = _time
		if len (total_time) == len (trainer_list):
			file_path = os.path.join (dirname, '../dml_file/train_time.txt')
			with open (file_path, 'w') as f:
				f.write (json.dumps (total_time))
				worker_utils.log ('train time collection completed, saved on ' + file_path)
	return ''


@app.route ('/time/test', methods=['POST'])
def route_stest ():
	worker_utils.log ('POST at /time/test')
	# trainer sends weights to here, so it can get the time for sending.
	_ = dml_utils.parse_weights (request.files.get ('weights'))
	return ''


@app.route ('/time/send', methods=['GET'])
def route_time_send ():
	worker_utils.log ('GET at /time/send')
	node = request.args.get ('node')
	_time = request.args.get ('time', type=float)
	worker_utils.log ('send: ' + node + ' use ' + str (_time))
	with lock:
		send_time [node] = _time
		if len (send_time) == len (trainer_list):
			file_path = os.path.join (dirname, '../dml_file/send_time.txt')
			with open (file_path, 'w') as f:
				f.write (json.dumps (send_time))
				worker_utils.log ('send time collection completed, saved on ' + file_path)

			count = 0
			for node in total_time:
				total_time [node] += send_time [node]
			file_path = os.path.join (dirname, '../dml_file/total_time.txt')
			with open (file_path, 'w') as f:
				f.write (json.dumps (total_time))
				worker_utils.log ('total time collection completed, saved on ' + file_path)
			for node in total_time:
				total_time [node] = 1 / (total_time [node] ** 0.5)
				count += total_time [node]
			for node in total_time:
				name_list.append (node)
				prob_list.append (round (total_time [node] / count, 3) * 1000)
			count = 0
			for i in range (len (prob_list)):
				count += prob_list [i]
			prob_list [-1] += 1000 - count
			for i in range (len (prob_list)):
				prob_list [i] /= 1000
			worker_utils.log ('prob_list = ')
			worker_utils.log (prob_list)
	return ''


def customized_selection (number):
	global selected_trainers
	selected_trainers = np.random.choice (name_list, number, p=prob_list, replace=False)
	conf["tot_train_dataset"] = 0
	for node in selected_trainers:
		conf["tot_train_dataset"] += conf["data_size_of_trainers"][node]
	worker_utils.log(f"total numbers of train samples in this round = {conf['tot_train_dataset']}")
	return selected_trainers


# <<< for customized selection

@app.route ('/log', methods=['GET'])
def route_log ():
	executor.submit (on_route_log)
	return ''


def on_route_log ():
	worker_utils.send_log (ctl_addr, log_file, node_name)


@app.route ('/start', methods=['GET'])
def route_start ():
	worker_utils.log ('GET at /start')
	executor.submit (on_route_start)
	return ''


def on_route_start ():
	timer.start()
	_, init_acc = dml_utils.test_on_batch (nn.model, test_images, test_labels, conf ['batch_size'])
	worker_utils.log("init_acc = " + str(init_acc))
	msg = dml_utils.log_acc (init_acc, 0) # 改：取消了最后一个参数:可能导致wrong
	worker_utils.send_print (ctl_addr, node_name + ': ' + msg)

	# 1. 从节点中抽样
	# trainers = dml_utils.random_selection (trainer_list, trainer_per_round)
	customized_selection (trainer_per_round)
	worker_utils.log("trainers that be selected:")
	worker_utils.log(selected_trainers)
	# 2. 发送权重
	if conf['algorithm'] in ['FedAvg', 'FedProx', 'FedNova', 'SCAFFOLD']:
		dml_utils.send_weights (global_weights, '/train', selected_trainers, conf ['connect'])
	worker_utils.log("send weights success")
	worker_utils.send_print (ctl_addr, 'start FL')


# ------------------------1. FedAvg + FedProx---------------------------
# combine request from the lower layer node.
@app.route ('/combine', methods=['POST'])
def route_combine ():
	worker_utils.log ('POST at /combine')
	weights = dml_utils.parse_weights (request.files.get ('weights'))
	executor.submit (on_route_combine, weights)
	return ''


def on_route_combine (weights):
	with lock:
		conf ['received_number'] += 1

		# 注意：默认发来的数据已经乘上了client的样本大小，加起来，最后只需要除上被选中的样本总数即可
		dml_utils.store_weights (conf ['received_weights'], weights, conf ['received_number'])
		if conf ['received_number'] == trainer_per_round:
			combine_weights ()


def combine_weights ():
	weights = dml_utils.avg_weights_by_size (conf ['received_weights'], conf["tot_train_dataset"])
	dml_utils.assign_weights (nn.model, weights)
	conf ['received_weights'].clear ()
	conf ['received_number'] = 0
	conf ['current_round'] += 1

	_, acc = dml_utils.test (nn.model, test_images, test_labels)
	msg = dml_utils.log_acc (acc, conf ['current_round'])
	worker_utils.send_print (ctl_addr, node_name + ': ' + msg)
	timer.stop()
	if conf ['current_round'] == conf ['sync']:
		worker_utils.log ('>>>>>training ended<<<<<')
		worker_utils.send_data ('GET', '/finish', ctl_addr)
		timer.printAggTimes()
	else:  # send down to train.
		# trainers = dml_utils.random_selection (trainer_list, trainer_per_round)
		timer.start()
		trainers = customized_selection (trainer_per_round)
		dml_utils.send_weights (weights, '/train', trainers, conf ['connect'])

# ------------------------ 2. FedNova  ------------------------------------
# combine request from the lower layer node.
@app.route ('/combine_FedNova', methods=['POST'])
def route_combine_FedNova ():
	worker_utils.log ('POST at /combine_FedNova')
	weights = dml_utils.parse_weights (request.files.get ('weights'))
	executor.submit (on_route_combine_FedNova, weights)
	return ''


def on_route_combine_FedNova (weights):
	with lock:
		conf ['received_number'] += 1

		# 注意：默认发来的数据已经乘上了client的样本大小，加起来，最后只需要除上被选中的样本总数即可
		dml_utils.store_weights (conf ['received_weights'], weights, conf ['received_number'])
		if conf ['received_number'] == trainer_per_round:
			combine_weights_FedNova ()


def combine_weights_FedNova (learning_rate = 0.01):
	global global_weights
	sum_weights = dml_utils.avg_weights_by_size (conf ['received_weights'], conf["tot_train_dataset"])
	nova_factor = 0
	for node in selected_trainers:
		steps_of_node = conf['epoch_of_trainers'] * math.ceil(conf['data_size_of_trainers'][node] / conf['batch_size'])
		nova_factor += conf['data_size_of_trainers'][node] * steps_of_node
	global_weights -= learning_rate * sum_weights * nova_factor
	
	dml_utils.assign_weights (nn.model, global_weights)
	conf ['received_weights'].clear ()
	conf ['received_number'] = 0
	conf ['current_round'] += 1

	_, acc = dml_utils.test (nn.model, test_images, test_labels)
	msg = dml_utils.log_acc (acc, conf ['current_round'])
	worker_utils.send_print (ctl_addr, node_name + ': ' + msg)
	timer.stop()
	if conf ['current_round'] == conf ['sync']:
		worker_utils.log ('>>>>>training ended<<<<<')
		worker_utils.send_data ('GET', '/finish', ctl_addr)
		timer.printAggTimes()
	else:  # send down to train.
		# trainers = dml_utils.random_selection (trainer_list, trainer_per_round)
		timer.start()
		trainers = customized_selection (trainer_per_round)
		dml_utils.send_weights (global_weights, '/train', trainers, conf ['connect'])


# ------------------------ 3. SCAFFOLD  ------------------------------------

# combine request from the lower layer node.
@app.route ('/combine_SCAFFOLD', methods=['POST'])
def route_combine_SCAFFOLD ():
	worker_utils.log ('POST at /combine_SCAFFOLD')
	weights = dml_utils.parse_weights (request.files.get ('weights'))
	delta_c = dml_utils.parse_c (request.files.get ('deta_c'))
	executor.submit (on_route_combine_SCAFFOLD, weights, delta_c)
	return ''


def on_route_combine_SCAFFOLD (weights, delta_c):
	with lock:
		conf ['received_number'] += 1

		# 注意：默认发来的数据已经乘上了client的样本大小，加起来，最后只需要除上被选中的样本总数即可
		dml_utils.store_weights (conf ['received_weights'], weights, conf ['received_number'])
		if conf ['received_number'] == trainer_per_round:
			combine_weights_SCAFFOLD (delta_c)


def combine_weights_SCAFFOLD (delta_c):
	global global_c
	weights = dml_utils.avg_weights_by_size (conf ['received_weights'], conf["tot_train_dataset"])
	dml_utils.assign_weights (nn.model, weights)
	global_c += conf["tot_train_dataset"] * delta_c
	conf ['received_weights'].clear ()
	conf ['received_number'] = 0
	conf ['current_round'] += 1

	_, acc = dml_utils.test (nn.model, test_images, test_labels)
	msg = dml_utils.log_acc (acc, conf ['current_round'])
	worker_utils.send_print (ctl_addr, node_name + ': ' + msg)
	timer.stop()
	if conf ['current_round'] == conf ['sync']:
		worker_utils.log ('>>>>>training ended<<<<<')
		worker_utils.send_data ('GET', '/finish', ctl_addr)
		timer.printAggTimes()
	else:  # send down to train.
		# trainers = dml_utils.random_selection (trainer_list, trainer_per_round)
		timer.start()
		trainers = customized_selection (trainer_per_round)
		dml_utils.send_weights (global_weights, '/train', trainers, conf ['connect'])



# app.debug = True
app.run (host='0.0.0.0', port=dml_port, threaded=True)

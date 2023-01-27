import json
import time
import tarfile
import os
from typing import Dict, IO

import requests
import numpy as np

def read_json (path: str):
	with open (path, 'r') as f:
		return json.loads (f.read ().replace ('\'', '\"'))


def send_data (method: str, path: str, address: str, port: int = None,
		data: Dict [str, str] = None, files: Dict [str, IO] = None) -> str:
	"""
	send a request to http://${address/path} or http://${ip:port/path}.
	@param method: 'GET' or 'POST'.
	@param path:
	@param address: ip:port if ${port} is None else only ip.
	@param port: only used when ${address} is only ip.
	@param data: only used in 'POST'.
	@param files: only used in 'POST'.
	@return: response.text
	"""
	# if path == '/start':
	# 	print("send data is called")
	# 	print(method, path, address, port, data, files)

	if port:
		address += ':' + str (port)
	if method.upper () == 'GET':
		res = requests.get ('http://' + address + '/' + path)
		return res.text
	elif method.upper () == 'POST':
		res = requests.post ('http://' + address + '/' + path, data=data, files=files)
		return res.text
	else:
		return 'err method ' + method

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
    def printTimes(self):
        """打印list"""
        outputs = [round(ele / 60.0, 2) for ele in self.times]
        print('cost time in each aggregation: ')
        print(outputs)
    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)
    def sum(self):
        """返回时间总和"""
        return sum(self.times)
    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()

def compress(tar_file_name, tar_file_list=[], tar_dir_list=[], model="w:gz"):
	"""
	tar_file_list所有文件和tar_dir_list所有目录下的所有文件，会被压缩到一个tar_file_name的压缩文件中
	model="w:gz"会将文件压缩为.gz
	"""
	with tarfile.open(tar_file_name, model) as tar_obj:
		# 压缩文件
		for tmp_file in tar_file_list:
			arcname = os.path.basename(tmp_file) # arcname是压缩文件的名字，区别于保存路径
			tar_obj.add(tmp_file, arcname=arcname)
		# 压缩目录。和zipfile相比tarfile允许直接压缩目录，而不需要去遍历目录一个个文件压
		for tmp_dir in tar_dir_list:
			tar_obj.add(tmp_dir) 

def compress_dataset(worker_name, conf_file_path):
	# 1.理清文件路径
	dirname = os.path.abspath (os.path.dirname (__file__)) # ./controller/base
	dataset_path = os.path.join(dirname, "../dataset/")
	train_path = os.path.join(dirname, "../dataset/FASHION_MNIST/train_data")
	test_path = os.path.join(dirname, "../dataset/FASHION_MNIST/test_data")
	# 2.读取worker节点对应的配置信息，保存在conf字典里
	conf = {}
	with open(conf_file_path) as f: 
		conf.update (json.loads (f.read()))
	if conf['useLocalData'] == 'True': # 判断需不需要压缩
		return ''
	# 3.把要压缩的文件名保存在list中
	def get_tar_file_list(file_type, data_type):
		if conf[data_type + '_len'] == -1:
			return []
		file_list = []
		begin, end = conf [data_type + '_start_index'], conf [data_type + '_start_index'] + conf [data_type + '_len']
		for i in range(begin, end): # (-1,-2)就是不包含date_type类型的数据文件
			fileName = file_type + str(i) + '.npy'
			if data_type == 'train':
				filePath = os.path.join(train_path, fileName).replace('\\', '/')
			else :
				filePath = os.path.join(test_path, fileName).replace('\\', '/')
			file_list.append(filePath)
		return file_list
	
	train_x = get_tar_file_list('images_', 'train')
	train_y = get_tar_file_list('labels_', 'train')
	test_x = get_tar_file_list('images_', 'test')
	test_y = get_tar_file_list('labels_', 'test')
	
	# 4.压缩文件
	tar_file_train = os.path.join(dataset_path, worker_name + '_train.tar')
	tar_file_test = os.path.join(dataset_path, worker_name + '_test.tar')
	compress(tar_file_train, train_x + train_y)
	compress(tar_file_test, test_x + test_y)

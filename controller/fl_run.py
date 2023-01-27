import os

from base import default_testbed
from base.utils import read_json
from fl_manager import FlManager

# path of this file.
dirName = os.path.abspath (os.path.dirname (__file__))

# we made up the following physical hardware so this example is NOT runnable.
if __name__ == '__main__':
	testbed = default_testbed (ip='106.53.110.245', dir_name=dirName, manager_class=FlManager)

	nfsApp = testbed.add_nfs (tag='dml_app', path=os.path.join (dirName, 'dml_app'))
	nfsDataset = testbed.add_nfs (tag='dataset', path=os.path.join (dirName, 'dataset'))

	# define your network >>>

	emu_zty = testbed.add_emulator (name='emulator1', ip='111.230.154.108', cpu=8, ram=32, unit='G')
	emu_whl = testbed.add_emulator (name='emulator2', ip='114.132.177.213', cpu=8, ram=32, unit='G')
	emu_hjb = testbed.add_emulator (name='emulator3', ip='159.75.112.104', cpu=8, ram=32, unit='G')
	emu_lzy = testbed.add_emulator (name='emulator4', ip='81.71.23.105', cpu=8, ram=32, unit='G')
	
	en = testbed.add_emulated_node (name='n1', working_dir='/home/worker/dml_app',
		cmd=['python3', 'fl_aggregator.py'], image='dml:v1.0', cpu=4, ram=16, unit='G', emulator=emu_zty)
	en.mount_local_path ('./dml_file', '/home/worker/dml_file')
	en.mount_local_path ('./local_dataset/FASHION_MNIST', '/home/worker/local_dataset/FASHION_MNIST')
	en.mount_nfs (nfs=nfsApp, node_path='/home/worker/dml_app')
	en.mount_nfs (nfsDataset, '/home/worker/dataset')

	en = testbed.add_emulated_node (name='n2', working_dir='/home/worker/dml_app',
		cmd=['python3', 'fl_trainer.py'], image='dml:v1.0', cpu=1, ram=4, unit='G', emulator=emu_zty)
	en.mount_local_path ('./dml_file', '/home/worker/dml_file')
	en.mount_local_path ('./local_dataset/FASHION_MNIST', '/home/worker/local_dataset/FASHION_MNIST')
	en.mount_nfs (nfs=nfsApp, node_path='/home/worker/dml_app')
	en.mount_nfs (nfsDataset, '/home/worker/dataset')

	en = testbed.add_emulated_node (name='n3', working_dir='/home/worker/dml_app',
		cmd=['python3', 'fl_trainer.py'], image='dml:v1.0', cpu=1, ram=4, unit='G', emulator=emu_zty)
	en.mount_local_path ('./dml_file', '/home/worker/dml_file')
	en.mount_local_path ('./local_dataset/FASHION_MNIST', '/home/worker/local_dataset/FASHION_MNIST')
	en.mount_nfs (nfs=nfsApp, node_path='/home/worker/dml_app')
	en.mount_nfs (nfsDataset, '/home/worker/dataset')

	en = testbed.add_emulated_node (name='n4', working_dir='/home/worker/dml_app',
		cmd=['python3', 'fl_trainer.py'], image='dml:v1.0', cpu=3, ram=8, unit='G', emulator=emu_whl)
	en.mount_local_path ('./dml_file', '/home/worker/dml_file')
	en.mount_local_path ('./local_dataset/FASHION_MNIST', '/home/worker/local_dataset/FASHION_MNIST')
	en.mount_nfs (nfs=nfsApp, node_path='/home/worker/dml_app')
	en.mount_nfs (nfsDataset, '/home/worker/dataset')

	en = testbed.add_emulated_node (name='n5', working_dir='/home/worker/dml_app',
		cmd=['python3', 'fl_trainer.py'], image='dml:v1.0', cpu=3, ram=8, unit='G', emulator=emu_whl)
	en.mount_local_path ('./dml_file', '/home/worker/dml_file')
	en.mount_local_path ('./local_dataset/FASHION_MNIST', '/home/worker/local_dataset/FASHION_MNIST')
	en.mount_nfs (nfs=nfsApp, node_path='/home/worker/dml_app')
	en.mount_nfs (nfsDataset, '/home/worker/dataset')

	en = testbed.add_emulated_node (name='n6', working_dir='/home/worker/dml_app',
		cmd=['python3', 'fl_trainer.py'], image='dml:v1.0', cpu=2, ram=8, unit='G', emulator=emu_hjb)
	en.mount_local_path ('./dml_file', '/home/worker/dml_file')
	en.mount_local_path ('./local_dataset/FASHION_MNIST', '/home/worker/local_dataset/FASHION_MNIST')
	en.mount_nfs (nfs=nfsApp, node_path='/home/worker/dml_app')
	en.mount_nfs (nfsDataset, '/home/worker/dataset')

	en = testbed.add_emulated_node (name='n7', working_dir='/home/worker/dml_app',
		cmd=['python3', 'fl_trainer.py'], image='dml:v1.0', cpu=2, ram=8, unit='G', emulator=emu_hjb)
	en.mount_local_path ('./dml_file', '/home/worker/dml_file')
	en.mount_local_path ('./local_dataset/FASHION_MNIST', '/home/worker/local_dataset/FASHION_MNIST')
	en.mount_nfs (nfs=nfsApp, node_path='/home/worker/dml_app')
	en.mount_nfs (nfsDataset, '/home/worker/dataset')

	en = testbed.add_emulated_node (name='n8', working_dir='/home/worker/dml_app',
		cmd=['python3', 'fl_trainer.py'], image='dml:v1.0', cpu=1, ram=4, unit='G', emulator=emu_hjb)
	en.mount_local_path ('./dml_file', '/home/worker/dml_file')
	en.mount_local_path ('./local_dataset/FASHION_MNIST', '/home/worker/local_dataset/FASHION_MNIST')
	en.mount_nfs (nfs=nfsApp, node_path='/home/worker/dml_app')
	en.mount_nfs (nfsDataset, '/home/worker/dataset')

	en = testbed.add_emulated_node (name='n9', working_dir='/home/worker/dml_app',
		cmd=['python3', 'fl_trainer.py'], image='dml:v1.0', cpu=1, ram=4, unit='G', emulator=emu_hjb)
	en.mount_local_path ('./dml_file', '/home/worker/dml_file')
	en.mount_local_path ('./local_dataset/FASHION_MNIST', '/home/worker/local_dataset/FASHION_MNIST')
	en.mount_nfs (nfs=nfsApp, node_path='/home/worker/dml_app')
	en.mount_nfs (nfsDataset, '/home/worker/dataset')

	en = testbed.add_emulated_node (name='n10', working_dir='/home/worker/dml_app',
		cmd=['python3', 'fl_trainer.py'], image='dml:v1.0', cpu=1, ram=4, unit='G', emulator=emu_lzy)
	en.mount_local_path ('./dml_file', '/home/worker/dml_file')
	en.mount_local_path ('./local_dataset/FASHION_MNIST', '/home/worker/local_dataset/FASHION_MNIST')
	en.mount_nfs (nfs=nfsApp, node_path='/home/worker/dml_app')
	en.mount_nfs (nfsDataset, '/home/worker/dataset')

	en = testbed.add_emulated_node (name='n11', working_dir='/home/worker/dml_app',
		cmd=['python3', 'fl_trainer.py'], image='dml:v1.0', cpu=1, ram=4, unit='G', emulator=emu_lzy)
	en.mount_local_path ('./dml_file', '/home/worker/dml_file')
	en.mount_local_path ('./local_dataset/FASHION_MNIST', '/home/worker/local_dataset/FASHION_MNIST')
	en.mount_nfs (nfs=nfsApp, node_path='/home/worker/dml_app')
	en.mount_nfs (nfsDataset, '/home/worker/dataset')

	en = testbed.add_emulated_node (name='n12', working_dir='/home/worker/dml_app',
		cmd=['python3', 'fl_trainer.py'], image='dml:v1.0', cpu=1, ram=4, unit='G', emulator=emu_lzy)
	en.mount_local_path ('./dml_file', '/home/worker/dml_file')
	en.mount_local_path ('./local_dataset/FASHION_MNIST', '/home/worker/local_dataset/FASHION_MNIST')
	en.mount_nfs (nfs=nfsApp, node_path='/home/worker/dml_app')
	en.mount_nfs (nfsDataset, '/home/worker/dataset')

	en = testbed.add_emulated_node (name='n13', working_dir='/home/worker/dml_app',
		cmd=['python3', 'fl_trainer.py'], image='dml:v1.0', cpu=1, ram=4, unit='G', emulator=emu_lzy)
	en.mount_local_path ('./dml_file', '/home/worker/dml_file')
	en.mount_local_path ('./local_dataset/FASHION_MNIST', '/home/worker/local_dataset/FASHION_MNIST')
	en.mount_nfs (nfs=nfsApp, node_path='/home/worker/dml_app')
	en.mount_nfs (nfsDataset, '/home/worker/dataset')

	en = testbed.add_emulated_node (name='n14', working_dir='/home/worker/dml_app',
		cmd=['python3', 'fl_trainer.py'], image='dml:v1.0', cpu=1, ram=4, unit='G', emulator=emu_lzy)
	en.mount_local_path ('./dml_file', '/home/worker/dml_file')
	en.mount_local_path ('./local_dataset/FASHION_MNIST', '/home/worker/local_dataset/FASHION_MNIST')
	en.mount_nfs (nfs=nfsApp, node_path='/home/worker/dml_app')
	en.mount_nfs (nfsDataset, '/home/worker/dataset')

	en = testbed.add_emulated_node (name='n15', working_dir='/home/worker/dml_app',
		cmd=['python3', 'fl_trainer.py'], image='dml:v1.0', cpu=1, ram=4, unit='G', emulator=emu_lzy)
	en.mount_local_path ('./dml_file', '/home/worker/dml_file')
	en.mount_local_path ('./local_dataset/FASHION_MNIST', '/home/worker/local_dataset/FASHION_MNIST')
	en.mount_nfs (nfs=nfsApp, node_path='/home/worker/dml_app')
	en.mount_nfs (nfsDataset, '/home/worker/dataset')

	# declare a physical node,
	# which should run the worker/agent.py in advance.
	p1 = testbed.add_physical_node (name='p1', nic='eth0', ip='43.138.145.87')
	# ${mount_point} can use absolute path starting from / or
	# relative path starting from the directory of the worker/agent.py file.
	# ${mount_point} means /path/in/p1/to/worker/dml_app in this example.
	p1.mount_nfs (nfs=nfsApp, mount_point='./dml_app')
	p1.mount_nfs (nfsDataset, './dataset')
	# set physical node's role.
	# ${working_dir} can use absolute path or relative path as above mount_nfs ().
	p1.set_cmd (working_dir='dml_app', cmd=['python3', 'fl_trainer.py'])

	
	# # declare an emulator, which should run the worker/agent.py in advance.
	# emu1 = testbed.add_emulator (name='emulator-1', ip='111.230.154.108', cpu=8, ram=32, unit='G')
	# en = testbed.add_emulated_node (name='n1', working_dir='/home/worker/dml_app',
	# 	cmd=['python3', 'fl_aggregator.py'], image='dml:v1.0', cpu=2, ram=8, unit='G', emulator=emu1)
	# en.mount_local_path ('./dml_file', '/home/worker/dml_file')
	# en.mount_local_path ('./local_dataset/FASHION_MNIST', '/home/worker/local_dataset/FASHION_MNIST')
	# en.mount_nfs (nfs=nfsApp, node_path='/home/worker/dml_app')
	# en.mount_nfs (nfsDataset, '/home/worker/dataset')

	# # emu2 = testbed.add_emulator (name='emulator-2', ip='114.132.177.213', cpu=8, ram=32, unit='G')
	# # for i in range(2, 6):
	# # 	en = testbed.add_emulated_node (name='n'+str(i), working_dir='/home/worker/dml_app',
	# # 		cmd=['python3', 'fl_trainer.py'], image='dml:v1.0', cpu=2, ram=8, unit='G', emulator=emu2)
	# # 	en.mount_local_path ('./dml_file', '/home/worker/dml_file')
	# # 	en.mount_local_path ('./local_dataset/FASHION_MNIST', '/home/worker/local_dataset/FASHION_MNIST')
	# # 	en.mount_nfs (nfs=nfsApp, node_path='/home/worker/dml_app')
	# # 	en.mount_nfs (nfsDataset, '/home/worker/dataset')


	# # add many emulated nodes.
	# emu2 = testbed.add_emulator ('emulator-2', '114.132.177.213', cpu=8, ram=32, unit='G')
	# emu3 = testbed.add_emulator ('emulator-3', '159.75.112.104', cpu=8, ram=32, unit='G')
	# emu4 = testbed.add_emulator ('emulator-4', '81.71.23.105', cpu=8, ram=32, unit='G')
	# emu5 = testbed.add_emulator ('emulator-5', '43.138.145.87', cpu=8, ram=32, unit='G')

	# en = testbed.add_emulated_node ('n2', '/home/worker/dml_app',
	# 		['python3', 'fl_trainer.py'], 'dml:v1.0', cpu=1, ram=6, unit='G', emulator=emu2)
	# en.mount_local_path ('./dml_file', '/home/worker/dml_file')
	# en.mount_local_path ('./local_dataset/FASHION_MNIST', '/home/worker/local_dataset/FASHION_MNIST')
	# en.mount_nfs (nfsApp, '/home/worker/dml_app')
	# en.mount_nfs (nfsDataset, '/home/worker/dataset')

	# en = testbed.add_emulated_node ('n3', '/home/worker/dml_app',
	# 		['python3', 'fl_trainer.py'], 'dml:v1.0', cpu=1, ram=6, unit='G', emulator=emu3)
	# en.mount_local_path ('./dml_file', '/home/worker/dml_file')
	# en.mount_local_path ('./local_dataset/FASHION_MNIST', '/home/worker/local_dataset/FASHION_MNIST')
	# en.mount_nfs (nfsApp, '/home/worker/dml_app')
	# en.mount_nfs (nfsDataset, '/home/worker/dataset')

	# en = testbed.add_emulated_node ('n4', '/home/worker/dml_app',
	# 		['python3', 'fl_trainer.py'], 'dml:v1.0', cpu=1, ram=6, unit='G', emulator=emu4)
	# en.mount_local_path ('./dml_file', '/home/worker/dml_file')
	# en.mount_local_path ('./local_dataset/FASHION_MNIST', '/home/worker/local_dataset/FASHION_MNIST')
	# en.mount_nfs (nfsApp, '/home/worker/dml_app')
	# en.mount_nfs (nfsDataset, '/home/worker/dataset')

	# en = testbed.add_emulated_node ('n5', '/home/worker/dml_app',
	# 		['python3', 'fl_trainer.py'], 'dml:v1.0', cpu=1, ram=6, unit='G', emulator=emu5)
	# en.mount_local_path ('./dml_file', '/home/worker/dml_file')
	# en.mount_local_path ('./local_dataset/FASHION_MNIST', '/home/worker/local_dataset/FASHION_MNIST')
	# en.mount_nfs (nfsApp, '/home/worker/dml_app')
	# en.mount_nfs (nfsDataset, '/home/worker/dataset')

	# load tc settings from links.json.
	links_json = read_json (os.path.join (dirName, 'links.json'))
	testbed.load_link (links_json)

	"""
	the contents in this example links.json are:

	{
	  "p1": [
	    {"dest": "n1", "bw": "5mbps"},
	    {"dest": "n4", "bw": "3mbps"}
	  ],
	  "n1": [
	    {"dest": "p1", "bw": "3mbps"},
	    {"dest": "n2", "bw": "3mbps"},
	    {"dest": "n3", "bw": "3mbps"},
	    {"dest": "n4", "bw": "3mbps"}
	  ],
	  "n2": [
	    {"dest": "n1", "bw": "1mbps"},
	    {"dest": "n3", "bw": "2mbps"},
	    {"dest": "n4", "bw": "3mbps"}
	  ],
	  "n3": [
	    {"dest": "n1", "bw": "4mbps"},
	    {"dest": "n2", "bw": "1mbps"}
	  ],
	  "n4": [
	    {"dest": "n1", "bw": "1mbps"},
	    {"dest": "n2", "bw": "5mbps"},
	    {"dest": "p1", "bw": "2mbps"}
	  ]
	}

	it takes a while to deploy the tc settings.
	when the terminal prints "tc finish", tc settings of all emulated and physical nodes are deployed.
	please make sure your node communicate with other nodes after "tc finish".

	you can send a GET request to ctl's /update/tc at any time
	to update the tc settings of emulated and/or physical nodes. 

	for example, curl http://192.168.1.10:3333/update/tc?file=links2.json
	the contents in this example links2.json are:

	{
	  "n1": [
	    {"dest": "p1", "bw": "1mbps"},
	    {"dest": "n2", "bw": "3mbps"},
	    {"dest": "n3", "bw": "3mbps"},
	    {"dest": "n4", "bw": "3mbps"}
	  ],
	  "n4": [
	    {"dest": "n1", "bw": "3mbps"},
	    {"dest": "n2", "bw": "5mbps"},
	    {"dest": "n3", "bw": "1mbps"},
	    {"dest": "p1", "bw": "2mbps"}
	  ]
	}

	we will clear the tc settings of n1 and n4 and deploy the new one dynamically
	without stop nodes.
	for the above reasons, even if the bw from n1 to n2, n3 and n4 does not change,
	they need to be specified.
	"""

	# <<< define your network

	# when you finish your experiments, you should restore your emulators and physical nodes.
	# GET at '/emulated/stop', '/emulated/clear' and '/emulated/reset',
	# these requests can be received by controller/base/manager.py, route_emulated_stop (), ect.
	# GET at '/physical/stop', '/physical/clear/tc', '/physical/clear/nfs' and 'physical/reset',
	testbed.start ()

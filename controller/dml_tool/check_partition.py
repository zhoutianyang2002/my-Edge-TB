'''
读取所有的conf文件下标，确保不重不漏
'''
import os
import json
import numpy as np

if __name__ == '__main__':
    conf = {}
    x_list = []
    y_list = []
    dirname = os.path.abspath (os.path.dirname (__file__))
    conf_path = os.path.join(dirname, '../dml_file/conf')
    node = ['n'+str(i) for i in range(1, 16)] + ['p1']
    tot1 = tot2 = 0
    

    for node_name in node:
        print("-----------" + node_name + "-----------")
        node_path = os.path.join(conf_path, node_name + '_dataset.conf')
        with open(node_path, 'r') as f:
            conf.update (json.loads (f.read()))
        
        if node_name == 'n1':
            assert conf["test_len"] == 10000, 'number of test data may not true'
            
            
        else :
            tot1 += conf["train_len"]
            
            
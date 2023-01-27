'''
注意这个文件在controller上只能划分fashion-mnist，不支持cifar10
因为controller内存太小了

为了节约时间，代码一些地方可扩展性不强，存在硬编码
需要仔细看注释，根据需要修改需要灵活调整的地方
'''

import os
import argparse
import json

import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.datasets import cifar10

from splitter_utils import split_data, save_data

train_images: np.ndarray
test_images: np.ndarray
train_labels: np.ndarray
test_labels: np.ndarray



def partition_strategy_factory(node_name, args):
    '''
    本来想搞成策略模式+工厂模式，但是太麻烦了，也没什么意思，就这样了。
    '''
    global train_images, test_images, train_labels, test_labels
    client_name = node_name
    strategy = args.strategy
    n_clients = args.n_clients
    dirName = os.path.abspath (os.path.dirname (__file__))
    save_path = os.path.join(dirName, '../dataset')
    if not os.path.exists (save_path):
        os.makedirs (save_path)

    # 1. 获取每个client对应的训练集下标和训练集总大小
    if strategy == 'unbalance':
        # fl只需要划分train data就可以
        idxMap, len_data = unbalance(beta=args.beta, n_clients = n_clients)
    elif strategy == 'label_skew_by_type':
        idxMap, len_data = label_skew_by_type(n_clients = n_clients)
    elif strategy == 'label_skew_by_diri':
        idxMap, len_data = label_skew_by_diri(n_clients = n_clients, beta=args.beta)
    elif strategy == 'feature_skew_by_noise':
        idxMap, len_data = feature_skew_by_noise(n_clients = n_clients)
    elif strategy == 'iid':
        idxMap, len_data = iid(n_clients=n_clients)

    print('the size of each clients: ')
    print(len_data)

    # 2. (1)add aggregator node
    test_idxs = list(range(0, 10000))
    fl_dataset_map = {"n1": {"test_len": 10000, "test_idxs": test_idxs, "batch_size": 32, "useLocalData": "True"}}

    #    (2)add trainer node
    fl_dataset_map.update({client_name[i]:{"train_len": len_data[i], "train_idxs": [int(a) for a in idxMap[i]], "batch_size": 32, "useLocalData": "True"} for i in range(n_clients)})
    
    # 3.序列化为json文件
    path_Name = os.path.join(dirName, 'fl_dataset.json')
    with open(path_Name, 'w') as f:
        json.dump(fl_dataset_map, f) 
    # 4.序列化为npy文件
    if args.save_data == 'False':
        return 
    np.save(save_path + '/n1_images', test_images)
    np.save(save_path + '/n1_labels', test_labels)
    for name in node_name:
        np.save(save_path + '/' + name + '_images', train_images[fl_dataset_map[name]["train_idxs"]])
        np.save(save_path + '/' + name + '_labels', train_labels[fl_dataset_map[name]["train_idxs"]])


def iid(n_clients):
    '''
    iid划分。注意没考虑不能平分的情况
    '''
    n_data = len(train_labels)
    idxs = np.arange(n_data)
    client_idxs = np.split(idxs, n_clients)
    data_map = {i: client_idxs[i] for i in range(n_clients)}
    len_data = [n_data / n_clients] * n_clients
    return data_map, len_data


def unbalance(beta, n_clients, min_data_num = 1):
    '''
    不存在样本重叠
    目的：让每个client拥有的样本数量不平衡（不一样），
    每个client的样本数服从狄利克雷分布：X~diri(beta)，
    beta用来衡量数据集不平衡的程度，越小越不平衡
    注意需要小心的设置client的最小样本数，设置过大有可能导致死循环
    返回一个字典，包括每个client对应的所有数据索引
    '''
    global train_labels
    n_data = len(train_labels)
    idxs = np.random.permutation(n_data) # idxs = np.arange(n_data) 
    # 注意数据集的下标和label的编号都是从0开始，只有节点的编号是从1开始

    min_size = tmp_cnt = 0
    while min_size < min_data_num: #client中最小的样本数
        tmp_cnt += 1
        assert tmp_cnt<1000, 'The minimum number of samples owned by the client may set too large!'
        proportions = np.random.dirichlet(np.repeat(beta, n_clients))
        proportions = proportions/proportions.sum() # 归一化
        min_size = np.min(proportions*n_data)
    
    proportions = (np.cumsum(proportions)*n_data).astype(int)[:-1] 
    # 注意proportions代表“分隔点的下标”，len(proportions) == n_client-1，proportions[0]=9，就代表第二个client的下标从9开始
    len_data = []
    
    client_idxs = np.split(idxs, proportions)
    data_map = {i: client_idxs[i] for i in range(n_clients)}
    for i in range(n_clients):
        len_data.append(len(data_map[i]))
        data_map[i] = data_map[i].tolist()
    assert sum(len_data) == n_data
    return data_map, len_data

def label_skew_by_type(n_clients, C = 5, drop_last=False):
    '''
    每个client拥有C种类型的标签，不存在样本重叠
    这里只考虑训练集的划分。
    划分时可能会产生“余数”，drop_last=True就是把这些样本丢掉
    在划分的时候，如果C设置过小，有可能导致某个label没有被所有client选中，导致总的训练集不全
    '''
    global train_labels
    num_data = len(train_labels)
    num_label = 10 # !!!label种类个数。是变量，注意灵活调整
    client_label = [] # 第i个client拥有哪C个label
    num_piece_of_label = [0] * num_label # 第i个label需要被分成几块
    data_map = {i:[] for i in range(n_clients)}
    len_data = [0] * n_clients

    for _ in range(0, n_clients):
        tmp = np.random.choice(num_label, C, replace=False) # 不能让它抽到相同的label
        tmp = tmp.tolist()
        client_label.append(tmp)
        for j in range(0, C):
            num_piece_of_label[tmp[j]] += 1
    print(client_label)
    print(num_piece_of_label)

    for k in range(0, num_label): # 注意label从0开始
        # 1.找出label为k的所有下标
        idx_k = np.where(train_labels==k)[0] # 注意[0]将type:tuple->ndarray
        np.random.shuffle(idx_k)
        label_size = len(idx_k) # 一种label对应多少样本
        # 2.把index分成p=num_piece_of_label[k]份
        p = num_piece_of_label[k]
        if p == 0:
            continue
        # 注意如果不能平分会抛出异常.取出idx_k尾部的数据: 
        if label_size % p:
            end_idx, last_idx = np.split(idx_k, [label_size - label_size % p]) # type:list
            proportions = np.split(end_idx, p)
            if drop_last == False: # 尾巴随机分给p份
                tmp = np.random.choice(p, len(last_idx), replace=True)
                for j in range(len(last_idx)):
                    proportions[tmp[j]] = np.append(proportions[tmp[j]], last_idx[j])
        else:
            proportions = np.split(idx_k, p) # 平分为p份，type为list。

        cnt = 0
        # 3.遍历所有的client，把这p份分给需要的client
        for i in range(0, n_clients):
            if k in client_label[i]:
                tmp = list(proportions[cnt])
                data_map[i] += tmp
                len_data[i] += len(tmp)
                cnt += 1
    # 注意shuffle每个client的训练集!不然可能对训练有影响 
    for i in range(0, n_clients):
        np.random.shuffle(data_map[i])
    print(sum(len_data))
    if drop_last == False and 0 not in num_piece_of_label:
        assert sum(len_data) == len(train_labels)
    return data_map, len_data

def label_skew_by_diri(beta = 0.5, n_clients = 4):
    """
    对于client j所拥有的标签为k的样本数 pk,j 服从狄利克雷分布。
    其结果是某些client的某种或某几种类型的样本特别多或特别少。
    """
    global train_labels

    num_data = len(train_labels)
    K = 10
    min_size = 0
    min_require_size = 10
    net_dataidx_map = {}
    data_len = []

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(n_clients)]
        for k in range(K):
            idx_k = np.where(train_labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, n_clients))
            # logger.info("proportions1: ", proportions)
            # logger.info("sum pro1:", np.sum(proportions))
            ## Balance
            proportions = np.array([p * (len(idx_j) < num_data / n_clients) for p, idx_j in zip(proportions, idx_batch)])
            # logger.info("proportions2: ", proportions)
            proportions = proportions / proportions.sum()
            # logger.info("proportions3: ", proportions)
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            # logger.info("proportions4: ", proportions)
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
            # if K == 2 and n_parties <= 10:
            #     if np.min(proportions) < 200:
            #         min_size = 0
            #         break


    for j in range(n_clients):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]
        data_len.append(len(idx_batch[j]))

    check_hash = [0] * 60000
    for i in range(n_clients):
        foo = len(idx_batch[i])
        for j in range(foo):
            assert idx_batch[i][j] < 60000 and idx_batch[i][j] >= 0, 'out of index'
            check_hash[idx_batch[i][j]] += 1
    for i in range(60000):
        assert check_hash[i] == 1, 'error'

    
    # idx_batch = [[] for _ in range(n_clients)]
    # for k in range(num_label):
    #     idx_k = np.where(train_labels == k)[0]
    #     np.random.shuffle(idx_k)
    #     proportions = np.random.dirichlet(np.repeat(beta, n_clients))
    #     proportions = np.array([p * (len(idx_j) < num_data / n_clients) for p, idx_j in zip(proportions, idx_batch)])
    #     proportions = proportions / proportions.sum()
    #     proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
    #     idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]

    assert sum(data_len) == 60000
    return net_dataidx_map, data_len
    

def feature_skew_by_noise(n_clients, mean=0, std=0.01, drop_last = False):
    """
    特征skew是指每个client拥有的样本数可能相同、相同标签的数量可能相同，但是特征的分布可能特别不一样。
    两个典型的例子：
    1. 某个client的照片都是狸花猫，另一个client的照片都是橘猫，虽然label都是猫，但是特征还是区别很大的。
    2. 把MNIST数据集按照作者进行划分（不同的人写字的特征不一样，比如2、7、9的写法）
    得到特征skew的数据，最好的方法是获取现成的真实数据集。
    如果要合成模拟的话，就是在不同client的数据集上加上不同程度的高斯噪声，或者聚类
    """
    global train_images, train_labels
    num_data = len(train_labels)
    idxs = np.random.permutation(num_data) # idxs = np.arange(n_data) 
    if num_data % n_clients:
        end_idx, last_idx = np.split(idxs, [num_data - num_data % n_clients]) # type:list
        proportions = np.split(end_idx, n_clients)
        if drop_last == False: # 尾巴随机分给n_clients
            tmp = np.random.choice(n_clients, len(last_idx), replace=True)
            for j in range(len(last_idx)):
                proportions[tmp[j]] = np.append(proportions[tmp[j]], last_idx[j])
    else:
        proportions = np.split(idxs, n_clients) # 平分为p份，type为list。

    # check_hash = [0] * 60000
    # for i in range(n_clients):
    #     foo = len(proportions[i])
    #     for j in range(foo):
    #         assert proportions[i][j] < 60000 and proportions[i][j] >= 0, 'out of index'
    #         check_hash[proportions[i][j]] += 1
    # for i in range(60000):
    #     assert check_hash[i] == 1, 'error'

    # add noise
    tmp = np.random.permutation(n_clients) + 1
    for i in range(0, n_clients):
        train_images[proportions[i]] += np.random.normal(loc=mean, scale=std) * tmp[i] / n_clients
    
    len_data = []
    data_map = {i: proportions[i] for i in range(n_clients)}
    for i in range(n_clients):
        len_data.append(len(data_map[i]))
        data_map[i] = data_map[i].tolist()
    if drop_last == False:
        assert sum(len_data) == num_data
    return data_map, len_data

def get_dataset(args):
    global train_images, train_labels, test_images, test_labels

    if args.dataset.upper() == 'fashion_mnist'.upper():
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data ()
    elif args.dataset.upper() == 'cifar10'.upper():
        (train_images, train_labels), (test_images, test_labels) = cifar10.load_data ()
    # normalize.
    train_images, test_images = train_images / 255.0, test_images / 255.0
    # convert to float32.
    train_images, test_images = train_images.astype (np.float32), test_images.astype (np.float32)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', type=str, default='iid', help='the data partitioning strategy')
    parser.add_argument('--dataset', type=str, default='fashion_mnist', help='the dataset name')
    parser.add_argument('--n_clients', type=int, default=15,  help='number of workers in a distributed cluster')
    parser.add_argument('--save_data', type=str, default='False', help="whether save data as xxx.npy")
    parser.add_argument('--savedir', type=str, required=False, default="../dataset/", help="dataset save path")
    parser.add_argument('--beta', type=float, default=1, help='The parameter for the dirichlet distribution')
    parser.add_argument('--noise_mean', type=float, default=0, help='The parameter for the feature skew by noise')
    parser.add_argument('--noise_std', type=float, default=0.01, help='The parameter for the feature skew by noise')
    # parser.add_argument('--shuffle', type=str, default='False', help="whether shuffle origin dataset")
    # parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    args = parser.parse_args()  # datatype : <class 'argparse.Namespace'>
    return args


if __name__ == '__main__':
    # !!!注意这里灵活修改（这里不包括aggregator）
    node_name = ['n'+str(i) for i in range(2, 16)] + ['p1'] 
    args = get_args()
    get_dataset(args)
    partition_strategy_factory(node_name, args)
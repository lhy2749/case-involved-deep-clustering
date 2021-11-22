# coding=utf8
"""
在VAE训练完了后，利用聚类损失进行微调
"""
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from model.dctc import DCTC
from utils.dataset import Mydataset
import torch
from torch.nn import functional as F
import numpy as np
from utils.config import model_config, train_config, data_config, eval_config
import pickle as pkl
from sklearn.cluster import KMeans
from utils.evaluation import eva
import scipy.sparse as sp
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def get_data():
    stop_words_path = './data/stop_words'
    f = open('./data/dic.pkl', 'rb')
    vocabulary_dic = pkl.load(f, encoding='latin1')
    f.close()
    vocabulary = ['<PAD>', '<CLS>', '<UNK>', '<EOT>']
    for k, v in vocabulary_dic.items():
        vocabulary.append(k)
    vocabulary_dic = {}
    for i in range(len(vocabulary)):
        vocabulary_dic[vocabulary[i]] = i
    with open('./data/y.txt', encoding='utf-8') as f:
        accusation = [i.strip() for i in f.readlines()]
    with open(stop_words_path, encoding='utf8') as f:
        stop_words = []
        for i in f:
            stop_words.append(i.strip())
    with open('./data/x_filte.txt', encoding='utf-8') as f:
        src = [i.strip() for i in f.readlines()]
    return src, accusation, stop_words, vocabulary_dic, vocabulary


def loss_function(x_bow, out, mu, var):
    BCE = -torch.sum(x_bow * F.log_softmax(out, dim=-1), dim=-1)
    KLD = -0.5 * torch.sum(1 + var - mu.pow(2) - var.exp())
    return torch.sum(BCE), torch.sum(KLD)


def get_topic(topic_vec, xh):
    # xh = xh.numpy()
    a = xh.shape[0]
    _, b = topic_vec.shape
    topic = np.zeros((a, b))
    xh = xh.tolist()
    for i in range(a):
        topic[i] = topic_vec[xh[i]]
    return topic


def preprocess_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.A


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized.A


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def train(dataset, train_config, eval_config, model_config):
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    #######################加载主题向量和邻接矩阵#########################
    f = open("./data/ind.news_topic_tword.adj", 'rb')
    adj = pkl.load(f, encoding='latin1')
    f.close()
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)  # 对称化
    features = sp.identity(adj.shape[0])  # 图卷积输入——————单位矩阵12652
    features = preprocess_features(features)
    support = [preprocess_adj(adj)]
    t_features = torch.tensor(features, dtype=torch.float)
    t_features = t_features.to(device)
    t_support = []
    for i in range(len(support)):
        t_support.append(torch.Tensor(support[i]).to(device))
    ####################################################################
    model = DCTC(vae_mid=model_config['vae_mid'],
                 num_words=model_config['num_words'],
                 vocab_size=model_config['vocab_size'],
                 bow_mid_hid=model_config['bow_mid_hid'],
                 seq_mid_hid=model_config['seq_mid_hid'],
                 seq_len=model_config['seq_len'],
                 num_heads=model_config['num_heads'],
                 dropout=model_config['dropout'],
                 is_traing=True,
                 support=t_support)
    model.load_state_dict(torch.load('./saves/params1.bin', map_location='cpu'))
    # model.cluster_layer.data = torch.tensor(np.load('./cluster_centers/center0.npy')).to(device)
    print("加载上次的完成")
    dataLoader = DataLoader(dataset, batch_size=train_config['batch_size'], shuffle=False)
    optimizer = optim.Adam(model.parameters(), lr=train_config['lr'])
    model.to(device)
    model.zero_grad()
    model.train()
    train_iterator = trange(1, int(train_config['epochs']), desc="TRAIN Epoch")
    for ep in train_iterator:
        print('\n')
        x_bow = dataLoader.dataset.x_bow
        x_seq = dataLoader.dataset.x_seq
        xh = dataLoader.dataset.xh
        mask = dataLoader.dataset.mask
        y_true = dataLoader.dataset.accusation
        y_true = list(map(int, y_true))
        x_bow = x_bow.to(device)
        x_seq = x_seq.to(device)
        mask = mask.to(device)
        out, mu, var, _, _, q = model(x_bow, x_seq, mask, device, xh, t_features)
        rc_loss, kl_loss = loss_function(x_bow, out, mu, var)
        vae_loss = (rc_loss + kl_loss) / 1000  # VAE loss
        tmp_q = q.data
        p = target_distribution(tmp_q)
        res1 = tmp_q.cpu().numpy().argmax(1)  # Q
        res2 = p.data.cpu().numpy().argmax(1)  # P
        e1 = eva(y_true, res1, str(ep) + 'Q')
        e2 = eva(y_true, res2, str(ep) + 'P')
        if (e1[2] > 0.90) or (e2[2] > 0.90):
            np.save("./saves/res1.npy", res1)
            np.save("./saves/res2.npy", res2)
            torch.save(model.state_dict(), './saves/params1.bin')
            break
        cluser_loss = F.kl_div(q.log(), p, reduction='batchmean')  # clustering loss
        lamda = 0.7
        total_loss = (1 - lamda) * (vae_loss / 10000) + lamda * cluser_loss
        print(str(ep) + "--cluster_loss:{:.5f}".format(cluser_loss.data))  # 0.3
        print(str(ep) + "--vae_loss:{:.5f}".format(vae_loss.item()))  # 34000
        print(str(ep) + "--totle_loss:{:.5f}".format(total_loss.item()))  # 34000
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config['clip_grad'])
        optimizer.step()
        model.zero_grad()


if __name__ == '__main__':

    src, accusation, stop_words, vocabulary_dic, vocabulary = get_data()
    print("vocabulary")
    dataset = Mydataset(src, accusation, model_config['seq_len'], vocabulary_dic, stop_word=stop_words,
                        vocabulary=vocabulary)
    if model_config['is_traing'] is not True:
        raise ValueError("Train flag must true before yor train model")
    train(dataset, train_config, eval_config, model_config)

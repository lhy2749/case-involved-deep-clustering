# coding=utf8
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import DCTC
from dataset import Mydataset
import torch
from torch.nn import functional as F
import numpy as np
from config import model_config, eval_config
import pickle as pkl
from sklearn.cluster import KMeans
from evaluation import eva
import scipy.sparse as sp
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def get_data():
    stop_words_path = './data/stop_words.txt'
    f = open('./data/dic.pkl', 'rb')
    vocabulary_dic = pkl.load(f, encoding='latin1')
    f.close()
    vocabulary = ['<PAD>', '<CLS>', '<UNK>', '<EOT>']
    for k, v in vocabulary_dic.items():
        vocabulary.append(k)
    vocabulary_dic = {}
    for i in range(len(vocabulary)):
        vocabulary_dic[vocabulary[i]] = i
    with open('./data/labels.txt', encoding='utf-8') as f:
        labels = [i.strip() for i in f.readlines()]
    with open(stop_words_path, encoding='utf8') as f:
        stop_words = []
        for i in f:
            stop_words.append(i.strip())
    with open('./data/news_filte.txt', encoding='utf-8') as f:
        src = [i.strip() for i in f.readlines()]
    return src, labels, stop_words, vocabulary_dic, vocabulary


def loss_function(x_bow, out, mu, var):
    BCE = -torch.sum(x_bow * F.log_softmax(out, dim=-1), dim=-1)
    KLD = -0.5 * torch.sum(1 + var - mu.pow(2) - var.exp())
    return torch.sum(BCE), torch.sum(KLD)


def get_topic(topic_vec, xh):
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


def eval_clustering_model(dataset, eval_config, model_config):
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    f = open("./data/ind.news_topic_element.adj", 'rb')
    adj = pkl.load(f, encoding='latin1')
    f.close()
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    features = sp.identity(adj.shape[0])
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
    model.load_state_dict(torch.load('./savers/params1.bin', map_location='cpu'))
    print("load model finished")
    dataLoader = DataLoader(dataset, batch_size=eval_config['batch_size'], shuffle=False)
    model.to(device)
    model.zero_grad()
    model.eval()
    iterator = tqdm(dataLoader, desc="Evaluate Iteration")
    all_out_list = []
    labels_list = []
    for step, batch in enumerate(iterator):
        x_bow, x_seq, mask, labels, xh = batch
        x_bow = x_bow.to('cpu')
        x_seq = x_seq.to('cpu')
        mask = mask.to('cpu')
        mu = model(x_bow, x_seq, mask, 'cpu', xh, t_features)[1].detach().numpy()  # out, mu, var, z, x
        all_out_list.extend(mu)
        labels_list.extend(labels)
    y_true = list(map(int, labels_list))
    x_km = np.array(all_out_list)
    km = KMeans(n_clusters=15)
    km.fit(x_km)
    eva(y_true, km.labels_)


if __name__ == '__main__':
    src, labels, stop_words, vocabulary_dic, vocabulary = get_data()
    print("start load latent feature...")
    dataset = Mydataset(src, labels, model_config['seq_len'], vocabulary_dic, stop_word=stop_words,
                        vocabulary=vocabulary)
    eval_clustering_model(dataset, eval_config, model_config)

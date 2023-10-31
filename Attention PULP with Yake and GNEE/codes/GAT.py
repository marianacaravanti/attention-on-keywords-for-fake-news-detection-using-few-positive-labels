from __future__ import division
from __future__ import print_function
import os, sys, glob, argparse
import random, torch, logging
import numpy as np 
import networkx as nx
from tqdm import tqdm
#from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, LabelEncoder
from os import listdir
from os.path import isfile, join
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import scipy.sparse as sp
from models import GAT, SpGAT
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer, LoggingHandler
language_model = SentenceTransformer('distiluse-base-multilingual-cased')
from event_graph_utils import regularization, process_event_dataset_from_networkx
from util.bibliotecas import *
from util.functions import *
import urllib.request
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

def regularization(G, dim, embedding_feature: str = 'embedding', iterations=15, mi=0.85):

    nodes = []
    # inicializando vetor f para todos os nodes
    for node in G.nodes():
        G.nodes[node]['f'] = np.array([0.0]*dim)
        if embedding_feature in G.nodes[node]:
            G.nodes[node]['f'] = G.nodes[node][embedding_feature]*1.0        
        nodes.append(node)
    
    pbar = tqdm(range(0, iterations))

    for iteration in pbar:
        random.shuffle(nodes)
        energy = 0.0

        # percorrendo cada node
        for node in nodes:
            f_new = np.array([0.0]*dim)
            f_old = np.array(G.nodes[node]['f'])*1.0
            sum_w = 0.0

            # percorrendo vizinhos do onde
            for neighbor in G.neighbors(node):
                w = 1.0
                if 'weight' in G[node][neighbor]:
                    w = G[node][neighbor]['weight']

                w /= np.sqrt(G.degree[neighbor])

                f_new += w*G.nodes[neighbor]['f']

                sum_w += w

            f_new /= sum_w

            G.nodes[node]['f'] = f_new*1.0

            if embedding_feature in G.nodes[node]:
                G.nodes[node]['f'] = G.nodes[node][embedding_feature] * \
                    mi + G.nodes[node]['f']*(1.0-mi)

            energy += np.linalg.norm(f_new-f_old)

        iteration += 1
        message = 'Iteration '+str(iteration)+' | Energy = '+str(energy)
        pbar.set_description(message)

    #sys.exit()
    return G

def process_event_dataset_from_networkx(G, features_attr="f"):
    """
    Builds an event graph dataset used in GAT model
    Parameters:
        G -> Graph representation of the event network (Networkx graph)
        df_labels -> user labeled data
        features_att -> Feature attribute of each node (str)
        random_state -> A random seed to train_test_split
    Returns:
        adj -> Sparse and symmetric adjacency matrix of our graph.
        features -> A NumPy matrix with our graph features.
        idx_train -> A NumPy array with the indexes of the training nodes.
        idx_val -> A NumPy array with the indexes of the validation nodes.
        idx_test -> A NumPy array with the indexes of the test nodes.
    """

    num_nodes = len(G.nodes)

    # validation_split_percentage = val_split / (1 - train_split)

    # df_val, df_test = train_test_split(
    #     df_test_and_val, train_size=validation_split_percentage, random_state=random_state)

    # Organizing our feature matrix...
    # feature_matrix = np.array([ G.nodes[i]['embedding'] if 'embedding' in G.nodes[i].keys() else G.nodes[i][features_attr] for i in G.nodes()])
    #features = np.array([G.nodes[i][features_attr] for i in G.nodes()])
    L_features = []
    L_train = []
    L_test = []
    L_labels = []
    label_codes = {}
    for node in G.nodes():
      L_features.append( (G.nodes[node]['id'], G.nodes[node]['f']) )
      if 'train' in G.nodes[node]: 
        L_train.append(G.nodes[node]['id'])
        #print('entrei aqui 1')
      if 'test' in G.nodes[node]: 
        L_test.append(G.nodes[node]['id'])
        #print('entrei aqui 2')
      if 'label' in G.nodes[node]:
        #print('entrei aqui 3')
        if G.nodes[node]['label'] not in label_codes: label_codes[G.nodes[node]['label']] = len(label_codes)
        L_labels.append( [G.nodes[node]['id'],G.nodes[node]['label'],label_codes[G.nodes[node]['label']]] )
    print(label_codes) #{'real':0, 'fake':1}
    real_code = label_codes['real']
    fake_code = label_codes['fake']
    print('real:', real_code, 'fake', fake_code)
    #sys.exit()
    #print(L_labels) [[0, 'real', 0], [1, 'real', 0] ... [198, 'fake', 1], [199, 'fake', 1]]
    #print(L_train)
    #print(L_test)
    
    df_features = pd.DataFrame(L_features)

    df_features.columns = ['node_id','embedding']
    #print(df_features)
    features = np.array(df_features.sort_values(by=['node_id'])['embedding'].to_list())

    #print('l_train:', L_train)
    #print('Label_codes: ', label_codes)

    idx_train = L_train
    idx_test = L_test
    labels = [-1]*num_nodes
    #print(labels)

    df_labels = pd.DataFrame(L_labels)
    df_labels.columns = ['news_id','label','label_code']
    #print(df_labels)
    #sys.exit()
    for index,row in df_labels.iterrows():
      labels[row['news_id']] = row['label_code']
    adj = nx.adjacency_matrix(G)
    
    return adj, features, labels, idx_train, idx_test, df_labels, real_code, fake_code


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

class Namespace(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


class GAT_wrapper():
    def __init__(self, args={"alpha": 0.2, "cuda": True, "dropout": 0.6, "epochs": 10, "fastmode": False, "hidden": 8, "lr": 0.005, "nb_heads": 1, "no_cuda": False, "patience": 100, "seed": 72, "sparse": False, "weight_decay": 0.0005}):

        if (type(args) == dict):
            args = Namespace(args)

        self.args = args

        self.model = None

        self.loss_test = 0.0
        self.acc_test = 0.0

        self.adj = None
        self.features = None
        self.labels = None
        self.idx_train = None
        self.idx_val = None
        self.idx_test = None

    def compute_test(self):
        self.model.eval()
        output = self.model(self.features, self.adj)
        loss_test = F.nll_loss(
            output[self.idx_test], self.labels[self.idx_test])
        acc_test = accuracy(output[self.idx_test], self.labels[self.idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))

        self.loss_test = loss_test
        self.acc_test = acc_test

        return loss_test, acc_test, output[self.idx_test].max(1)[1]

    def train_pipeline(self, adj, features, labels, idx_train, idx_val, idx_test, *args):

        adj = normalize_adj(adj + sp.eye(adj.shape[0]))

        if (sp.issparse(adj)):
            adj = adj.todense()

        if (sp.issparse(features)):
            features = features.todense()

        # With networkx, we no longer need to convert from one-hot encoding...
        #labels = np.where(labels)[1]

        adj = torch.FloatTensor(adj)
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(labels)
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if self.args.cuda:
            torch.cuda.manual_seed(self.args.seed)

        # Load data
        # adj, features, labels, idx_train, idx_val, idx_test = new_load_data(
        #     *args, custom_function=custom_function, function=function)

        # Model and optimizer
        print(features.shape[1])
        if self.args.sparse:
            model = SpGAT(nfeat=features.shape[1],
                          nhid=self.args.hidden,
                          nclass=int(labels.max()) + 1,
                          dropout=self.args.dropout,
                          nheads=self.args.nb_heads,
                          alpha=self.args.alpha)
        else:
            model = GAT(nfeat=features.shape[1],
                        nhid=self.args.hidden,
                        nclass=int(labels.max()) + 1,
                        dropout=self.args.dropout,
                        nheads=self.args.nb_heads,
                        alpha=self.args.alpha)
        optimizer = optim.Adam(model.parameters(),
                               lr=self.args.lr,
                               weight_decay=self.args.weight_decay)

        if self.args.cuda:
            model.cuda()
            features = features.cuda()
            adj = adj.cuda()
            labels = labels.cuda()
            idx_train = idx_train.cuda()
            idx_val = idx_val.cuda()
            idx_test = idx_test.cuda()

        features, adj, labels = Variable(
            features), Variable(adj), Variable(labels)

        # TODO: Test if these lines could be written below line 41.
        self.adj = adj
        self.features = features
        self.labels = labels
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test

        def train(epoch):
            t = time()
            model.train()
            optimizer.zero_grad()
            output = model(features, adj)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            acc_train = accuracy(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            if not self.args.fastmode:
                # Evaluate validation set performance separately,
                # deactivates dropout during validation run.
                model.eval()
                output = model(features, adj)
                #print(type(features))
                #print('segundo', features)
                #sys.exit()

            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])
            print('Epoch: {:04d}'.format(epoch+1),
                  'loss_train: {:.4f}'.format(loss_train.data.item()),
                  'acc_train: {:.4f}'.format(acc_train.data.item()),
                  'loss_val: {:.4f}'.format(loss_val.data.item()),
                  'acc_val: {:.4f}'.format(acc_val.data.item()),
                  'time: {:.4f}s'.format(time() - t))

            return loss_val.data.item()

        # Train model
        t_total = time()
        loss_values = []
        bad_counter = 0
        best = self.args.epochs + 1
        best_epoch = 0
        for epoch in range(self.args.epochs):
            loss_values.append(train(epoch))

            torch.save(model.state_dict(), '{}.pkl'.format(epoch))
            if loss_values[-1] < best:
                best = loss_values[-1]
                best_epoch = epoch
                bad_counter = 0
            else:
                bad_counter += 1

            if bad_counter == self.args.patience:
                break

            files = glob.glob('*.pkl')
            for file in files:
                epoch_nb = int(file.split('.')[0])
                if epoch_nb < best_epoch:
                    os.remove(file)

        files = glob.glob('*.pkl')
        for file in files:
            epoch_nb = int(file.split('.')[0])
            if epoch_nb > best_epoch:
                os.remove(file)

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time() - t_total))

        # Restore best model
        print('Loading {}th epoch'.format(best_epoch))
        model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

        self.model = model

        return model


exp_id = sys.argv[1]
pulp_id = sys.argv[2]
indexes = np.load('dataset/indexes.npy')
labels = np.load('dataset/labels.npy')
doc_relations_file = sys.argv[3]
net_relations = doc_relations_file.split('/')[5]
labels_file = sys.argv[4]
dir_bow_op = sys.argv[5]
attention_heads = int(sys.argv[6])

exp_metadata = pd.read_csv('params.metadata', sep='\t', index_col=0)
exp_metadata = exp_metadata.loc[exp_id]

pulp_metadata = pd.read_csv(dir_bow_op+'params_pulp.metadata', sep='\t', index_col=0)
pulp_metadata = pulp_metadata.loc[pulp_id]

gat_metadata = pd.read_csv('GAT.metadata', sep='\t')
gat = gat_metadata.iloc[0]
alpha = gat.alpha
cuda = gat.cuda
dropout = gat.dropout
epochs = gat.epochs
fastmode = gat.fastmode
hidden = gat.hidden
lr = gat.lr 
no_cuda = gat.no_cuda
patience = gat.patience
seed = gat.seed
sparse = gat.is_sparse
weight_decay = gat.weight_decay


G = nx.Graph()

with open(doc_relations_file) as f:
  for row in f:
    relation = row.strip
    x, y, weight = row.split('\t')
    weight = float(weight)
    
    G.add_edge(x, y, weight=weight)

doc2vec_rep = load_representation(exp_metadata.representation_model)
doc2vec_txt = doc2vec_rep.text_vectors

counter = 0
fold = str(pulp_metadata.fold)
train_folds = fold.split(',')
train_folds_list = []

for i in train_folds:
    file = 'folds/fold'+str(i)
    f = open(file, 'r')
    for row in f:
        index = row.replace('\n','')
        train_folds_list.append(index)
    f.close()

#Marcando no grafo quais são os nós de treinamento
with open(labels_file) as f:
  for row in f:
    relation = row.strip()
    node, infoclass = relation.split('\t')
    node_index = node.split(':')[0]
    infofake, inforeal = infoclass.split(',')
    infofake, inforeal = int(infofake), int(inforeal)
    G.nodes[node]['train'] = 1
    if infofake > inforeal: G.nodes[node]['label']='fake'
    else: G.nodes[node]['label']='real'
    if not (node_index in train_folds_list): 
        G.nodes[node]['y_pred_pulp'] = 1
        index_position = np.where(indexes == node.split(':')[0])
        label = int(labels[index_position[0]])
        if (label == 1): G.nodes[node]['real_class'] = 'fake'
        else:G.nodes[node]['real_class'] = 'real'

#Marcando labels no grafo e quais são os dados de teste
for node in G.nodes():
  G.nodes[node]['id'] = counter
  counter+=1
  if ':news' in node:
    id_news = node.split(':')[0]     
    index_position = np.where(indexes == id_news)
    label = int(labels[index_position[0]])
    feature = doc2vec_txt[index_position[0]]
    if not ('label' in G.nodes[node]):
        if (label == 1): G.nodes[node]['label'] = 'fake'
        else: G.nodes[node]['label'] = 'real'
    G.nodes[node]['features'] = feature[0]
    if not ('train' in G.nodes[node]): G.nodes[node]['test'] = 1    


experimental_results = []

regularization(G, 1000, embedding_feature='features')

adj, features, labels, idx_train, idx_test, df_labels, real_code, fake_code = process_event_dataset_from_networkx(G)
print(type(features))
print(features)
#sys.exit()
print(adj.shape,features.shape,len(idx_train),len(idx_test))

gat = GAT_wrapper({"alpha": alpha, "cuda": cuda, "dropout": dropout, "epochs": epochs, "fastmode": fastmode, "hidden": hidden, "lr": lr, "nb_heads": attention_heads, "no_cuda": no_cuda, "patience": patience, "seed": seed, "sparse": sparse, "weight_decay": weight_decay})
gat.train_pipeline(adj, features, labels, idx_train, idx_train, idx_test)


embedding = gat.model.get_attention_heads_outputs(gat.features,gat.adj).cpu().detach().numpy()
counter = 0
X = []
Y = []
for node in G.nodes():
  if ':news' in node:
    X.append(embedding[counter])
    Y.append(G.nodes[node]['label'])

  counter += 1

X = np.array(X)



import numpy as np
from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2).fit_transform(X)
X_embedded.shape

df = pd.DataFrame(X_embedded)
df['label'] = Y
df = df.dropna()

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

g = sns.scatterplot(x=0, y=1, data=df, hue="label", legend=True)
g.set(xlabel=None)
g.set(ylabel=None)

plt.savefig('GNEE.pdf')

loss, acc, output = gat.compute_test()
y_pred = output.cpu().numpy()
y_pred = y_pred.tolist()
y_true = []
y_label = []


for news_id in idx_test:
  for node in G.nodes():
    if ':news' in node:
      if G.nodes[node]['id'] == news_id:
        y_true.append(df_labels[df_labels.news_id==news_id].label_code.values[0])
        y_label.append(df_labels[df_labels.news_id==news_id].label.values[0])

for node in G.nodes():
    if 'y_pred_pulp' in G.nodes[node]:
        idx_test.append(G.nodes[node]['id'])
        y_true.append(fake_code if G.nodes[node]['real_class'] == 'fake' else real_code)
        y_pred.append(fake_code if G.nodes[node]['label'] == 'fake' else real_code)
        y_label.append(G.nodes[node]['real_class'])        

matriz_confusao = pd.DataFrame(0, columns={'pred_pos', 'pred_neg'},index={'classe_pos', 'classe_neg'})

for l in range(len(y_pred)):
    rotulo_original =  'classe_pos' if y_true[l] == fake_code else 'classe_neg'
    predito = 'pred_pos' if y_pred[l] == fake_code else 'pred_neg'
    matriz_confusao[predito][rotulo_original] += 1

string_confusao = 'pred_pos class_pos: {}, pred_neg class_neg: {}, pred_pos class_neg: {}, pred_neg class_pos: {}'.format(str(matriz_confusao['pred_pos'].loc['classe_pos']),
    str(matriz_confusao['pred_neg'].loc['classe_neg']), str(matriz_confusao['pred_pos'].loc['classe_neg']), str(matriz_confusao['pred_neg'].loc['classe_pos']))
print(string_confusao)


#matriz_confusao.to_csv('results/confusion/confusion_gat_{}.csv'.format(pulp_id), sep='\t')
positive_total = matriz_confusao['pred_pos'].sum() #total de noticias classificadas como positivo
true_positive_news = matriz_confusao['pred_pos'].loc['classe_pos'].sum()

f = open('results/avaliados.csv', 'a')
id = pulp_id
dataset = exp_metadata.dataset
representation = exp_metadata.representation_model
stopwords = exp_metadata.stopwords
language =  exp_metadata.language
option = str(exp_metadata.option)
deduplication_algo = str(exp_metadata.deduplication_algo)
window_size = str(exp_metadata.window_size)
numOfKeywords = str(exp_metadata.numOfKeywords)
n_gram = str(exp_metadata.n_gram)
#arg_min_weigth = str(exp_metadata.arg_min_weigth) 


fold = str(pulp_metadata.fold)
k = str(pulp_metadata.k)
a = str(pulp_metadata.a)
m = str(pulp_metadata.m)
l = str(pulp_metadata.l)


list_params_exp = "\t".join([id, dataset, representation, stopwords, language, option, deduplication_algo, window_size, numOfKeywords, n_gram, net_relations])
list_params_pulp = "\t".join([fold, k, a, m, l])
list_params_gat = "\t".join([str(alpha), str(cuda), str(dropout), str(epochs), str(fastmode), str(hidden), str(lr), str(no_cuda), str(patience), str(seed), str(sparse), str(weight_decay)])

TP = matriz_confusao['pred_pos'].loc['classe_pos'].sum()
FP = matriz_confusao['pred_pos'].loc['classe_neg'].sum()
TN = matriz_confusao['pred_neg'].loc['classe_neg'].sum()
FN = matriz_confusao['pred_neg'].loc['classe_pos'].sum()
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = (2*precision*recall)/(precision+recall)
fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=fake_code)
auc_fake = auc(fpr, tpr)
fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=real_code)
auc_real = auc(fpr, tpr)
f.write("\t".join(['gat', list_params_exp, list_params_pulp, list_params_gat,
    str(f1_score(y_true, y_pred, average='macro')), str(f1_score(y_true, y_pred, average='micro')), 
    str(accuracy_score(y_true, y_pred)), str(precision_score(y_true, y_pred, average='macro')), 
    str(precision_score(y_true, y_pred, average='micro')), str(recall_score(y_true, y_pred, average='macro')), 
    str(recall_score(y_true, y_pred, average='micro')), str(true_positive_news/positive_total), 
    str(precision), str(recall), str(f1), str(auc_fake), str(auc_real), string_confusao])+'\n')

f.close()

print('Macro f1:', f1_score(y_true, y_pred, average='macro'))
print('Micro f1:', f1_score(y_true, y_pred, average='micro'))
print('accuracy:', accuracy_score(y_true, y_pred))
print('Macro Precision:', precision_score(y_true, y_pred, average='macro'))
print('Micro Precision:', precision_score(y_true, y_pred, average='micro'))
print('Macro Recall:', recall_score(y_true, y_pred, average='macro'))
print('Micro Recall:', recall_score(y_true, y_pred, average='micro'))
print('True Positive:', true_positive_news/positive_total)
print('Interest-class precision: ', precision)
print('Interest-class recall:', recall)
print('Interest-class f1:', f1)

del gat
del adj
del features
del G

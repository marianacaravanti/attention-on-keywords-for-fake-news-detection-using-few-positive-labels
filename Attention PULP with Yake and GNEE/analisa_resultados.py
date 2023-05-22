import numpy as np
import pandas as pd
import networkx as nx
import os, sys


file = 'results/avaliados.csv'

list_index = {0:'algprop', 1:'id', 2:'dataset', 3:'exp_name', 4:'stopwords', 5:'lang', 6:'option', 7:'deduplication_algo', 8:'window_size', 9:'numOfKeywords', 10:'n_gram', 11:'network', 12:'fold', 13:'k', 14:'a', 15:'m', 
16:'l', 17:'alpha', 18:'cuda',19:'dropout', 20:'epochs', 21:'fastmode', 22:'hidden', 23:'lr', 24:'no_cuda', 25:'patience', 26:'seed', 27:'sparse', 28:'weight_decay',29:'macro_f1', 30:'micro_f1', 31:'acuracy', 
32:'macro_precision', 33:'micro_precision', 34:'macro_recall', 35:'micro_recall', 36:'tp', 37:'interest_prec', 38:'interest_recall', 39:'interest_f1', 40:'auc_fake', 41:'auc_real', 42:'string_confusao'}

df = pd.read_csv(file, sep='\t', index_col=None, header=None, dtype={13: str}).rename(columns=list_index).fillna(0)
df['training_percent'] = df.fold.astype('str').str.len() 
metric = 'interest_f1'
num_labeles_exes = df['training_percent'].unique()
df_groups = []
num_algs = ['gat']

nets = df['network'].unique()
table_results = []
print('>>>', metric)
for net in nets:
	net_result = []
	net_result.append(net)
	for nalg in num_algs:
		for nle in num_labeles_exes:
			df_selected = df[df['network'] == net]
			df_selected = df_selected[df_selected['training_percent'] == nle]
			df_selected = df_selected[df_selected['algprop'] == nalg]
			df_grouped = df_selected.sort_values(by=['dataset', 'exp_name', 'numOfKeywords', 'n_gram', 'k', 'a', 'm', 'l', 'fold'])
			df_grouped = df_grouped.drop( ['fold', 'id'], axis=1).groupby(['dataset', 'exp_name', 'numOfKeywords', 'n_gram', 'k', 'a', 'm', 'l']).mean(numeric_only=True)
			net_result.append(df_grouped[metric].max())
	table_results.append(net_result)

df_results = pd.DataFrame(table_results, columns=['network','gat_10', 'gat_20', 'gat_30'])
print(df_results)

table_results = []

metric = 'macro_f1'
print('>>>', metric)
for net in nets:
	net_result = []
	net_result.append(net)
	for nalg in num_algs:
		for nle in num_labeles_exes:
			df_selected = df[df['network'] == net]
			df_selected = df_selected[df_selected['training_percent'] == nle]
			df_selected = df_selected[df_selected['algprop'] == nalg]
			df_grouped = df_selected.sort_values(by=['dataset', 'exp_name', 'numOfKeywords', 'n_gram', 'k', 'a', 'm', 'l', 'fold'])
			df_grouped = df_grouped.drop(['fold', 'id'], axis=1).groupby(['dataset', 'exp_name', 'numOfKeywords', 'n_gram', 'k', 'a', 'm', 'l']).mean(numeric_only=True)
			net_result.append(df_grouped[metric].max())
	table_results.append(net_result)


df_results = pd.DataFrame(table_results, columns=['network','gat_10','gat_20','gat_30'])
print(df_results)


import sys, json, pickle
import numpy as np
from hashlib import md5
from util.functions import *
import pandas as pd

#Bag-of-words parameters example
representation_model = "dataset/rep4.rep"
arg_tsv_file = "dataset/Fact_checked_news.tsv"
dataset_file = "dataset/complete dataset.pkl"
arg_stopwords = "dataset/stopwords.txt"
arg_language = "portuguese"
folds = 10
arg_options = ['2']

#Yake parameters
deduplication_treshold = 0.9
deduplication_algo = 'seqm'
window_size = 1
numOfKeywords = 25
n_gram = 3


#k-nn matrix parameters
arg_k = ['5', '6','7']
#W matrix parameters
arg_a = ['0.005', '0.1']

#Parâmetros do PU-LP
arg_l = [0.6, 0.8]
arg_m = [2]


#Params GAT
alpha = 0.2
cuda = False
dropout = 0.5
epochs = 20 
fastmode = False
hidden = 8
lr = 0.005
no_cuda = False
patience = 100
seed = 72
is_sparse = True
weight_decay = 0.0005

output_file_dict = 'dataset/'
dir_base = 'scripts/'
dir_bow = 'PULP/op={}/'
dir_bow_op = dir_bow+'{}/'
dir_pulp_output = dir_bow_op+'pulp_{}0%/'

dir_results = 'results/confusion/'
dir_repr_input = dir_bow_op + 'representation_input/'
dir_graph = dir_repr_input + '{}/'
dir_adj_matrix = dir_repr_input + 'adj_matrix.csv'
dir_matrix = dir_graph + '{}/'
relations = dir_graph + '{}.relations' 

graph = dir_graph + 'knn_fakenews.graphml'
w_matrix = dir_matrix + 'w_matrix.csv'
dir_fila_bow = dir_base + 'fila_pre/'
dir_fila_pu_lp = dir_base + 'fila_pulp/'
dir_params_pulp = dir_bow_op + 'params_pulp.metadata'
#dir_params_prop = dir_bow_op + 'params_labelprop.metadata'

create_path(dir_base)
create_path(dir_fila_bow)
create_path(dir_fila_pu_lp)
create_path('logs')
create_path('folds')
nucleos_bow_adj = 1
multiple = sys.argv[1]
nucleos_bow_adj = int(nucleos_bow_adj)
multiple = int(multiple) 
nucleos_pulp = multiple

params = open('params.metadata', 'w')
params.write('id\tdataset\trepresentation_model\tstopwords\tlanguage\toption\tdeduplication_treshold\tdeduplication_algo\twindow_size\tnumOfKeywords\tn_gram\ttsv_dataset\n')
net_metadata = pd.read_csv('params_features.metadata', sep='\t', header=0)

params_gat = open('GAT.metadata', 'w')
params_gat.write('alpha\tcuda\tdropout\tepochs\tfastmode\thidden\tlr\tno_cuda\tpatience\tseed\tis_sparse\tweight_decay\n')
params_gat.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(alpha,cuda,dropout,epochs,fastmode,hidden,lr,no_cuda,patience,seed,is_sparse,weight_decay))
params_gat.close()

dataset = {'1': {}, '-1': {}}
f = open(arg_tsv_file, 'r')

for line in f:
	index, text, label = line.split('\t')
	label = label.replace('\n', '')
	dataset[label][index] = text
f.close()


count = 0
for index in dataset['1']:
	fold = count%folds
	f = open('folds/fold'+str(fold), 'a')
	f.write(index + "\n")
	f.close()
	count+=1

dict_file = open(output_file_dict+'dict_dataset.pkl', 'wb')
pickle.dump(dataset, dict_file)
dict_file.close()


count = 0
count_pulp = 0

list_fila_dir = []
list_fila_pulp = []
for op in arg_options:
	dir = count % nucleos_bow_adj
	fbow = dir_fila_bow + 'fila_{}.sh'.format(dir)	
	
	if not dir in list_fila_dir: 
		list_fila_dir.append(dir)
		fb = open(dir_fila_bow+'fila_pre.sh', 'a')
		fb.write('nohup {} > logs/fila_pre_{}.log &\n'.format(fbow, dir))	
		fb.close()

		
	
	exp_id = md5('{} {} {} {} {} {} {} {} {} {} {}'.format(dataset_file, representation_model, arg_stopwords, arg_language,
		op, deduplication_treshold, deduplication_algo, window_size, numOfKeywords, n_gram, arg_tsv_file).encode()).hexdigest()
	
	params.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(exp_id, dataset_file, representation_model, 
		arg_stopwords, arg_language, op, deduplication_treshold, deduplication_algo, window_size, numOfKeywords, n_gram, arg_tsv_file))

	f = open(fbow, 'a')
	f.write("python3 preprocess.py {} > logs/{}-bow.log\n".format(exp_id, exp_id))
	f.close()							
	
	f = open(fbow, 'a')
	f.write("python3 adjacency_matrix.py {} {} {} {} > logs/{}-adj.log\n".format(exp_id, dir_bow_op, ','.join(arg_k),','.join(arg_a), exp_id))
	f.close()
	
	count+=1
	

	create_path(dir_bow_op.format(op,exp_id))
	
	params_pulp = open(dir_params_pulp.format(op, exp_id), 'w')
	params_pulp.write('id\tdataset\trepresetation\tfold\tk\ta\tm\tl\n')


	count_multiple = 0
	for p in range(3):

		create_path(dir_pulp_output.format(op,exp_id,p+1))

		for k in arg_k:
			input_graph = dir_graph.format(op, exp_id, k)
			for a in arg_a:
				path_dir_matrix = dir_matrix.format(op,exp_id,k,a)	
				input_matrix_w = w_matrix.format(op, exp_id, k, a)
				create_path(path_dir_matrix)
				
				fold = ''										
							
				for i in range(10):

					for m in arg_m:

						for l in arg_l:													

							if p == 0: fold = '{}'.format(i%10)

							if p == 1: fold = '{},{}'.format(i%10, (i+1)%10)
								
							if p == 2: fold = '{},{},{}'.format(i%10, (i+1)%10, (i+2)%10)
					
							pulp_id = md5('{} {} {} {} {} {} {}'.format(arg_tsv_file, representation_model, fold, k, a, m, l).encode()).hexdigest()
							params_pulp.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(pulp_id, arg_tsv_file, representation_model, fold, k, a, m, l))
							

							dir_pulp = count_pulp % nucleos_bow_adj
							dir_pulp_multiple = count_multiple % multiple
							pu_lp_output = dir_pulp_output+'{}.labels'
							pu_lp_output = pu_lp_output.format(op, exp_id, p+1, pulp_id)
							file_name = dir_fila_pu_lp+'fila_{}_{}.sh'.format(dir_pulp, dir_pulp_multiple)
							
							f = open(file_name, 'a')
							f.write("python3 PU-LP.py {} {} {} {}  > logs/{}_{}-pulp.log\n". format(pulp_id, input_graph, input_matrix_w, pu_lp_output, exp_id, pulp_id))
							

							create_path(input_graph)

			
							for net in range(len(net_metadata)):
								network_name_relation = dir_graph.format(op, exp_id, k) + net_metadata.network_name[net]
								network_relation = net_metadata.network_name[net].split('.')[0]
								attention_heads = net_metadata.attention_heads[net]
								f = open(file_name, 'a')

								output_dir_bow = dir_bow.format(op)
								f.write("python3 GAT.py {} {} {} {} {} {} > logs/{}-gat.log\n".format(exp_id, pulp_id, relations.format(op, exp_id, k, network_relation), 
									pu_lp_output, dir_bow_op.format(op, exp_id), attention_heads, pulp_id)) 

								
								f.write('\necho "---------- proximo ----------"\n\n')
								
								f.close()
							count_multiple +=1
				
				count_pulp += 1	
											

count_pulp = 0
list_fila_pulp = []

for op in arg_options:
	count_multiple = 0
	for p in range(3):
		for k in arg_k:
			for a in arg_a:
				for i in range(10):
					for m in arg_m:
						for l in arg_l:	
							dir_pulp = count_pulp % nucleos_bow_adj
							dir_pulp_multiple = count_multiple % multiple
							string = str(dir_pulp)+'_'+str(dir_pulp_multiple)
							if not string in list_fila_pulp: 
								list_fila_pulp.append(string)
								dir_pulp = count_pulp % nucleos_bow_adj
								dir_pulp_multiple = count_multiple % multiple
								output_fila_bow = dir_fila_bow + 'fila_{}.sh'.format(dir_pulp)	
								fb = open(output_fila_bow, 'a')
								file_name = dir_fila_pu_lp+'fila_{}_{}.sh'.format(dir_pulp, dir_pulp_multiple)
								fb.write('nohup {} > logs/fila_pulp_{}_{}.log &\n'.format(file_name, dir_pulp, dir_pulp_multiple))	
								fb.close()
							
								count_multiple += 1
	count_pulp += 1	
						

							
params.close()							
params_pulp.close()		
				
							

					
							

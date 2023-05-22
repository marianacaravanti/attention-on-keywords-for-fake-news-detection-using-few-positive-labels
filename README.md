# Attention on Keywords for Fake News Detection using few positive labels

- Mariana Caravanti de Souza (ICMC/USP)
- Marcos Paulo Silva Gôlo (ICMC/USP)
- Alípio Mário Guedes Jorge (DCC/FCUP)
- Evelin Carvalho Freire de Amorim (DCC/FCUP)
- Ricardo Campos (Polytechnic Institute of Tomar)
- Ricardo Marcondes Marcacini (ICMC/USP)
- Solange Oliveira Rezende (ICMC/USP)

# Abstract

To minimize the problem of fake news, approaches proposed in the literature generally learn models through supervised multiclass algorithms for classifying news at different levels of falsehood. However, labeling a significant news set for effectively extracting features that discriminate between real and false content is costly. In this context, we propose improvements in the Positive and Unlabeled Learning by Label Propagation (PU-LP) algorithm. PU-LP is a Positive and Unlabeled Learning (PUL) approach based on similarity networks, which have achieved competitive results in fake news detection incorporating relations between news and representative terms in the network, using only 10% to 30% of labeled fake news. In particular, we propose integrating an attention mechanism in PU-LP that can define which terms in the network are more relevant to discriminate between real and false content. For that, we use GNEE, a state-of-the-art classification algorithm based on graph attention heterogeneous neural networks. The results achieved by our approach are very close to the binary semi-supervised baseline, even with nearly half of the data labeled. It surpasses state-of-the-art classification algorithms of one class based on Graph Neural Networks, improving from 2% to 10% the F1 of the PU-LP originally proposed for classifying fake news, especially using 10% of labeled data. We also present an analysis of the approach's limitations, relating the results achieved with the type of text found in each fake news dataset.

# Datasets

The datasets are available from a google drive link in the Heterogeneous PU-LP/Datasets folder.

# Proposed Approach: Attention PU-LP for Fake News Detection

![proposedapproach_page-0001](https://github.com/marianacaravanti/attention-on-keywords-for-fake-news-detection-using-few-positive-labels/assets/8595261/269fe1ac-6f23-4299-b215-202df7b19ed2)

# Features of each news Dataset

![image](https://github.com/marianacaravanti/attention-on-keywords-for-fake-news-detection-using-few-positive-labels/assets/8595261/08f258a3-dc75-4571-a6e1-df8b1dab0653)

# Results

![Desenho sem título (5)](https://github.com/marianacaravanti/attention-on-keywords-for-fake-news-detection-using-few-positive-labels/assets/8595261/3b18e953-8ba2-4864-acef-8d219c138573)

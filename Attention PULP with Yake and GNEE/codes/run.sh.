#Python==3.8.16 
pip3 install numpy==1.24.2 scipy==1.10.0 scikit-learn==1.2.1 pandas==1.1.3 networkx==3.0 gensim==4.3.0 nltk==3.8.1 yake==0.4.8 seaborn==0.12.2  
pip3 install testresources
pip3 install --quiet spektral
pip3 install gdown
pip3 install git+https://github.com/rmarcacini/sentence-transformers
gdown https://drive.google.com/uc?id=1NV5t1YhyyOzMF5zAovfbSLdZZLvqrfZ_
sudo apt install unzip
unzip distiluse-base-multilingual-cased.zip -d language_model
mkdir -p results

#$1 is the number of threads after preprocessing phase
python3 create_scripts.py $1 
chmod -R 777 scripts
nohup ./scripts/fila_bow/fila_bow.sh &

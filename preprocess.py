from util.bibliotecas import *
from util.functions import *
import yake

def convert(vector, type):
    vector = [type(i) for i in vector]
    return vector

def domain_stopwords(stopwords=None):
    #get domain stopwords
    domain_stopwords = ""
    if stopwords==None:
        return domain_stopwords
    f = open(stopwords, "r")
    for line in f:
        domain_stopwords +=  line.strip() + " "
    f.close()
    return word_tokenize(domain_stopwords)


class BoW():
    def __init__(self, exp_id, exp_metadata):
        self.exp_id = exp_id
        self.dataset = exp_metadata.dataset  #dataset.tsv
        self.stopwords = exp_metadata.stopwords #stopwords.txt
        self.language = exp_metadata.language  #portuguese or english
        stop_words = stopwords.words(self.language)
        self.option = exp_metadata.option #1 (remove stopwords) or 2 (stopwords and stemming)
        
        self.deduplication_treshold = exp_metadata.deduplication_treshold
        self.deduplication_algo =  exp_metadata.deduplication_algo
        self.window_size = exp_metadata.window_size
        self.numOfKeywords = exp_metadata.numOfKeywords
        self.n_gram = exp_metadata.n_gram
        
        #self.output_file = sys.argv[2]
        self.domain_stopwords = domain_stopwords(self.stopwords)
        self.documents = None
        self.indexes = None
        self.labels = None
        self.keywords = []
        self.preprocess() 
       
    def preprocess(self):
        
        self.labels = []
        self.documents = []
        self.indexes = []
        self.keywords = []

        #Read dicts and load in a numpy list
        documents = pickle.load(open(self.dataset, "rb"))
        indexes = np.load('dataset/indexes.npy')
        labels = np.load('dataset/labels.npy')

        for i in range(len(indexes)):
            index = indexes[i]
            label = labels[i]
            text = documents[index]
            tokens = word_tokenize(text)
            percent_yake = int(self.numOfKeywords)
            if self.language == 'portuguese': yake_lan = 'pt'
            else: yake_lan = 'en'
            custom_kw_extractor = yake.KeywordExtractor(lan=yake_lan, n=self.n_gram, dedupLim=self.deduplication_treshold, dedupFunc=self.deduplication_algo, windowsSize=self.window_size, top=percent_yake, features=None)
            keywords = custom_kw_extractor.extract_keywords(text)
            
            self.keywords.append(keywords)
            self.documents.append(text.strip())
            self.indexes.append(index)
        
        #np.save('dataset/indexes',self.indexes)
        #np.save('dataset/documents',self.documents)
        #np.save('dataset/labels',self.labels)
        self.keywords = np.array(self.keywords, dtype=object)
        np.save('dataset/keywords', self.keywords)

        #f.close()
        

    def remove_stopwords(self,text):
        stop_words = nltk.corpus.stopwords.words(self.language)
        s = str(text).lower() # lower case
        table = str.maketrans({key: None for key in string.punctuation})
        s = s.translate(table) # remove punctuation
        tokens = word_tokenize(s) #get tokens
        v = []
        for i in tokens:
            if not (i in stop_words or i in self.domain_stopwords or i.isdigit() or len(i)<= 1): # remove stopwords
                v.append(i)
        s = ""
        for token in v:
            s += token + " "
        return s

    def stemming(self, text):
        stemmer = PorterStemmer() # stemming for English
        if self.language=='portuguese':
            stemmer = nltk.stem.RSLPStemmer() # stemming for portuguese
        tokens = word_tokenize(text) 
        sentence_stem = ''
        doc_text_stems = [stemmer.stem(i) for i in tokens]
        for stem in doc_text_stems:
            sentence_stem += stem+" "
        return sentence_stem.strip()
                                 

exp_id = sys.argv[1]
exp_metadata = pd.read_csv('params.metadata', sep='\t', index_col=0)
exp_metadata = exp_metadata.loc[exp_id]

bow = BoW(exp_id, exp_metadata)



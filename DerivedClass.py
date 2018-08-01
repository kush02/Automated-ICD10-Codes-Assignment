import sys
sys.dont_write_bytecode = True

from TextFile import TextFile as Base
import nltk, string, pickle, os, sklearn


class Query(Base):

    def __init__(self,name,folder):
        """
            Loading the query text file. Loads query contents into the 'text' member variable.
        """
        super(Query,self).__init__(name,folder)


    def clean(self):
        """
        Cleaning the query file. Returns a string.
        """
        clean_query = super().text_preprocess()
        clean_query = ' '.join(clean_query) 
        
        return clean_query


    def convert_to_list(self):
        """
        Converting query to a list of strings because CountVectorizer requires an iterable as input. Returns a list.
        """
        query_list = []
        query_list.append(self.clean())

        return query_list


    def transform_query(self,tf_vec):
        """
        Transforming query into vector. Returns a numpy vector.
        """
        query_list = self.convert_to_list()
        test_vector = tf_vec.transform(query_list).toarray()
        
        return test_vector



#############################################################################################################################################################
    


class MESH(Base):

    def __init__(self,name,folder,encode='latin-1'):
        """
        Load MESH terms from text file. 
        """
        super(MESH,self).__init__(name,folder,encode=encode)
        self.vocab = []
        self.vocab_size = 0
        self.mapping = {}
        self.mapping_size = 0


    def create_MESH_vocab_and_IDmapping(self):
        """
        Create MESH vocabulary from text file. Stores vocab into member variable 'vocab'.
        """
        temp = 0; added_mesh = 0; start = 0
        tree_nodes = ('A','B','C')
        
        for line in self.text.splitlines(): ## take the terms in the line containing 'MH' and put them in a list
            line = nltk.word_tokenize(line)
            if 'RECTYPE' in line:
                start = 1
            if start == 1:
                if 'MH' in line:
                    temp = ' '.join(line[2:])
                    temp = temp.lower()
                if 'MN' in line:
                    if added_mesh == 0 and line[2][0] in tree_nodes:
                        self.vocab.append(temp)
                        added_mesh = 1
            if 'UI' in line:
                start = 0
                mesh_id = line[2]
                if added_mesh == 1:
                    self.mapping[temp] = mesh_id
                added_mesh = 0
                temp = 0                
            
        #vocab = self.lemmatize(vocab)
        #self.vocab = list(set(self.vocab))    ## make sure the vocab has no duplicates
        self.vocab_size = len(self.vocab)
        self.mapping_size = len(self.mapping)

        return


    def save_MESH_vocab(self,name):
        """
            Saving the MESH vocab into a text file. Returns void.
        """
        with open(name, 'wb') as f:
            pickle.dump(self.vocab, f)

        return


    def read_MESH_vocab(self,name):
        """
            Reading the MESH vocab from the text file. Returns a list.
        """
        with open(name, 'rb') as f:
            MESH_vocab = pickle.load(f)

        self.vocab_size = len(MESH_vocab)

        return MESH_vocab


    def save_MESH_IDmapping(self,name):
        """
            Saving the MESH vocab into a text file. Returns void.
        """
        with open(name, 'wb') as f:
            pickle.dump(self.mapping, f)

        return


    def read_MESH_IDmapping(self,name):
        """
            Reading the MESH vocab from the text file. Returns a list.
        """
        with open(name, 'rb') as f:
            MESH_mapping = pickle.load(f)

        self.mapping_size = len(MESH_mapping)

        return MESH_mapping



####################################################################################################################################################################



class Corpus(Base):

    def __init__(self,folder,n=1):
        """
            Creating the corpus of n documents. Stored in member variable 'corpus'.
        """
        self.corpus = []
        self.filenames = []

        for file in os.listdir(folder):    #folder = "corpus/"
            if file.endswith(".txt"):
                super(Corpus,self).__init__(file,folder)
                self.corpus.append(self.text)
                self.filenames.append(file)

        self.corpus = self.corpus[0:n]


    def clean(self):
        """
        Cleaning the corpus using the preprocessing function. Returns a list.
        """
        clean_corpus = []

        for doc in self.corpus:
            self.text = doc
            doc = self.text_preprocess()
            doc = ' '.join(doc)
            clean_corpus.append(doc)
        
        return clean_corpus


    def vectorize_corpus(self,clean_corpus,vocab):
        """
        Creating document-term matrix using CountVectorizer and the corpora. Returns tf_vec sklearn object and numpy matrix tf_matrix. 
        """
        tf_vec = sklearn.feature_extraction.text.CountVectorizer(max_df=1,min_df=2,stop_words='english',strip_accents='unicode', analyzer='word',ngram_range=(1,4),vocabulary=vocab,lowercase=True)
        tf_matrix = tf_vec.fit_transform(clean_corpus).toarray()

        return tf_vec, tf_matrix




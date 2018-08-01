import sys
sys.dont_write_bytecode = True

import nltk, string, sklearn

class TextFile:
    
    def __init__(self,name,folder,encode='utf8'):
        """
            Opens the file and reads the text in the file. Text is stored into TextFile's member variable 'text'.
        """
        with open(folder+name,'r',encoding=encode) as f:
            self.text = f.read()


    def remove_punc(self,punc_list):
        """
            Removes the punctuations specified by punc_list from the text. Returns a string.
        """                
        tokens = nltk.word_tokenize(self.text)
        take_tokens = True
        newtext = []
        
        for char in charlist:   ## remove punctuations from each token
            newlist = []
            if take_tokens:
                newtext = tokens
                take_tokens = False
            for token in newtext:
                newlist.append(token.replace(char,' '))
            newtext = newlist
            
        newtext = ' '.join(newtext) ## join the tokens back into a string
            
        return newtext


    def remove_figs(self, newtext):
        """
            Remove Fig ... / Figure .../ Figures ... from the text. Returns a list.
        """
        tokens = nltk.word_tokenize(newtext)
        
        for token in tokens:    ## remove words like 'fig', 'figure', 'figures' and any variations of those from the text 
            if token == 'Fig' or token == 'Figure' or token == 'Figures':
                if tokens[tokens.index(token)+1].isdigit():
                    tokens.remove(tokens[tokens.index(token)+1])
                tokens.remove(token)
            
        return tokens


    def to_lowercase(self, newlist):
        """
            Convert all the words to lowercase in the text. Passing in a list of the tokens. Returns a list.
        """        
        for n,token in enumerate(newlist):   ## make all tokens lowercase
            token = token.lower()
            newlist[n] = token
            
        return newlist


    def remove_n_char(self, newlist, n):
        """
            Remove three character words from the text. Passing in a list of the tokens. Returns a list.
        """
        tokens = []
        
        for token in newlist:   ## remove tokens that have a length of less than n
            if len(token) > n:
                tokens.append(token)

        return tokens


    def num2str(self, newlist):
        """
            Convert numbers to strings in the text. Passing in a list of the tokens. Returns a list
        """
        from num2words import num2words

        tokens = []
        
        for token in newlist:   ## convert tokens that have numbers to their string equivalent
            if token.isdigit():
                token = int(token)
                token = num2words(token)
            tokens.append(token)
            
        return tokens


    def lemmatize(self, newlist):
        """
            Lemmatize the text. Passing in a list of the tokens. Returns a list
        """
        lemmer = nltk.stem.wordnet.WordNetLemmatizer()
        
        for n,token in enumerate(newlist):   ## lemmatize each token
            token = lemmer.lemmatize(token)
            newlist[n] = token
            
        return newlist


    def stemmer(self, newlist):
        """
            Stem the text. Passing in a list of the tokens. Returns a list
        """
        stems = nltk.stem.SnowballStemmer("english")
        
        for n,token in enumerate(newlist):   ## stem each token
            token = stems.stem(token)
            newlist[n] = token
            
        return newlist


    def text_preprocess(self, char_length=3,punc_user=['-','.','(',')',',',';','/',"'",':','_']):
        """
            Combine the text cleaning functions into one preprocessing step. Passing in the document as a string. Returns a list
        """
        data = self.text
        data = self.remove_punc(punc_user)
        data = self.remove_figs(data)
        data = self.lemmatize(data)
        data = self.to_lowercase(data)
        data = self.remove_n_char(data,char_length)
        data = self.num2str(data)
        
        #data = self.stemmer(data)
        
        return data


    def get_MESH_terms(self):
        """
            Get the MESH terms in each document. Returns a dict.
        """
        doc_terms = {}
        
        for line in self.text.splitlines(): ## take the terms in front of the 'MH' field and put that in a dict with the 'PMID' as the key
            terms = ''
            line = line.replace('*','')
            line = nltk.word_tokenize(line)
            if 'PMID-' in line:                   
                pmid = line[1]
                doc_terms[pmid] = set()
            elif 'MH' in line:
                line = ' '.join(line[2:])
                line = line.lower()
                if '/' in line: ## ignoring everything after the '/'
                    pos = line.index('/')
                    line = line[:pos]
                terms = line
                doc_terms[pmid].add(terms)
        
        return doc_terms


    def get_UI(self,mesh_id_mapping):
        """
            Map each MESH term in a document to a MESH ID. Returns a dict.
        """
        doc_terms = self.get_MESH_terms()
        doc_UI = {}

        for key in doc_terms.keys():    ## mapping each MESH term present in the case reports to a UI
            doc_UI[key] = set()
            value = doc_terms[key]
            for i in value:
                if i in mesh_id_mapping:
                    doc_UI[key].add(mesh_id_mapping[i])

        return doc_UI


    def get_keywords(self):
        """
            Extracts 'OT' field from Medical Case Reports. Returns a dictionary.
        """
        doc_keywords = {}
        
        for line in self.text.splitlines(): ## take the terms in front of the 'OT' field and put that in a dict with 'PMID' as the key
            ts = ''
            line = line.replace('*','')
            line = line.replace('.','')
            line = nltk.word_tokenize(line)
            if 'PMID-' in line:
                pmid = line[1]
                doc_keywords[pmid] = set()
            elif 'OT' in line:
                line = ' '.join(line[2:])
                line = line.lower()
                if ('(' in line and '/' in line): ## ignoring everything after '(' or '/', whichever one comes first if both are in the same line
                    pos1 = line.index('(')
                    pos2 = line.index('/')
                    if pos1 > pos2:
                        ts = line[:pos2-1]
                    else:
                        ts = line[:pos1-1]
                elif '(' in line:   ## ignoring everything after '('
                    pos = line.index('(')
                    ts = line[:pos-1]
                elif '/' in line:   ## ignoring everything after '/'
                    pos = line.index('/')
                    ts = line[:pos-1]
                else:   ## this means line contains neither '(' nor '/' so take the entire line
                    ts = line

                if "'" in ts:
                    pos = ts.index("'")
                    ts = ts[:pos] + ts[(pos+2):]
                    ts = ' '.join(ts.split())
                doc_keywords[pmid].add(ts)
    
        return doc_keywords


    def get_titles(self):
        """
            Get the 'TI' field from Medical Case Reports. Returns a dictionary.
        """
        headings = ("AB","AD","CA","CY","DA","DCOM","DEP","DP","EA","EDAT","ID","IP","IS","JC","JID","JT","LID","LR","MHDA","NI",
                       "OAB","OWN","PG","PL","PMC","PMID","PST","PUBM","RF","SB","SO","STAT","TA","TI","TT","VI","YR",
                       "AB-","AD-","CA-","CY-","DA-","DCOM-","DEP-","DP-","EA-","EDAT-","ID-","IP-","IS-","JC-","JID-","JT-","LID-","LR-","MHDA-","NI-",
                       "OAB-","OWN-","PG-","PL-","PMC-","PMID-","PST-","PUBM-","RF-","SB-","SO-","STAT-","TA-","TI-","TT-","VI-","YR-")
        
        doc_titles = {}
        index = 0; start = 0; 
        
        text = self.text.splitlines()
        while index != len(text):   ## take the terms in front of the 'TI' or 'TI-' field and put that in a dict with 'PMID' as key 
            line = nltk.word_tokenize(text[index])
            if 'PMID-' in line:
                pmid = line[1]
                temp = ""
                doc_titles[pmid] = set()
            elif 'TI' in line:
                temp = line[2:]
                start = 1
            elif 'TI-' in line:
                temp = line[1:]
                start = 1
            else:
                if start == 1:
                    if line[0] not in headings: ## check if the next line after 'TI' does not have any headings. This means the titles is longer than one line.
                        temp += line
                    else:
                        start = 0
                        temp = [ch for ch in temp if len(ch) > 2]
                        temp = [ch for ch in temp if ch not in string.punctuation]
                        temp = " ".join(temp).lower()
                        doc_titles[pmid] = temp
                        temp = ""
            index += 1

        for key in doc_titles.copy():   ## removing words in title that are either punctuation or less than 3 characters long
            title = []; title.append(doc_titles[key])
            cv = sklearn.feature_extraction.text.CountVectorizer(stop_words='english',strip_accents='unicode', analyzer='word',ngram_range=(1,6))   ## get the ngrams for the titles
            cv.fit(title)    ## getting ngrams is important because various subphrases of the title can match with description of ICD10 codes
            doc_titles[key] = cv.get_feature_names()    ## put the ngrams into a dict with 'PMID' as key
            
        return doc_titles



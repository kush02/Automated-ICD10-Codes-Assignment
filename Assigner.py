import sys
sys.dont_write_bytecode = True

import nltk, collections, string
import matplotlib.pyplot as plt
import pandas as pd
import pymysql as sql


class Assigner:

    def __init__(self,mesh_to_icd10_mapping,mesh_id_mapping):
        """
            Takes in the mesh_to_icd10_mapping. Creates inverse mapping from MeSH terms to their respective UIs
        """
        self.mesh_to_icd10_mapping = pd.read_csv(mesh_to_icd10_mapping)
        self.mesh_id_mapping = mesh_id_mapping
        self.inverse_mapping = dict((v, k) for k, v in self.mesh_id_mapping.items())
        self.no_MESHterms_codes_per_report = {}
        self.no_MESHterms_codes = []


    def assign_MESHterms_ICD10(self,ui):
        """
            Assign ICD10 codes to each document based on the MESH IDs present in the document. Returns a dict.
        """
        df = self.mesh_to_icd10_mapping
        mesh_terms_ICD10 = {}
        
        for key in ui.keys():   ## taking the UI of each MESH term present in the case reports and finding the corresponding ICD10 codes in the mapping file
            self.no_MESHterms_codes_per_report[key] = set()
            for i in ui[key]:
                if i in df['MESH_ID'].values:   ## check if UI is in the mapping file
                    mesh_terms_ICD10[key] = set(df[df['MESH_ID']==i]['ICD10CM_CODE'].values.tolist())
                else:   ## if UI is not mapping file, add it to a list containing all the missing UIs. Also add UI to a dict that contains all the missing UIs for each case report
                    self.no_MESHterms_codes_per_report[key].add(i)
                    self.no_MESHterms_codes.append(i)
    
        return mesh_terms_ICD10


    def assign_keywords_ICD10(self,keywords,marker="*"):
        """
            Assigns ICD10 codes to keywords of each document. Returns a dictionary. Codes have a '*' attached to them.
        """
        conn = sql.connect(host="localhost",user="root",passwd="pass",db="umls")    ## connecting to the MySQL server to run queries
        cur = conn.cursor()

        keyword_ICD10 = {}
        
        for key in keywords.keys(): ## find ICD10 codes for keywords
            keyword_ICD10[key] = set()
            for i in keywords[key]:
                cur.execute("select * from mrconso where sab='ICD10CM' and str='%s';" % i)
                for row in cur.fetchall():  ## go through the SQL table
                    keyword_ICD10[key].add(marker+row[13])  ## ICD10 code is in row[13]
        
        conn.close()

        return keyword_ICD10


    def assign_titles_ICD10(self,titles,marker="^"):
        """
            Assigns ICD10 codes to titles of each document. Returns a dictionary. Codes have a '^' attached to them.
        """
        titles_ICD10 = self.assign_keywords_ICD10(titles,marker=marker) ## the marker for titles is different than that of keywords

        return titles_ICD10


    def create_stopword_list(self):
        """
            Calculating the term frequency of a word in each term. Returns a dict.
        """
        UIs = set(self.no_MESHterms_codes)
        lemmer = nltk.stem.wordnet.WordNetLemmatizer()
        words = []
        
        for UI in UIs:
            term = self.inverse_mapping[UI] ## get term corresponding to the UI
            for t in nltk.word_tokenize(term):
                if (t not in string.punctuation and t.isdigit() != True):   ## clean term of punctations and non-alpha characters
                    t = lemmer.lemmatize(t)
                    words.append(t)

        stopword_list = {}
        c = collections.Counter(words)  ## counts of each term
        
        stopword_list = {key: c[key]/float(len(words)) for key in c.keys()} ## populate stopword_list with term frequency
        
        return stopword_list
        

    def assign_MESHterms_partial_match_single_codes(self,n_least_common=2,num_rows=100,marker="~"):
        """
            Assigns ICD10 codes based on partial matches of MeSH terms that could not be found in the mapping file. Returns a dict.
        """
        conn = sql.connect(host="localhost",user="root",passwd="pass",db="umls")    ## connecting to the MySQL server to run queries
        cur = conn.cursor()

        stopword_list = self.create_stopword_list()
        stopword_list_threshold = collections.Counter(list(stopword_list.values())).most_common()[n_least_common-1][0] ## using the term frequency of nth least common words as threshold
                
        UIs = set(self.no_MESHterms_codes)
        lemmer = nltk.stem.wordnet.WordNetLemmatizer()
        single_codes = {}
        
        for UI in UIs:
            codes = []
            term = self.inverse_mapping[UI] ## get term for the corresponding UI
            
            if ',' in term: ## formatting the term to match the formatting of other terms that do not contain punctuation
                term = term.split(',')
                term = term[::-1]
                term = ' '.join(term)
            term = term.split()

            term_len = len(term)
            for i in range(term_len):
                term[i] = lemmer.lemmatize(term[i])
                try:
                    if stopword_list[term[i]] <= stopword_list_threshold:   ## keep the words that have term frequency less than or equal to the threshold
                        term.append(term[i])
                except KeyError:    ## if word not in keys, skip it. Used to detect all the non-alpha deleted from stopword_list
                    continue                        
            term = term[term_len:]

            term_len = len(term)
            if term_len > 0:
                if term_len == 2:   ## checking for two partial string matches
                    cur.execute("select * from mrconso where sab='ICD10CM' and str like '%%%s%%' and str like '%%%s%%';" %(term[0], term[1]))
                    results = cur.fetchall()
                    term = ' '.join(term)
                elif term_len == 3: ## checking for three partial string matches 
                    cur.execute("select * from mrconso where sab='ICD10CM' and str like '%%%s%%' and str like '%%%s%%' and str like '%%%s%%';" %(term[0], term[1], term[2]))
                    results = cur.fetchall()
                    term = ' '.join(term)
                else:   ## checking if single or more than three string matches gets singular codes
                    term = ' '.join(term)
                    cur.execute("select * from mrconso where sab='ICD10CM' and str like '%%%s%%';" % term)
                    results = cur.fetchall()
                
            if len(results) < num_rows and len(results) > 0:    ## only go through results if they are under 100 rows in length
                for row in results:
                    if term_len > 1:
                        codes.append(marker + row[13])
                    else:
                        tokens = nltk.word_tokenize(row[14].lower())    ## tokenize the string description for the code
                        if term in tokens:  ## check if the term appears independently (not as a substring) in the string
                            codes.append(marker + row[13])
                         
            for key in self.no_MESHterms_codes_per_report.keys():
                single_codes[key] = set()
                if UI in self.no_MESHterms_codes_per_report[key]:   ##  check if UI appears in a case report
                    if len(set(codes)) == 1:    ## if all the codes are the same, then only one code exists for the UI. Achieved precision. Same code.
                        single_codes[key] = set(codes)
                    else:
                        codes = [code.split('.')[0] for code in codes]
                        if len(set(codes)) == 1:    ## this means the all the codes share the same tree node for the UI. Achieved generality. Same tree node.
                            single_codes[key] = set(codes)
                            
        conn.close()
        
        return single_codes


    def assign_all_ICD10(self,ui,keywords,titles):
        """
            Assigns all possible codes for a case report. Returns a dict.
        """
        mesh_codes = self.assign_MESHterms_ICD10(ui)
        keywords_codes = self.assign_keywords_ICD10(keywords)
        titles_codes = self.assign_titles_ICD10(titles)
        mesh_partial_match_codes = self.assign_MESHterms_partial_match_single_codes()

        a = self.join_codes(mesh_codes,keywords_codes)
        b = self.join_codes(a,titles_codes)
        c = self.join_codes(b,mesh_partial_match_codes)

        return c
        

    def join_codes(self,x,y):
        """
            Join ICD10 codes from two dictionaries into one dictionary. Returns a dictionary.
        """
        joint_dict = {}
        
        for key in x.keys():
            joint_dict[key] = x[key].union(y[key])
            
        return joint_dict


    def write_codes_to_csv(self,codes,name):
        """
            Puts a dictionary of codes into a csv file.
        """
        df = pd.DataFrame.from_dict(codes,orient='index').transpose()
        df.to_csv(name,index=False)

        return 



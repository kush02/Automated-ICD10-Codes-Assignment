import sys
sys.dont_write_bytecode = True

import nltk, itertools, collections, string, sklearn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pymysql as sql


class Assigner:

    def __init__(self,mesh_to_icd10_mapping,mesh_id_mapping):
        """
            Takes in the mesh_to_icd10_mapping.
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
                if i in df['MESH_ID'].values:
                    mesh_terms_ICD10[key] = set(df[df['MESH_ID']==i]['ICD10CM_CODE'].values.tolist())
                else:
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
                for row in cur.fetchall():
                    keyword_ICD10[key].add(marker+row[13])
        
        conn.close()

        return keyword_ICD10


    def assign_titles_ICD10(self,titles,marker="^"):
        """
            Assigns ICD10 codes to titles of each document. Returns a dictionary. Codes have a '^' attached to them.
        """
        titles_ICD10 = self.assign_keywords_ICD10(titles,marker=marker)

        return titles_ICD10


    def create_stopword_list(self):
        """
            Calculating the term frequency of a word in each term. Returns a dict.
        """
        UIs = set(self.no_MESHterms_codes)
        lemmer = nltk.stem.wordnet.WordNetLemmatizer()
        words = []
        
        for UI in UIs:
            term = self.inverse_mapping[UI]
            for t in nltk.word_tokenize(term):
                if (t not in string.punctuation and t.isdigit() != True):
                    t = lemmer.lemmatize(t)
                    words.append(t)

        stopword_list = {}
        c = collections.Counter(words)
        
        stopword_list = {key: c[key]/float(len(words)) for key in c.keys()}
        
        return stopword_list
        

    def assign_MESHterms_partial_match_single_codes(self,num_rows=100,marker="~"):
        """

        """
        conn = sql.connect(host="localhost",user="root",passwd="pass",db="umls")    ## connecting to the MySQL server to run queries
        cur = conn.cursor()

        stopword_list = self.create_stopword_list()
        stopword_list_threshold = collections.Counter(list(stopword_list.values())).most_common()[1][0] ## using the tf of second least common words as threshold
                
        UIs = set(self.no_MESHterms_codes)
        lemmer = nltk.stem.wordnet.WordNetLemmatizer()
        single_codes = {}
        
        for UI in UIs:
            codes = []
            term = self.inverse_mapping[UI]
            
            if ',' in term:
                term = term.split(',')
                term = term[::-1]
                term = ' '.join(term)
            term = term.split()

            term_len = len(term)
            for i in range(term_len):
                term[i] = lemmer.lemmatize(term[i])
                try:
                    if stopword_list[term[i]] <= stopword_list_threshold:
                        term.append(term[i])
                except KeyError:
                    continue                        
            term = term[term_len:]

            term_len = len(term)
            if term_len > 0:
                if term_len == 2:   ## checking if two partial string matches gets singular codes
                    cur.execute("select * from mrconso where sab='ICD10CM' and str like '%%%s%%' and str like '%%%s%%';" %(term[0], term[1]))
                    results = cur.fetchall()
                    term = ' '.join(term)
                elif term_len == 3: ## checking if three partial string matches gets singular codes
                    cur.execute("select * from mrconso where sab='ICD10CM' and str like '%%%s%%' and str like '%%%s%%' and str like '%%%s%%';" %(term[0], term[1], term[2]))
                    results = cur.fetchall()
                    term = ' '.join(term)
                else:   ## checking if single or more than three string matches gets singular codes
                    term = ' '.join(term)
                    cur.execute("select * from mrconso where sab='ICD10CM' and str like '%%%s%%';" % term)
                    results = cur.fetchall()
                
            if len(results) < num_rows and len(results) > 0:
                for row in results:
                    if term_len > 1:
                        codes.append(marker + row[13])
                    else:
                        tokens = nltk.word_tokenize(row[14].lower())
                        if term in tokens:
                            codes.append(marker + row[13])
                         
            for key in self.no_MESHterms_codes_per_report.keys():
                single_codes[key] = set()
                if UI in self.no_MESHterms_codes_per_report[key]:
                    if len(set(codes)) == 1:
                        single_codes[key] = set(codes)
                        #print("same code",term, codes)
                    else:
                        codes = [code.split('.')[0] for code in codes]
                        if len(set(codes)) == 1:
                            single_codes[key] = set(codes)
                            #print("same tree node",term, codes)
                            
        conn.close()
        
        return single_codes


    def assign_all_ICD10(self,ui,keywords,titles):
        """

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



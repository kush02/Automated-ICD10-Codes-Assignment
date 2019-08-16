import nltk, collections, string, sklearn, itertools
import matplotlib.pyplot as plt
import pandas as pd
import pymysql as sql


class Assigner:

    def __init__(self,mesh_to_icd10_mapping,mesh_id_mapping):
        """
            Takes in the mesh_to_icd10_mapping. Creates inverse mapping from UIs to their respective MeSH terms
        """
        self.mesh_to_icd10_mapping = pd.read_csv(mesh_to_icd10_mapping)
        self.mesh_id_mapping = mesh_id_mapping
        self.inverse_mapping = dict((v, k) for k, v in self.mesh_id_mapping.items())
        self.no_MESHterms_codes_per_report = {}
        self.unassigned_MESHterms_per_report = {}
        self.no_MESHterms_codes = []


    def assign_MESHterms_ICD10(self,ui):
        """
            Assign ICD10 codes to each document based on the MESH IDs present in the document. Returns a dict.
        """
        df = self.mesh_to_icd10_mapping
        count = 0
        mesh_terms_ICD10 = {}
        
        for key in ui.keys():   ## taking the UI of each MESH term present in the case reports and finding the corresponding ICD10 codes in the mapping file
            self.no_MESHterms_codes_per_report[key] = set()
            self.unassigned_MESHterms_per_report[key] = set()
            mesh_terms_ICD10[key] = set()
            for i in ui[key]:
                if i in df['MESH_ID'].values:   ## check if UI is in the mapping file
                    for j in df[df['MESH_ID']==i]['ICD10CM_CODE'].values.tolist():
                        mesh_terms_ICD10[key].add((j,float(1)))
                else:   ## if UI is not mapping file, add it to a list containing all the missing UIs. Also add UI to a dict that contains all the missing UIs for each case report
                    self.no_MESHterms_codes_per_report[key].add(i)
                    self.unassigned_MESHterms_per_report[key].add(self.inverse_mapping[i])
                    self.no_MESHterms_codes.append(i)

        #print(self.unassigned_MESHterms_per_report) 
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
                cur.execute("select distinct(code) from mrconso where sab='ICD10CM' and tty!='AB' and str='%s';" % i)
                for row in cur.fetchall():  ## go through the SQL table
                    keyword_ICD10[key].add((row[0],float(1)))  ## ICD10 code is in row[13]
        
        conn.close()
        
        return keyword_ICD10


    def assign_titles_ICD10(self,titles,marker="^"):
        """
            Assigns ICD10 codes to titles of each document. Returns a dictionary. Codes have a '^' attached to them.
        """
        titles_ICD10 = self.assign_keywords_ICD10(titles,marker=marker) ## the marker for titles is different than that of keywords

        return titles_ICD10


    def create_stopword_list(self,stopword_percent_include):
        """
            Calculating the term frequency of a word in each term. Returns a dict.
        """
        UIs = set(self.no_MESHterms_codes)
        lemmer = nltk.stem.wordnet.WordNetLemmatizer()
        words = []
        
        for UI in UIs:
            term = self.inverse_mapping[UI] ## get term corresponding to the UI
            for t in nltk.word_tokenize(term):
                if t not in string.punctuation:   ## clean punctations, non-alpha characters
                    t = lemmer.lemmatize(t)
                    words.append(t)

        stopword_list = {}
        c = collections.Counter(words)  ## counts of each term
        
        stopword_list = {key: c[key]/float(len(words)) for key in c.keys()} ## populate stopword_list with term frequency
        c1 = collections.Counter(list(stopword_list.values())).most_common()
        count = 0; percent = 0; index = 0
        for n,item in enumerate(c1):
            count += item[1]
            percent = count/float(len(stopword_list))
            if percent <= stopword_percent_include:
                index = n
            else:
                break
        stopword_threshold = collections.Counter(list(stopword_list.values())).most_common()[index][0] ## using the term frequency as threshold
        
        return stopword_list,stopword_threshold


    def assign_context_aware_codes(self,stopword_percent_include=0.8):
        """

        """
        conn = sql.connect(host="localhost",user="root",passwd="pass",db="umls")    ## connecting to the MySQL server to run queries
        cur = conn.cursor()

        stopword_list,stopword_threshold = self.create_stopword_list(stopword_percent_include)
        eng_stopwords = set(nltk.corpus.stopwords.words('english'))
        
        lemmer = nltk.stem.wordnet.WordNetLemmatizer()
        context_aware_codes = {}
        for key in self.unassigned_MESHterms_per_report.keys():
            context_aware_codes[key] = set(); codes_dict = {};
            all_codes = set()
            terms = set([lemmer.lemmatize(j) for i in self.unassigned_MESHterms_per_report[key] for j in i.split() if j not in string.punctuation])
            #print(key,self.unassigned_MESHterms_per_report[key])
            for val in self.unassigned_MESHterms_per_report[key]:
                codes = []; more_context = []; less_context = []
                val = nltk.word_tokenize(val)
                val = [lemmer.lemmatize(i) for i in val if i not in string.punctuation]
                val_save = val; val_len = len(val)
                val = [i for i in val if stopword_list[i] <= stopword_threshold]    ## remove stopwords
                ##  search in UMLS database
                if len(val) == 1:
                    cur.execute("select distinct(code),str from mrconso where sab='ICD10CM' and tty!='AB' and str like '%%%s%%';" % val[0])
                elif len(val) == 2:
                    cur.execute("select distinct(code),str from mrconso where sab='ICD10CM' and tty!='AB' and str like '%%%s%%' and str like '%%%s%%';" % (val[0],val[1]))
                elif len(val) == 3:
                    cur.execute("select distinct(code),str from mrconso where sab='ICD10CM' and tty!='AB' and str like '%%%s%%' and str like '%%%s%%' and str like '%%%s%%';"
                                % (val[0],val[1],val[2]))
                elif len(val) == 4:
                    cur.execute("select distinct(code),str from mrconso where sab='ICD10CM' and tty!='AB' and str like '%%%s%%' and str like '%%%s%%' and str like '%%%s%%' and str like '%%%s%%';"
                                % (val[0],val[1],val[2],val[3]))
                elif len(val) == 5:
                    cur.execute("select distinct(code),str from mrconso where sab='ICD10CM' and tty!='AB' and str like '%%%s%%' and str like '%%%s%%' and str like '%%%s%%' and str like '%%%s%%' and str like '%%%s%%';"
                                % (val[0],val[1],val[2],val[3],val[4]))
                elif val_len == 2 and len(val) == 0:    ## if term only contains stopwords, then run the entire term through a partial match so as to not skip whole terms
                    cur.execute("select distinct(code),str from mrconso where sab='ICD10CM' and tty!='AB' and str like '%%%s%%' and str like '%%%s%%';" % (val_save[0],val_save[1]))
                elif val_len == 3 and len(val) == 0:
                    cur.execute("select distinct(code),str from mrconso where sab='ICD10CM' and tty!='AB' and str like '%%%s%%' and str like '%%%s%%' and str like '%%%s%%';"
                                % (val_save[0],val_save[1],val_save[2]))
                elif val_len == 4 and len(val) == 0:
                    cur.execute("select distinct(code),str from mrconso where sab='ICD10CM' and tty!='AB' and str like '%%%s%%' and str like '%%%s%%' and str like '%%%s%%' and str like '%%%s%%';"
                                % (val_save[0],val_save[1],val_save[2],val_save[3]))
                elif val_len == 5 and len(val) == 0:
                    cur.execute("select distinct(code),str from mrconso where sab='ICD10CM' and tty!='AB' and str like '%%%s%%' and str like '%%%s%%' and str like '%%%s%%' and str like '%%%s%%' and str like '%%%s%%';"
                                % (val_save[0],val_save[1],val_save[2],val_save[3],val_save[4]))
                else:
                    continue
                    
                for row in cur.fetchall():
                    descrp = set([lemmer.lemmatize(k.lower()) for k in nltk.word_tokenize(row[1]) if k not in string.punctuation and k not in eng_stopwords])   ## clean the description
                    intersect = set.intersection(terms,descrp)  ## calculate intersection of description and terms
                    union = set.union(set(val),descrp)  ## calculate union of val and description
                    jaccard_score = float(len(intersect))/float(len(union)) ## calculate modified Jaccard coefficient
                    if jaccard_score != 0.0:
                        if len(val) != 0 and len(val) < val_len and len(intersect) > len(val):  ## this means some words in 'terms' are part of intersect. So there is more context
                            more_context.append((row[0],jaccard_score))
                        else:
                            less_context.append((row[0],jaccard_score))
                
                if more_context:    ##  if more_context is not empty, then choose it
                    codes = more_context
                else:
                    codes = less_context
                
                for code in codes:
                    node = code[0].split('.')[0]
                    try:
                        if codes_dict[node][1] < code[1]:   ##  replace codes that have lower jaccard score 
                            codes_dict[node] = code
                        elif codes_dict[node][1] == code[1]:
                            if len(codes_dict[node][0]) > len(code[0]):   ## For same node, pick the more general code(which is always shorter in length)
                                codes_dict[node] = code
                    except KeyError:
                        codes_dict[node] = code
            
            if codes_dict:
                for v in codes_dict.values():
                    context_aware_codes[key].add(v)
        
        conn.close()
        
        return context_aware_codes
        

    def assign_all_ICD10(self,ui,keywords,titles,partial_match=True,create_stopword=True,stopword_percent_include=0.8):
        """
            Assigns all possible codes for a case report. Returns a dict.
        """
        mesh_codes = self.assign_MESHterms_ICD10(ui)
        keywords_codes = self.assign_keywords_ICD10(keywords)
        titles_codes = self.assign_titles_ICD10(titles)
        mesh_partial_match_codes = {}
        if partial_match:
            mesh_partial_match_codes = self.assign_context_aware_codes(stopword_percent_include=stopword_percent_include)

        a = self.join_codes(mesh_codes,keywords_codes)
        b = self.join_codes(a,titles_codes)
        c = self.join_codes(b,mesh_partial_match_codes)            

        return c
        

    def join_codes(self,x,y):
        """
            Join ICD10 codes from two dictionaries into one dictionary. Returns a dictionary.
        """
        joint_dict = {}

        if len(x) >= len(y):
            for key in x.keys():
                try:
                    joint_dict[key] = set.union(x[key],y[key])
                except KeyError:
                    joint_dict[key] = x[key]
        else:
            for key in y.keys():
                try:
                    joint_dict[key] = set.union(x[key],y[key])
                except KeyError:
                    joint_dict[key] = y[key]
            
        return joint_dict


    def write_codes_to_csv(self,codes,name):
        """
            Puts a dictionary of codes into a csv file.
        """
        df = pd.DataFrame.from_dict(codes,orient='index')
        df.to_csv(name)

        return 

################################################################################################################################################################################

    def assign_MESHterms_partial_match_single_codes(self,create_stopword=True,stopword_percent_include=0.8,num_rows=100,marker="~"):
        """
            Assigns ICD10 codes based on partial matches of MeSH terms that could not be found in the mapping file. Returns a dict.
        """
        if not create_stopword:
            return
        
        conn = sql.connect(host="localhost",user="root",passwd="pass",db="umls")    ## connecting to the MySQL server to run queries
        cur = conn.cursor()

        stopword_list,stopword_threshold = self.create_stopword_list(stopword_percent_include)

        UIs = set(self.no_MESHterms_codes)
        lemmer = nltk.stem.wordnet.WordNetLemmatizer()

        single_codes = {}
        for k in self.no_MESHterms_codes_per_report.keys():
                single_codes[k] = set()
                
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

            term_len = len(term); results = []
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
                        codes.append(row[13])
                    else:
                        tokens = nltk.word_tokenize(row[14].lower())    ## tokenize the string description for the code
                        if term in tokens:  ## check if the term appears independently (not as a substring) in the string
                            codes.append(row[13])

            for key in self.no_MESHterms_codes_per_report.keys():
                if UI in self.no_MESHterms_codes_per_report[key]:   ##  check if UI appears in a case report
                    if len(set(codes)) == 1:    ## if all the codes are the same, then only one code exists for the UI. Achieved precision. Same code.
                        single_codes[key] = set(codes) 
                    else:
                        codes = [code.split('.')[0] for code in codes]
                        if len(set(codes)) == 1:    ## this means the all the codes share the same tree node for the UI. Achieved generality. Same tree node.
                            single_codes[key] = set(codes)

        conn.close()
        
        return single_codes


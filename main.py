import nltk, string, sklearn, os, time, scipy, collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from TextFile import *
from DerivedClass import *
from Assigner import *
import LinkPrediction as lp



def main():    
    ##### Testing MESH class
    mv = MESH('MESH Terms.txt','')
    #mv.create_MESH_vocab_and_IDmapping(); mv.save_MESH_IDmapping('MESH ID Mapping.txt')
    mesh_id_mapping = mv.read_MESH_IDmapping('MESH ID Mapping.txt')
    
    ##### Testing Corpus class
    #folder = "corpus/"
    #num = 100
    #corp = Corpus(folder,n=num)
    
    ##### Vectorizing
    #tf_vectorizer, tf_matrix = corp.vectorize_corpus(corp.clean(),voc)
    #tv = q.transform_query(tf_vectorizer).flatten()

    ##### Getting data
    tf = TextFile('Medical Case Reports.txt','',encode='latin-1')
    #tf = TextFile('pubmed_result.txt','',encode='latin-1')
    mesh_terms = tf.get_MESH_terms()
    ui = tf.get_UI(mesh_id_mapping)
    keywords = tf.get_keywords()    
    titles = tf.get_titles()

    ##### Assigning ICD10 codes
    asg = Assigner('MESH_ICD10_Mapping.csv',mesh_id_mapping)
    mesh_codes = asg.assign_MESHterms_ICD10(ui)
    #print("Printing direct mapped codes", mesh_codes)
    #keywords_codes = asg.assign_keywords_ICD10(keywords)
    #print("Printing codes using keywords", keywords_codes)
    #titles_codes = asg.assign_titles_ICD10(titles)
    #print("Printing codes using titles", titles_codes)
    #partial_codes = asg.assign_context_aware_codes(stopword_percent_include=0.92)
    #print("Printing context-aware codes", partial_codes)
    #total_codes = asg.assign_all_ICD10(ui,keywords,titles,stopword_percent_include=0.92)
    #print("Printing all codes", total_codes)
    #asg.write_codes_to_csv(tot,'all_codes.csv')  # Case Reports_ICD10_Mapping.csv 


    ##### Link Prediction
    G = nx.Graph()
    d = mesh_codes
    G = lp.create_weighted_bipartite_graph(G,d)
    
    ratings = []
    for i in nx.connected_components(G):
        G_new = nx.Graph()
        print(i)
        for j in i:
            if j.isdigit():
                for v in d[j]:
                    G_new.add_node(v[0],bipartite='code')
                    G_new.add_edge(j,v[0],weight=v[1])

        df = lp.item_based_CF(G_new)
        ratings.append(df)
    
    return 



main()

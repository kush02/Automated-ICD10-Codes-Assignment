import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
import pandas as pd



def create_weighted_bipartite_graph(G,d):
	"""
		Creates a weighted bipartite graph. G is an instance of empty graph. It is filled with data in dict d
	"""

	for k in d.keys():
		for v in d[k]:
			G.add_node(v[0],bipartite='code')
			G.add_edge(k,v[0],weight=v[1])

	return G


def item_based_CF(G):
	"""
		Implements the item-based Collaborative Filtering algorithm. Returns a dataframe.
	"""

	code, pmid = nx.bipartite.sets(G)
	X = nx.bipartite.biadjacency_matrix(G,pmid,column_order=code)
	mean_X = np.mean(X,axis=0)
	adjusted_X = X - mean_X
	similarity = cosine_similarity(X.T)
	rating = mean_X + np.dot(adjusted_X,similarity)/np.sum(np.abs(similarity),axis=1)
	df = pd.DataFrame(data=rating,index=pmid,columns=code)

	return df


def NMF_LP(G,n_components=None, init=None, solver='cd', beta_loss='frobenius', tol=0.0001, max_iter=200, random_state=None, alpha=0.0, l1_ratio=0.0, verbose=0, shuffle=False):
	"""
		Implements NMF for link prediction
	"""
	code, pmid = nx.bipartite.sets(G)
	X = nx.bipartite.biadjacency_matrix(G,pmid,column_order=code)
	
	model = NMF(n_components=n_components, init=init, solver=solver, beta_loss=beta_loss, tol=tol, max_iter=max_iter, 
		random_state=random_state, alpha=alpha, l1_ratio=l1_ratio, verbose=verbose, shuffle=shuffle)
	W = model.fit_transform(X)
	H = model.components_
	
	return np.dot(W,H)



def draw_bipartite_graph(G):
	"""
		Draws the bipartite graph with unweighted edges
	"""

	top, bot = nx.bipartite.sets(G)
	pos = nx.bipartite_layout(G, top)
	nx.draw_networkx(G,pos=pos)
	plt.show()

	return


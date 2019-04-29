
# coding: utf-8

# In[6]:


from networkx.readwrite import *
from networkx.readwrite import json_graph
import networkx as nx
import json
import numpy as np
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import os

dataset_dir = '.'
prefix = 'reddit'


# In[2]:


def load_data(prefix, normalize=True, load_walks=False):
    G_data = json.load(open(prefix + "-G.json"))
    G = json_graph.node_link_graph(G_data)
    if isinstance(G.nodes()[0], int):
        conversion = lambda n : int(n)
    else:
        conversion = lambda n : n
        
    import os
    if os.path.exists(prefix + "-feats.npy"):
        feats = np.load(prefix + "-feats.npy")
    else:
        print("No features present.. Only identity features will be used.")
        feats = None
    id_map = json.load(open(prefix + "-id_map.json"))
    id_map = {conversion(k):int(v) for k,v in id_map.items()}
    walks = []
    class_map = json.load(open(prefix + "-class_map.json"))
    if isinstance(list(class_map.values())[0], list):
        lab_conversion = lambda n : n
    else:
        lab_conversion = lambda n : int(n)

    class_map = {conversion(k):lab_conversion(v) for k,v in class_map.items()}

    ## Remove all nodes that do not have val/test annotations
    ## (necessary because of networkx weirdness with the Reddit data)
    broken_count = 0
    for node in G.copy().nodes():
        if not 'val' in G.node[node] or not 'test' in G.node[node]:
            G.remove_node(node)
            broken_count += 1
    print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))

    ## Make sure the graph has edge train_removed annotations
    ## (some datasets might already have this..)
    print("Loaded data.. now preprocessing..")
    for edge in G.edges():
        if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
            G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False

    if normalize and not feats is None:
        from sklearn.preprocessing import StandardScaler
        train_ids = np.array([id_map[n] for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']])
        train_feats = feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)
    
    if load_walks:
        with open(prefix + "-walks.txt") as fp:
            for line in fp:
                walks.append(map(conversion, line.split()))

    return G, feats, id_map, walks, class_map
data = load_data(prefix)
(G, feats, id_map, walks, class_map) = data


# In[3]:


train_ids = [n for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']]
test_ids = [n for n in G.nodes() if G.node[n]['test']]
val_ids = [n for n in G.nodes() if G.node[n]['val']]
ids = train_ids + test_ids + val_ids

train_labels = [class_map[i] for i in train_ids]
test_labels = [class_map[i] for i in test_ids]
val_labels = [class_map[i] for i in val_ids]
labels = train_labels + test_labels + val_labels

ids, labels = zip(*sorted(zip(ids, labels)))
name_to_id = {}
for i, name in enumerate(ids):
    name_to_id[name] = i


# In[4]:


print(len(train_ids), len(train_labels))
print(len(test_ids), len(test_labels))
print(len(val_ids), len(val_labels))
print(len(ids), len(labels))


# # Generate

# In[5]:


graph_file = open(prefix + '.graph', "w")
adj_matrix = {}
for node in G.node:
    neighbors = G.neighbors(node)
    adj_matrix[name_to_id[node]] = [str(name_to_id[n]) for n in neighbors]

for i in range(len(adj_matrix)):
    print(" ".join(adj_matrix[i]), file = graph_file)
graph_file.close()


# In[7]:


split_file = open(prefix + '.split', "w")
split_dict = {}
for i, node in enumerate(G.node):
    split = 0
    if node in train_ids:
        split = 1
    elif node in val_ids:
        split = 2
    elif node in test_ids:
        split = 3
    split_dict[name_to_id[node]] = split
    if i % 1000 == 0:
        print(i)
    
for i in range(len(split_dict)):
    split = split_dict[i]
    print(split, file = split_file)
    
split_file.close()


# In[ ]:


final_features = []
final_labels = []
for i, id in enumerate(ids):
    final_features.append(feats[id_map[id]])l
    final_labels.append(labels[i])
    from sklearn import datasets
datasets.dump_svmlight_file(final_features, final_labels, prefix + ".svmlight")


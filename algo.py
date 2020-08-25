################################################################################
import sys
import csv
import re
import os
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import math
import scipy.spatial.distance as ssd
from sklearn import cluster, metrics
from time import time
from scipy import signal
from scipy.cluster.hierarchy import dendrogram, linkage
from Bio.PDB import PDBList
from Bio.PDB.PDBParser import PDBParser
################################################################################
def residues_from_pdb(pdb_path):
    nodes = []
    structure = PDBParser(QUIET=True).get_structure('Protein', pdb_path)
    for chain in structure[0]:
        for residue in chain:
            if residue.id[0] == ' ':
                node = '{}:{}:{}'.format(chain.id if chain.id != ' ' else '_', residue.id[1], residue.id[2] if residue.id[2] != ' ' else '_')
                nodes.append(node)
    return nodes
################################################################################
def edges_filenames(path):
    files = os.listdir(path)
    def atoi(text):
        return int(text) if text.isdigit() else text
    def natural_keys(text):
        return [atoi(c) for c in re.split(r'(\d+)', text)]
    files.sort(key = natural_keys)
    return files
################################################################################
def edges_file_saving(filenames_list, path):    
    edf = []
    for i in filenames_list:
        p = path + '/' + i
        edf.append(pd.read_csv(p, sep = '\t').loc[:, ('NodeId1', 'Interaction', 'NodeId2')])
    n = len(edf)
    for i in range(n):
        edf[i]['Interaction'] = edf[i]['Interaction'].replace(r'\:.*', '', regex = True)            
        for index, row in edf[i].iterrows():
            row['NodeId1'] = row['NodeId1'][::-1].split(':', 1)[1][::-1]
            row['NodeId2'] = row['NodeId2'][::-1].split(':', 1)[1][::-1]
    return edf
################################################################################
def interaction(edges_list):   
    itn = []
    for i in edges_list:
        for j in i['Interaction']:
            itn.append(j)
    itn_list = sorted(list(set(itn)))
    dict_itn = {}
    dict_itn.fromkeys(itn_list, 0)
    for i in itn_list:
        dict_itn[i] = itn.count(i)
    return dict_itn
################################################################################
def weight(interactions_list, N):
    tf_idf = []
    for i in interactions_list:
        tf = (interactions_list[i]/N)/(sum(interactions_list.values())/N)
        idf = np.log(N/(N*(interactions_list[i]/sum(interactions_list.values()))))
        tf_idf.append(tf*idf)
    return tf_idf
################################################################################
def split(edges, residues, interaction):
    n = len(residues)
    matrix = pd.DataFrame(np.zeros((n, n)), index = residues, columns = residues)
    for i in range(edges.shape[0]):
        j = edges["Interaction"][i]
        if( (j == interaction) and (j != 'HBOND') ):
            n1, n2 = edges['NodeId1'][i], edges['NodeId2'][i]
            matrix[n1][n2] = 1  
        else:
            distance = np.abs(int(re.findall('\d+', edges['NodeId1'][i])[0]) - 
                              int(re.findall('\d+', edges['NodeId2'][i])[0]))
            if(distance >= 11):
                n1, n2 = edges['NodeId1'][i], edges['NodeId2'][i]
                matrix[n1][n2] = 1   
    return matrix

def contact(edges, residues, interaction):
    l = []
    for i in interaction:
        l.append(split(edges, residues, i))
    return l

def contact_saving(edges_list, residues, interaction):
    l = []
    for i in edges_list:
        l.append(contact(i, residues, interaction))
    return l

def distance_split_contact(contacts_list, interaction):
    n = len(contacts_list)
    m = np.zeros((n,n))
    
    def distance(A, B):
        a = np.reshape(A.to_numpy(), -1)
        b = np.reshape(B.to_numpy(), -1)
        return scipy.spatial.distance.hamming(a,b)*1000
    
    for i in range(n):
        d1 = contacts_list[i][interaction]
        for j in range(n):
            d2 = contacts_list[j][interaction]
            m[i][j] = distance(d1, d2)
    return m
################################################################################
def opt_n_clust(dmat, linked):
    n_clust = range(2, int(np.sqrt(dmat.shape[0])))
    score = []
    for i in n_clust:
        labels = []
        clust_labels = scipy.cluster.hierarchy.cut_tree(linked, n_clusters = i)
        for j in clust_labels:
            labels.append(int(j))
        score.append(metrics.silhouette_score(dmat, labels, metric = 'precomputed'))
    return np.array(score).argmax() + 2
################################################################################   
def most_important_snapshot(dmat, linked, k):
    dmat = pd.DataFrame(dmat)
    labels, snapshots = [], []
    for i in scipy.cluster.hierarchy.cut_tree(linked, n_clusters = k):
        labels.append(int(i))
    for i in range(k):
        lab = pd.DataFrame(labels)
        lab = lab[lab[0] == i].index.values
        silhouette_mat = pd.DataFrame(metrics.silhouette_samples(dmat, labels, metric = 'precomputed'))
        silhouette_max = silhouette_mat.loc[lab, 0].idxmax()
        snapshots.append(silhouette_max)
    return labels, snapshots
################################################################################
def relevant_interaction(snapshot, edges_list):
    df = pd.DataFrame(columns=['FromSnapshot', 'ToSnapshot','FromCluster', 'ToCluster', 'NodeId1', 'Interaction', 'NodeId2'])
    l = []
    for i in snapshot:
        l.append(edges_list[i])
    for i in range(len(l)-1):
        for j in range(i+1, len(l)):
            df1 = l[i]
            df2 = l[j]
            dfm = pd.concat([df1, df2]).iloc[:,0:3]
            dfm = dfm[ (dfm["Interaction"] != "VDW") ].drop_duplicates(keep = False, ignore_index = True)
            for k in range(dfm.shape[0]):
                values_to_add = {'FromSnapshot' : snapshot[i],
                                 'ToSnapshot' : snapshot[j],
                                 'FromCluster' : i,
                                 'ToCluster' : j,
                                 'NodeId1': dfm.loc[k,'NodeId1'],
                                 'Interaction' : dfm.loc[k,'Interaction'],
                                 'NodeId2' : dfm.loc[k,'NodeId2']}
                row_to_add = pd.Series(values_to_add)
                df = df.append(row_to_add, ignore_index = True)
    return df   
################################################################################
def output_info(matrix, representative_list):
    matrix = pd.DataFrame(matrix)
    labels = representative_list[0]
    info = pd.DataFrame(np.zeros((3,len(labels))), index = ['ClusterLabels', 'ClusterRep', 'DistanceFromCenter'])
    info.loc['ClusterLabels',:] = labels
    info.loc['ClusterRep', representative_list[1]] = 1
    labels = pd.DataFrame(labels)
    for i in range(len(representative_list[1])):
        val = matrix.loc[representative_list[1][i], labels[labels[0] == i].index.values]        
        info.loc['DistanceFromCenter',  labels[labels[0] == i].index.values] = (val - np.min(val))/(np.max(val) - np.min(val))
    return info   
################################################################################
def main():
    p1, p2, p3 = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(p3 + '/' + 'output')
    
    residues_list = residues_from_pdb(p1)
    filenames_list = edges_filenames(p2)
    edges_list = edges_file_saving(filenames_list, p2)
    interaction_list = interaction(edges_list)
    w = weight(interaction_list, len(filenames_list))
    
    ###############################################################
    print('\n')
    print('############### INFORMATION ###############')
    print(' - Number of residues: {}'.format(len(residues_list)))
    print(' - Number of snapshots: {}'.format(len(filenames_list)))
    print(' - Types of interaction: {}'.format(len(interaction_list)))
    ###############################################################
    
    contacts_list = contact_saving(edges_list, residues_list, interaction_list)
    l = []
    for i in range(len(interaction_list)):
        l.append(distance_split_contact(contacts_list, i))
    for i in range(len(l)):
        np.nan_to_num(l[i], copy= False)
    dmat = sum([w[i]*l[i] for i in range(len(w))])
    X = ssd.squareform(dmat)
    linked = linkage(X, method = 'average')

    ###############################################################
    k = opt_n_clust(dmat, linked)
    print(' - Number of clusters: {}'.format(k))
    ###############################################################
    
    ###############################################################
    representative_list = most_important_snapshot(dmat, linked, k)
    print(' - Representative snapshots: {}'.format(sorted(representative_list[1])))
    print("###########################################")
    ###############################################################
    
    relevant_interaction_list = relevant_interaction(representative_list[1], edges_list)
    for i in range(relevant_interaction_list.shape[0]):
    	distance = np.abs(int(re.findall('\d+', relevant_interaction_list.loc[i,'NodeId1'])[0]) -
                          int(re.findall('\d+', relevant_interaction_list.loc[i,'NodeId2'])[0]))
    	if(relevant_interaction_list.loc[i,'Interaction'] == 'HBOND' and distance < 11):
            relevant_interaction_list.drop(labels = i, inplace = True)
    relevant_interaction_list.reset_index(drop = True, inplace = True)
    info = output_info(dmat, representative_list)

    ###############################################################
    ###############################################################
    ###############################################################
    pd.DataFrame(dmat).to_csv(p3 + '/' + 'output/distance.txt', index = False)
    pd.DataFrame(info).to_csv(p3 + '/' + 'output/info.txt')
    pd.DataFrame(relevant_interaction_list).to_csv(p3 + '/' + 'output/interaction.txt')
    plt.figure(figsize=(24,12))
    dend = dendrogram(linked, 
                      leaf_rotation=90, 
                      leaf_font_size=10,  
                      color_threshold=linked[-k+1, 2],
                      above_threshold_color='grey')
    plt.title('Hierarchical Clustering Dendrogram', fontsize = 20)
    plt.xlabel('Snapshot', fontsize = 15)
    plt.ylabel('Distance', fontsize = 15)
    plt.savefig(p3 + '/' + 'output/dend.jpg')
    ###############################################################
    ###############################################################
    ###############################################################
    
main()














































































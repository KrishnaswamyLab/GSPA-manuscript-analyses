"""
Evaluate gene embeddings for GSPA and comparisons based on anti-correlation between gene embedding distance and true gene-gene coexpression. Stratified for library size (due to differences in correlation based on library size) and true coexpression (due to large imbalance and majority of gene pairs not coexpressed).
"""

import numpy as np
from collections import defaultdict
import sys, os
from scipy.stats import spearmanr

model = sys.argv[1]
dataset = sys.argv[2]

if dataset == '2_branches':
    datafile = 'splatter_simulated_data_2_branches.npz'
    extension = '_2_branches'
elif dataset == '3_branches':
    datafile = 'splatter_simulated_data_3_branches.npz'
    extension = '_3_branches'
elif dataset == 'linear':
    datafile = 'splatter_simulated_data.npz'
    extension = ''

# confirm model choice
if model not in ['Eigenscore', 'GFMMD', 'Signals', 'DiffusionEMD', 'GSPA', 'GSPA_QR', 'MAGIC', 'Node2Vec_Gcell', 'GAE_noatt_Gcell', 'GAE_att_Gcell', 'Node2Vec_Ggene', 'GAE_noatt_Ggene', 'GAE_att_Ggene', 'SIMBA', 'siVAE']:
    sys.exit('Model choice not in [Eigenscore GFMMD Signals DiffusionEMD GSPA GSPA_QR MAGIC Node2Vec_Gcell GAE_noatt_Gcell GAE_att_Gcell Node2Vec_Ggene GAE_noatt_Ggene GAE_att_Ggene SIMBA siVAE]')

trajectory_data = np.load(f'../data/{datafile}')
data = trajectory_data['data']
true_counts = trajectory_data['true_counts']
true_lib_size = true_counts.T.sum(axis=1)

# get coexpression embeddings
coexpression_results = {}

for id in [7, 8, 9]:
    run = f'results/{model}/{id}_results{extension}.npz'
    res = np.load(run, allow_pickle=True)
    name = res['config'][()]['save_as']
    coexpression_results[name] = res['signal_embedding']
    
print ('Stratify Spearman correlation...')
spearman_res = spearmanr(true_counts)
np.fill_diagonal(spearman_res.correlation, 0)
corr_bins = np.linspace(spearman_res.correlation.min(), spearman_res.correlation.max(), 4)

min_bin_size = float('inf')
for i,corr in enumerate(corr_bins):
    if i == 0: continue
    choices = np.array(np.where((spearman_res.correlation > corr_bins[i-1]) & (spearman_res.correlation < corr) & (spearman_res.correlation != 0))).T
    if choices.shape[0] < min_bin_size:
        min_bin_size = choices.shape[0]

spearmans = defaultdict(list)

print ('Stratify library size...')
min_lib_size= float('inf')
for i,corr in enumerate(corr_bins):

    choices_bin = []
    if i == 0: continue

    ## res.correlation does not equal zero, excluding self edges
    choices = np.array(np.where((spearman_res.correlation > corr_bins[i-1]) & (spearman_res.correlation < corr) & (spearman_res.correlation != 0))).T
    
    lib_size_mean_per_pair = np.vstack((true_lib_size[choices[:, 0]], true_lib_size[choices[:, 1]])).mean(axis=0)
    lib_size = np.linspace(lib_size_mean_per_pair.min(), lib_size_mean_per_pair.max(), 3)

    for j,bin in enumerate(lib_size):
        if j == 0: continue
        choices_ = np.array(np.where((lib_size_mean_per_pair > lib_size[j-1]) & (lib_size_mean_per_pair < bin))).T
        if choices_.shape[0] < min_lib_size:
            min_lib_size = choices_.shape[0]

print ('Sample by stratification...')
samples = []
for i,corr in enumerate(corr_bins):

    choices_bin = []
    if i == 0: continue

    ## res.correlation does not equal zero, excluding self edges
    choices = np.array(np.where((spearman_res.correlation > corr_bins[i-1]) & (spearman_res.correlation < corr) & (spearman_res.correlation != 0))).T
    
    lib_size_mean_per_pair = np.vstack((true_lib_size[choices[:, 0]], true_lib_size[choices[:, 1]])).mean(axis=0)
    lib_size = np.linspace(lib_size_mean_per_pair.min(), lib_size_mean_per_pair.max(), 3)

    for j,bin in enumerate(lib_size):
        if j == 0: continue
        choices_ = np.array(np.where((lib_size_mean_per_pair > lib_size[j-1]) & (lib_size_mean_per_pair < bin))).T
        choices_bin.append(choices_[np.random.choice(choices_.shape[0], size=min_lib_size, replace=False, )])

    samples.append(choices[np.vstack(choices_bin).flatten()])

samples = np.vstack(samples)

f = open(f"results/{model}/spearmanr{extension}_demap_789.txt", "w")

for (name, embedding) in coexpression_results.items():
    X= []
    y=[]

    for (a,b) in samples:
        X.append(np.linalg.norm(embedding[a] - embedding[b]))
        y.append(spearman_res.correlation[a][b])
    
    X = np.array(X)
    y = np.array(y)

    spearmans[name].append(spearmanr(X, y).correlation)
        
    f.write(f'{name} Spearman {np.median(spearmans[name])}\n')

f.close()

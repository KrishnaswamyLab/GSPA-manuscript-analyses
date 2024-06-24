import numpy as np
from collections import defaultdict
import os, sys, glob
from scipy.stats import spearmanr

model = sys.argv[1]
dataset = sys.argv[2]

if dataset == '2_branches':
    extension = '_2_branches'
if dataset == 'sparse_branches':
    extension = '_sparse_branches'
if dataset == '3_branches':
    extension = '_3_branches'
elif dataset == 'linear':
    extension = ''

# confirm model choice
if model not in ['SIMBA', 'siVAE', 'Eigenscore','GFMMD', 'Signals', 'DiffusionEMD', 'GSPA', 'GSPA_QR', 'MAGIC', 'Node2Vec_Ggene', 'GAE_noatt_Ggene', 'GAE_att_Ggene']:
    sys.exit('Model choice not in [SIMBA siVAE Eigenscore GFMMD Signals DiffusionEMD GSPA GSPA_QR MAGIC Node2Vec_Ggene GAE_noatt_Ggene GAE_att_Ggene]')

spearmans = defaultdict(list)
labels_y = np.load(f'../data/localization_signals{extension}.npz')['spread']

# get embeddings
localization_scores = {}

for id in [7, 8, 9]:
    run = f'results/{model}/{id}_results{extension}.npz'
    res = np.load(run, allow_pickle=True)
    name = res['config'][()]['save_as']
    localization_scores[name] = res['localization_score']
    
f = open(f'results/{model}/spearmanr{extension}_789.txt', 'w')

for (name, score) in localization_scores.items():
    spearmans[name] = spearmanr(score, labels_y).correlation
    f.write(f'{name} Spearman {spearmans[name]}\n')
    
f.close()
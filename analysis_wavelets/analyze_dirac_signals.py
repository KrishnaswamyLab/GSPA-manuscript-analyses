import phate, scipy, keras, magic, sys
import numpy as np
from keras import layers
from sklearn.model_selection import RepeatedKFold
from sklearn import linear_model, decomposition
from scipy.stats import spearmanr
from collections import defaultdict

run = sys.argv[1]

datasets = {}

print ('Load data...')
trajectory_data = np.load('../data/splatter_simulated_data.npz')
data_libnorm_sqrt = trajectory_data['data']
pseudotime = trajectory_data['pseudotime']
data_libnorm_sqrt= data_libnorm_sqrt[np.argsort(pseudotime)]
pseudotime = np.array(list(range(10000))) / 10000    

print ('Compute PHATE...')
phate_op = phate.PHATE(random_state=0, use_pygsp=True, verbose=0, n_jobs=1)
data_phate = phate_op.fit_transform(data_libnorm_sqrt)
G = phate_op.graph
del(phate_op)

gspa_op = gspa.GSPA(graph=G, qr_decompose=False)
gspa_op.build_diffusion_operator()
gspa_op.build_wavelet_dictionary()
datasets['GSPA'] = gspa_op.wavelet_dictionary

gspa_op = gspa.GSPA(graph=G, qr_decompose=True)
gspa_op.build_diffusion_operator()
gspa_op.build_wavelet_dictionary()
datasets['GSPA_QR'] = gspa_op.wavelet_dictionary

all_signals = np.eye(10000)

print ('Compute MAGIC...')
magic_op = magic.MAGIC(verbose=False, n_jobs=8)
magic_op.graph = G
datasets['MAGIC'] = magic_op.transform(all_signals.T).T

dirac_embeddings = defaultdict(dict)
f = open(f"results/spearmanr_{run}.txt", "a")

for spacing in [int(2**i) for i in range(1,11)][::-1]:
    datasets_curr_run = {**datasets}
    
    for (name, signals) in datasets_curr_run.items():
        print (f'{name} Spacing {spacing}')
    
        index = np.array(list(range(run, 10000-run, spacing)))
        labels_y = pseudotime[index]
    
        all_signals_subsampled = signals[index]
        signals_reduced = gspa.embedding.svd(all_signals_subsampled)
        data_ae = gspa.embedding.run_ae(signals_reduced)
    
        if spacing == 64:
            np.save(f'results/{name}_{spacing}.npy')
    
        kf = RepeatedKFold(n_splits=2, n_repeats=20)
        splits = kf.split(data_ae)
    
        for (train_index, test_index) in splits:
            X_train = data_ae[train_index]
            X_test = data_ae[test_index]
            y_train = labels_y[train_index]
            y_test = labels_y[test_index]
    
            regr = linear_model.Ridge()
            regr.fit(X_train, y_train)
            y_pred = regr.predict(X_test)
    
            spearmans[name].append(spearmanr(y_test, y_pred).correlation)
    
        f.write(f'{spacing} {name} Spearman {np.median(spearmans[name])}\n')
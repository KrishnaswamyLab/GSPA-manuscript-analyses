import sys
sys.path.append('..')
from run.run_gspa import calculate_wavelet_dictionary
from run.run_ae_default_config import run_ae
from run.gspa_helper import *
import leidenalg
import scanpy, phate, meld
import numpy as np
import scprep
import matplotlib.pyplot as plt
import pandas as pd
from utils import *
import sklearn
from sklearn.preprocessing import scale
import magic
import time

qr_decompose = sys.argv[1] == 'True'
perform_condensation = sys.argv[2] == 'True'
if perform_condensation:
    prefix = 'exact'
else:
    prefix = 'fast'

acute = scanpy.read_h5ad('data/acute_tcells.h5ad')
chronic = scanpy.read_h5ad('data/chronic_tcells.h5ad')
adata = scanpy.concat((acute,chronic))
adata.obs['batch'] = adata.obs['batch'].astype('category')

start= time.time()
phate_op = phate.PHATE(random_state=42, n_jobs=-1, knn=30)
adata.obsm['X_phate'] = phate_op.fit_transform(adata.to_df())
end = time.time()
print ('Construct graph', end - start)

G_without_regression = phate_op.graph.to_pygsp()
data, data_hvgs = scprep.select.highly_variable_genes(adata.to_df(), adata.var_names, percentile=90)
data_hvg = data[data_hvgs]

gspa_op = gspa.GSPA(graph=phate_op.graph, qr_decompose=qr_decompose, perform_condensation=perform_condensation)
gspa_op.build_diffusion_operator()
gspa_op.build_wavelet_dictionary()

# Embed gene signals from wavelet dictionary
gene_signals = data_hvg.T # embed highly variable genes
gene_ae, gene_pc = gspa_op.get_gene_embeddings(gene_signals)
gene_localization = gspa_op.calculate_localization()

np.savez(f'./results/{prefix}_{qr_decompose}_signal_embedding.npz', signal_embedding=gene_ae,
         localization_score=gene_localization, genes=data_hvgs.values)
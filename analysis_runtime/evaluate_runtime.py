import sys
import numpy as np
import pandas as pd
import time
import gpsa

count = int(sys.argv[1])
qr_decompose = sys.argv[2] == 'True'

data = np.load(f'../data/large_splatter_simulated_data.npz')
counts = pd.DataFrame(data['data']).sample(count).values

start = time.time()
gspa_op = gspa.GSPA(qr_decompose=qr_decompose)
gspa_op.construct_graph(counts)
end = time.time()
print ('Construct graph', end - start)

start = time.time()
gspa_op.build_diffusion_operator()
gspa_op.build_wavelet_dictionary()
end = time.time()
print ('Wavelet dictionary', end - start)

start = time.time()
gene_signals = data.T # embed all measured genes
gene_ae, gene_pc = gspa_op.get_gene_embeddings(gene_signals)
gene_localization = gspa_op.calculate_localization()
end = time.time()
print ('Gene embedding', end - start)
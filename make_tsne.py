#!/usr/local/bin/python2.7

from sklearn.preprocessing import StandardScaler
from tsne import bh_sne
import numpy as np
import sys

data_dir = './output/features'

names = sys.argv[1:]
print(names)

for name in names:
    X_svd = np.load(name+'.mat')
    X_scaled = StandardScaler().fit_transform(X_svd)
    X_tsne = bh_sne(X_scaled)
    X_tsne.dump(data_dir+'/'+'tSNE_'+name+'.mat')

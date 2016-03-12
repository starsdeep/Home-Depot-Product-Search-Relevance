#def get_synomy_from_query():
##

## 近义词 和 关联词 可以分开做
from nltk.corpus import wordnet as wn
from load_data import load_data
from time import time
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.metrics import pairwise
#wn.synsets('dog')

t0 = time()
print("start in %fs" % t0)


N_line = 100 #optional 100 for test
df, num_train, num_test = load_data(N_line)


print("load data done in %fs" % (time() - t0))

#Format into matrix X

print("format data X done in %fs" % (time() - t0))


#
svd = TruncatedSVD(n_components=30)
normalizer = Normalizer(copy = False)
lsa = make_pipeline(svd, normalizer)

X = lsa.fit_transform(X)

print("SVD done in %fs" % (time() - t0))

D = pairwise.pairwise_distances(X) #D_{i,j} is the distance of i_th and j_th in X

#两层循环需要的内存空间更少,可能更快
#for i in X..
#for j in X..
	

def build_synomy_library(word):
	pass


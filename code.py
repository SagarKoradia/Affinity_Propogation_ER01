from numpy import genfromtxt
from sklearn.cluster import AffinityPropagation

fn = r'C:\Users\DELL I5558\Desktop\Python\NSW.csv'
my_data = genfromtxt(fn, delimiter=',')

af = AffinityPropagation(damping=0.6, max_iter=2000, convergence_iter=1000, copy=True, preference=None, affinity='euclidean', verbose=False).fit(my_data)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_

n_clusters_ = len(cluster_centers_indices)

print(n_clusters_)

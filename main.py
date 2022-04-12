import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from sklearn.utils import shuffle
from sklearn.preprocessing import scale
from sklearn.manifold import TSNE

from sklearn.manifold import MDS
from sklearn.metrics.pairwise import pairwise_distances

from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import KMeans

from itertools import cycle

df = pd.read_csv('data/features.csv', nrows=1000000, index_col=0)

# # Basic analysis for data
# print('\nDataFrame:', '\n', df)
# print('\nDataFrame columns: ', '\n', df.columns)
# print('\nDataFrame info:', '\n', df.info(verbose=True))
# print('\nDataFrame description:', '\n', df.describe())
# print('\nDataFrame shape:', '\n', df.shape)
#
# # Correlation for data
# sns.heatmap(df[df.columns[1:]].corr(), square=True)
# plt.plot()
# print('\nCorrelation:', '\n', df[df.columns[1:]].corr() > 0.9)


# Reduction to 2D
sdata = shuffle(df, random_state=321)
subset_l = 1000 # number of rows
selected_features = df.columns[1:]
objects_with_nan = sdata.index[np.any(np.isnan(sdata[selected_features].values), axis=1)]
data_subset = scale(sdata[selected_features].drop(objects_with_nan, axis=0)[:subset_l])
response_subset = sdata["out_parties_n"].drop(objects_with_nan, axis=0)[:subset_l]


# tSNE
tsne = TSNE(random_state=321)
tsne_representation = tsne.fit_transform(data_subset)

colors = cm.rainbow(np.linspace(0, 1, len(set(response_subset))))
for y, c in zip(set(df.index), colors):
    plt.scatter(tsne_representation[response_subset.values==y, 0],
                tsne_representation[response_subset.values==y, 1],
                color=c, alpha=0.5, label=str(y))

# palette = sns.color_palette("bright", 10)
# sns.scatterplot(tsne_representation[:,0], tsne_representation[:,1], legend='full', palette=palette)

plt.title("T-SNE for %d rows" % subset_l)
plt.savefig('graphs/tsne_%d.png' % subset_l)
plt.plot()
plt.show()



# # MDS
# mds = MDS(random_state = 321)
# MDS_transformed = mds.fit_transform(data_subset)
#
# colors = cm.rainbow(np.linspace(0, 1, len(set(response_subset))))
# for y, c in zip(set(response_subset), colors):
#     plt.scatter(MDS_transformed[response_subset.values==y, 0],
#                 MDS_transformed[response_subset.values==y, 1],
#                 color=c, alpha=0.5, label=str(y))
#
# # plt.xlim(-5, 5)
# # plt.ylim(-5, 5)  # -
#
# plt.title("MDS for %d rows" % subset_l)
# plt.savefig('graphs/mds_%d.png' % subset_l)
# plt.plot()



# # MDS with cos
# data_subset_cosine = pairwise_distances(data_subset, metric='cosine')
# MDS_transformed_cos = MDS(dissimilarity='precomputed',random_state = 321).fit_transform(data_subset_cosine)
#
# colors = cm.rainbow(np.linspace(0, 1, len(set(response_subset))))
# for y, c in zip(set(response_subset), colors):
#     plt.scatter(MDS_transformed_cos[response_subset.values[:subset_l]==y, 0],
#                 MDS_transformed_cos[response_subset.values[:subset_l]==y, 1],
#                 color=c, alpha=0.5, label=str(y))
# plt.legend(
# plt.title("MDS_cosine for %d rows" % subset_l)
# plt.savefig('graphs/mds_cosine_%d.png' % subset_l)
# plt.plot()



# Cluster algorithms

# MeanShift
bandwidth = estimate_bandwidth(tsne_representation, quantile=0.2, n_samples=subset_l)
model = MeanShift(bandwidth=bandwidth)
clustering = model.fit(tsne_representation)

labels = model.labels_
cluster_centers = model.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("Number of estimated clusters : %d" % n_clusters_)

# # K-Means
# # To decide how many clusters to choose for K-Means

# inertia = []
# for k in range(1, 30):
#     kmeans = KMeans(n_clusters=k, random_state=1).fit(tsne_representation)
#     inertia.append(np.sqrt(kmeans.inertia_))
#
# plt.plot(range(1, 30), inertia, marker='s')
# plt.xlabel('$k$')
# plt.ylabel('$J(C_k)$')
# plt.savefig('graphs/k_for_%d.png' % subset_l)
# plt.show()

# Try 4 clusters for 1000 rows
# kmeans = KMeans(n_clusters=4, random_state=1).fit(tsne_representation)
# labels = kmeans.labels_
# cluster_centers = kmeans.cluster_centers_
#
# labels_unique = np.unique(labels)
# n_clusters_ = len(labels_unique)
#
# print("number of estimated clusters : %d" % n_clusters_)



# Plot clusters
plt.figure(1)
plt.clf()

colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(tsne_representation[my_members, 0], tsne_representation[my_members, 1], col + ".")
    plt.plot(
        cluster_center[0],
        cluster_center[1],
        "o",
        markerfacecolor=col,
        markeredgecolor="k",
        markersize=14,
    )
plt.title("Estimated number of clusters: %d" % n_clusters_)
plt.savefig('graphs/clusters_for_%d.png' % subset_l)
plt.plot()
plt.show()
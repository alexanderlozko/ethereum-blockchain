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


# Reduction to 2D
def data_preparation(df, random_state=321, sample=1000):

    # Create new two features: in_value_range, out_value_range
    df['in_value_range'] = df['in_value_max'] - df['in_value_min']
    df['out_value_range'] = df['out_value_max'] - df['out_value_min']

    subset_l = sample # number of rows
    sdata = shuffle(df, random_state=random_state)
    selected_features = df.columns[1:]
    objects_with_nan = sdata.index[np.any(np.isnan(sdata[selected_features].values), axis=1)]
    drop = sdata[selected_features].drop(objects_with_nan, axis=0)[:subset_l]
    data_subset = scale(drop)
    response_subset = sdata["out_parties_n"].drop(objects_with_nan, axis=0)[:subset_l]

    return drop, data_subset, response_subset

# tSNE
def tsne(data_subset, random_state=321):
    tsne = TSNE(random_state=random_state)
    tsne_representation = tsne.fit_transform(data_subset)

    return tsne_representation

def plot_tsne(tsne_representation, response_subset):
    colors = cm.rainbow(np.linspace(0, 1, len(set(response_subset))))
    for y, c in zip(set(df.index), colors):
        plt.scatter(tsne_representation[response_subset.values == y, 0],
                    tsne_representation[response_subset.values == y, 1],
                    color=c, alpha=0.5, label=str(y))

    # palette = sns.color_palette("bright", 10)
    # sns.scatterplot(tsne_representation[:, 0], tsne_representation[:, 1], legend='full', palette=palette)

    plt.title("T-SNE for %d rows" % len(tsne_representation))
    plt.savefig('graphs/tsne_%d.png' % len(tsne_representation))
    plt.plot()
    plt.show()

# MDS
def mds(data_subset, random_state=321):
    mds = MDS(random_state = random_state)
    MDS_transformed = mds.fit_transform(data_subset)

    return MDS_transformed

def plot_mds(MDS_transformed, response_subset):
    colors = cm.rainbow(np.linspace(0, 1, len(set(response_subset))))
    for y, c in zip(set(response_subset), colors):
        plt.scatter(MDS_transformed[response_subset.values==y, 0],
                    MDS_transformed[response_subset.values==y, 1],
                    color=c, alpha=0.5, label=str(y))

    # plt.xlim(-5, 5)
    # plt.ylim(-5, 5)  # -

    plt.title("MDS for %d rows" % len(MDS_transformed))
    plt.savefig('graphs/mds_%d.png' % len(MDS_transformed))
    plt.plot()
    plt.show()

# MDS with cos
def mds_cos(data_subset, random_state=321):
    data_subset_cosine = pairwise_distances(data_subset, metric='cosine')
    MDS_transformed_cos = MDS(dissimilarity='precomputed',random_state = random_state).fit_transform(data_subset_cosine)

    return MDS_transformed_cos

def plot_mds_cos(MDS_transformed_cos, response_subset):
    colors = cm.rainbow(np.linspace(0, 1, len(set(response_subset))))
    for y, c in zip(set(response_subset), colors):
        plt.scatter(MDS_transformed_cos[response_subset.values==y, 0],
                    MDS_transformed_cos[response_subset.values==y, 1],
                    color=c, alpha=0.5, label=str(y))
    plt.legend()
    plt.title("MDS_cosine for %d rows" % len(MDS_transformed_cos))
    plt.savefig('graphs/mds_cosine_%d.png' % len(MDS_transformed_cos))
    plt.plot()
    plt.show()


# Cluster algorithms

# MeanShift
def meanshift(data_representation, subset_l=1000):

    bandwidth = estimate_bandwidth(data_representation, quantile=0.2, n_samples=subset_l)
    model = MeanShift(bandwidth=bandwidth)
    clustering = model.fit(data_representation)

    labels = model.labels_
    cluster_centers = model.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    print("Number of estimated clusters : %d" % n_clusters_)

    return clustering, cluster_centers, labels, n_clusters_

# K-Means
# To decide how many clusters to choose for K-Means

def cluster_kmeans(data_representation):
    inertia = []
    for k in range(1, 30):
        kmeans = KMeans(n_clusters=k, random_state=1).fit(data_representation)
        inertia.append(np.sqrt(kmeans.inertia_))

    plt.plot(range(1, 30), inertia, marker='s')
    plt.xlabel('$k$')
    plt.ylabel('$J(C_k)$')
    plt.savefig('graphs/k_for_%d.png' % len(data_representation))
    plt.show()

# Try 4 clusters for 1000 rows
def kmeans(data_representation, n_clusters=4, random_state=321):

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    clustering = kmeans.fit(data_representation)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    print("number of estimated clusters : %d" % n_clusters_)

    return clustering, cluster_centers, labels, n_clusters_


# Plot clusters
def plot_clusters(n_clusters_, labels, cluster_centers, data_representation):
    plt.figure(1)
    plt.clf()

    colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        plt.plot(data_representation[my_members, 0], data_representation[my_members, 1], col + ".")
        plt.plot(
            cluster_center[0],
            cluster_center[1],
            "o",
            markerfacecolor=col,
            markeredgecolor="k",
            markersize=14,
        )
    plt.title("Estimated number of clusters: %d" % n_clusters_)
    plt.savefig('graphs/clusters_for_%d.png' % len(data_representation))
    plt.plot()
    plt.show()


df = pd.read_csv('data/features.csv', nrows=1000000, index_col=0)

# Basic analysis for data
print('\nDataFrame:', '\n', df)
print('\nDataFrame columns: ', '\n', df.columns)
print('\nDataFrame info:', '\n', df.info(verbose=True))
print('\nDataFrame description:', '\n', df.describe())
print('\nDataFrame shape:', '\n', df.shape)

# Correlation for data
sns.heatmap(df[df.columns[1:]].corr(), square=True)
plt.plot()
plt.show()
print('\nCorrelation:', '\n', df[df.columns[1:]].corr() > 0.9)

sdata, data_subset, response_subset = data_preparation(df)
data_representation = tsne(data_subset)
plot_tsne(data_representation, response_subset)

clustering, cluster_centers, labels, n_clusters_ = meanshift(data_representation)
print('clustering: ', clustering)
print('cluster_centers: ', cluster_centers)
print('labels: ', labels)
print('len labels: ', len(labels))
print('n_clusters_: ', n_clusters_)
plot_clusters(n_clusters_, labels, cluster_centers, data_representation)





